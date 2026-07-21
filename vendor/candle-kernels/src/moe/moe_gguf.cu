/**
 * @brief CUDA kernel for Mixture-of-Experts (MoE) GEMM using GGUF quantized weights.
 *
 * This kernel performs a dot-product between quantized input tokens and
 * quantized expert weight matrices, accumulating into float outputs.
 * It supports per-token top-k weighting and tiling along the K dimension
 * for efficient vectorized execution.
 *
 * Adapted from: https://github.com/guoqingbao/attention.rs/tree/main/src/kernels/src/moe_gemm_gguf.cu
 */
#include "gguf.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <type_traits>
#include <cassert>
constexpr int MATRIX_ROW_PADDING = 512;

constexpr int pad(int size, int padding) {
    if (padding == 0) return size;  // avoid divide-by-zero
    return ((size + padding - 1) / padding) * padding;
}

// Optional helper if you want ceil division explicitly
constexpr int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

namespace vllm_rs {

/*
* Template Parameters:
 * @tparam T                 Type of output elements (float, half, etc.)
 * @tparam qk                Quantization block size for weights (e.g., 32)
 * @tparam qi                Quantization block size for inputs (e.g., 32)
 * @tparam block_q_t         Type of quantized weight block (e.g., block_q8_0)
 * @tparam vdr               Vectorization factor (number of elements per lane)
 * @tparam vec_dot_q_cuda    Function for computing vectorized dot-product between quantized blocks
 *
 * Kernel Parameters:
 * @param all_weights         Pointer to all expert weight matrices, [num_experts, N, K] (quantized)
 * @param all_inputs          Pointer to all input tokens, [M_total, K] (quantized)
 * @param sorted_token_ids    Sorted token indices for batch processing
 * @param expert_ids          Expert ID for each token
 * @param topk_weights        Optional top-k MoE weight per token
 * @param all_outputs         Output buffer [M_total, N] (float)
 * @param num_experts         Number of experts
 * @param topk                Top-k experts selected per token
 * @param size_m              Number of tokens processed (M dimension)
 * @param size_n              Output feature dimension (N dimension)
 * @param size_k              Input feature dimension (K dimension)
 * @param k_padded            Padded K dimension for GGUF stride
*/
template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda>
__global__ void moe_gemm_gguf_kernel(
    const void * __restrict__ all_weights,       // [num_experts, N, K] (quantized)
    const void * __restrict__ all_inputs,        // [M_total, K] (quantized, M_total is total tokens)
    const int32_t* __restrict__ sorted_token_ids,// [M] (M = num tokens processed)
    const int32_t* __restrict__ expert_ids,      // [M]
    const float* __restrict__ topk_weights,      // [M]
    float * __restrict__ all_outputs,            // [M_total, N] (float)
    int num_experts,
    int topk,
    int size_m, int size_n, int size_k, // M, N, K are the logical dims
    int k_padded // Padded K-dim for GGUF stride
) {
    const int laneId = threadIdx.x;
    const int wrapId = threadIdx.y;
    const int nWraps = blockDim.y;
    const int row = blockIdx.x * nWraps + wrapId; // This is the 'n' dimension (output row)
    const int m_idx = blockIdx.y; // This is the 'm' dimension (token index)
    
    // This block computes the dot product for `output[token_id][n_row]`
    
    if (row >= size_n || m_idx >= size_m) {
        return;
    }

    // strides
    const size_t weight_expert_stride_bytes = (size_t)(size_n * size_k) / qk * sizeof(block_q_t);
    const size_t input_task_stride_bytes    = (size_t)k_padded / QK8_1 * sizeof(block_q8_1);
    const size_t output_task_stride_elems   = (size_t)size_n;

    const int token_id = sorted_token_ids[m_idx]; // The *actual* row in input/output tensors
    const int expert = expert_ids[m_idx];
    
    // If expert is invalid, this token does not participate.
    if (expert < 0 || expert >= num_experts) return;

    // Get the scaling factor for this token/expert pair
    const float scale = (topk_weights) ? topk_weights[token_id] : 1.0f;

    const block_q_t * __restrict__ w_expert =
        (const block_q_t *)((const char *)all_weights + (size_t)expert * weight_expert_stride_bytes);

    const int input_index = topk_weights ? token_id : (token_id / topk);
    const block_q8_1 * __restrict__ y_ptr =
        (const block_q8_1 *)((const char *)all_inputs + (size_t)input_index * input_task_stride_bytes);

    // dot-product tiling along k
    const int blocks_per_row_x = size_k / qk;
    const int blocks_per_iter  = vdr * WARP_SIZE / qi; // no nwarps factor: one warp per batch item

    // accumulators for rows_per_block rows (usually 1)
    float acc = 0.0f;

    #pragma unroll
    for (int kbx = laneId / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1);
        const int kqs = vdr * (laneId % (qi / vdr));
        acc += vec_dot_q_cuda(
            &w_expert[kbx + row * blocks_per_row_x],
            &y_ptr[kby],
            kqs);
    }

    float v = warp_reduce_sum(acc) * scale;
    if (laneId == 0) {
        float * __restrict__ out_ptr =
            all_outputs + ((size_t)token_id) * output_task_stride_elems;
        out_ptr[row] = v;
    }
}

}

// Single-token MiniMax decode specialization. Gate and up use the same input
// and route list, so one launch covers both matrices after quantizing the
// activation once.
template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda>
__global__ void moe_gate_up_decode_kernel(
    const void *__restrict__ gate_weights,
    const void *__restrict__ up_weights,
    const block_q8_1 *__restrict__ inputs,
    const int32_t *__restrict__ sorted_token_ids,
    const int32_t *__restrict__ expert_ids,
    float *__restrict__ gate_outputs,
    float *__restrict__ up_outputs,
    int num_experts,
    int topk,
    int size_n,
    int size_k,
    int k_padded) {
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int row = blockIdx.x * blockDim.y + warp;
    const int route_index = blockIdx.y;
    if (row >= size_n || route_index >= topk) {
        return;
    }

    const int token_id = sorted_token_ids[route_index];
    const int expert = expert_ids[route_index];
    if (expert < 0 || expert >= num_experts) {
        return;
    }
    const bool is_up = blockIdx.z != 0;
    const void *all_weights = is_up ? up_weights : gate_weights;
    float *all_outputs = is_up ? up_outputs : gate_outputs;
    const size_t expert_stride =
        (size_t)size_n * size_k / qk * sizeof(block_q_t);
    const block_q_t *weight = reinterpret_cast<const block_q_t *>(
        static_cast<const char *>(all_weights) + (size_t)expert * expert_stride);
    const block_q8_1 *input = inputs
        + (size_t)(token_id / topk) * (k_padded / QK8_1);
    const int blocks_per_row = size_k / qk;
    const int blocks_per_iter = vdr * WARP_SIZE / qi;

    float acc = 0.0f;
#pragma unroll
    for (int kbx = lane / (qi / vdr); kbx < blocks_per_row; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1);
        const int kqs = vdr * (lane % (qi / vdr));
        acc += vec_dot_q_cuda(&weight[row * blocks_per_row + kbx], &input[kby], kqs);
    }
    const float value = warp_reduce_sum(acc);
    if (lane == 0) {
        all_outputs[(size_t)token_id * size_n + row] = value;
    }
}

__global__ void quantize_silu_mul_q8_1(
    const float *__restrict__ gate,
    const float *__restrict__ up,
    block_q8_1 *__restrict__ output,
    int rows,
    int size_k,
    int k_padded) {
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int row = blockIdx.y;
    if (ix >= k_padded || row >= rows) {
        return;
    }
    const int output_index = row * k_padded + ix;
    const int block = output_index / QK8_1;
    const int quant_index = output_index % QK8_1;
    float value = 0.0f;
    if (ix < size_k) {
        const float g = gate[(size_t)row * size_k + ix];
        value = up[(size_t)row * size_k + ix] * (g / (1.0f + expf(-g)));
    }
    float amax = warp_reduce_max(fabsf(value));
    float sum = warp_reduce_sum(value);
    const float scale = amax / 127.0f;
    output[block].qs[quant_index] =
        amax == 0.0f ? 0 : static_cast<int8_t>(roundf(value / scale));
    if (quant_index == 0) {
        reinterpret_cast<half &>(output[block].ds.x) = scale;
        reinterpret_cast<half &>(output[block].ds.y) = sum;
    }
}

// One block computes one final hidden row. Its eight warps process the routed
// experts and reproduce Candle's eight-element tree reduction in-block.
template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda>
__global__ void moe_down_decode_kernel(
    const void *__restrict__ all_weights,
    const block_q8_1 *__restrict__ inputs,
    const int32_t *__restrict__ sorted_token_ids,
    const int32_t *__restrict__ expert_ids,
    const float *__restrict__ topk_weights,
    float *__restrict__ output,
    int num_experts,
    int topk,
    int size_n,
    int size_k,
    int k_padded) {
    const int lane = threadIdx.x;
    const int route_index = threadIdx.y;
    const int row = blockIdx.x;
    const int token_id = sorted_token_ids[route_index];
    const int expert = expert_ids[route_index];
    const int blocks_per_row = size_k / qk;
    const int blocks_per_iter = vdr * WARP_SIZE / qi;
    const size_t expert_stride =
        (size_t)size_n * size_k / qk * sizeof(block_q_t);
    float acc = 0.0f;
    if (route_index < topk && expert >= 0 && expert < num_experts) {
        const block_q_t *weight = reinterpret_cast<const block_q_t *>(
            static_cast<const char *>(all_weights) + (size_t)expert * expert_stride);
        const block_q8_1 *input = inputs + (size_t)token_id * (k_padded / QK8_1);
#pragma unroll
        for (int kbx = lane / (qi / vdr); kbx < blocks_per_row; kbx += blocks_per_iter) {
            const int kby = kbx * (qk / QK8_1);
            const int kqs = vdr * (lane % (qi / vdr));
            acc += vec_dot_q_cuda(&weight[row * blocks_per_row + kbx], &input[kby], kqs);
        }
        acc = warp_reduce_sum(acc) * topk_weights[token_id];
    }
    __shared__ float route_values[8];
    if (lane == 0) {
        route_values[token_id] = acc;
    }
    __syncthreads();
    if (route_index == 0) {
        float sum = lane < 8 ? route_values[lane] : 0.0f;
        const float add4 = __shfl_down_sync(0xffffffff, sum, 4);
        if (lane < 4) {
            sum += add4;
        }
        const float add2 = __shfl_down_sync(0xffffffff, sum, 2);
        if (lane < 2) {
            sum += add2;
        }
        const float add1 = __shfl_down_sync(0xffffffff, sum, 1);
        if (lane == 0) {
            output[row] = sum + add1;
        }
    }
}

#define LAUNCH_MOE_GGUF(qk, qi, block_q_t, vdr, vec_dot_q_cuda) \
    vllm_rs::moe_gemm_gguf_kernel<qk, qi, block_q_t, vdr, vec_dot_q_cuda> \
        <<<grid_dim, block_dim, 0, stream>>>(\
        weights, y_q8_1,\
        sorted_token_ids, expert_ids, topk_weights,\
        outputs,\
        num_experts, topk,\
        size_m, size_n, size_k,\
        kx_padded\
    );\


extern "C" void moe_gemm_gguf(
    const float* inputs, //must be float
    const void* weights,
    const int32_t* sorted_token_ids,
    const int32_t* expert_ids,
    const float* topk_weights,
    float* outputs,
    int num_experts,
    int topk,
    int size_m,         // M (num tokens to process)
    int size_n,         // N (output dim)
    int size_k,         // K (input dim)
    int quant_type,     // Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5,
    cudaStream_t stream
) {
    const int QUANTIZE_BLOCK_SIZE = CUDA_QUANTIZE_BLOCK_SIZE;
    const int kx_padded = pad(size_k, MATRIX_ROW_PADDING);
    const int num_blocks = ceil_div(kx_padded, QUANTIZE_BLOCK_SIZE);
    int m = topk_weights ? size_m : size_m / topk;
    dim3 grid_dim_quant(num_blocks, m, 1);
    dim3 block_dim_quant(QUANTIZE_BLOCK_SIZE, 1, 1);
    int y_size_in_bytes =
        m * (kx_padded / QK8_1 * sizeof(block_q8_1));
    void* y_q8_1 = nullptr;
    cudaMallocAsync(&y_q8_1, y_size_in_bytes, stream);
    quantize_q8_1<<<grid_dim_quant, block_dim_quant, 0, stream>>>(inputs, y_q8_1, size_k, kx_padded);

    const int nWraps = 2;
    dim3 grid_dim(ceil_div(size_n, nWraps), size_m, 1);
    dim3 block_dim(WARP_SIZE, nWraps, 1);

    //Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5,
    switch (quant_type) {
        case 0: // Q8_0
        {
            LAUNCH_MOE_GGUF(QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1);
            break;
        }
        case 1: // Q4K
        {
            LAUNCH_MOE_GGUF(QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1);
            break;
        }
        case 2: // Q2_K
        {
            LAUNCH_MOE_GGUF(QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1);
            break;
        }
        case 3: // Q3_K
        {
            LAUNCH_MOE_GGUF(QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1);
            break;
        }
        case 4: // Q5_K
        {
            LAUNCH_MOE_GGUF(QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1);
            break;
        }
        case 5: // Q6K
        {
            LAUNCH_MOE_GGUF(QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1);
            break;
        }
        default:
            break;
    }
    cudaFreeAsync(y_q8_1, stream);
}

#define LAUNCH_GATE_UP_DECODE(qk, qi, block_q_t, vdr, vec_dot_q_cuda)          \
    moe_gate_up_decode_kernel<qk, qi, block_q_t, vdr, vec_dot_q_cuda>          \
        <<<gate_grid, gate_block, 0, stream>>>(                                 \
            gate_weights, up_weights, input_q8_1, sorted_token_ids, expert_ids,\
            gate_outputs, up_outputs, num_experts, topk, intermediate_size,     \
            hidden_size, hidden_padded)

#define LAUNCH_DOWN_DECODE(qk, qi, block_q_t, vdr, vec_dot_q_cuda)             \
    moe_down_decode_kernel<qk, qi, block_q_t, vdr, vec_dot_q_cuda>             \
        <<<down_grid, down_block, 0, stream>>>(                                 \
            down_weights, down_input_q8_1, sorted_token_ids, expert_ids,        \
            topk_weights, outputs, num_experts, topk, hidden_size,              \
            intermediate_size, intermediate_padded)

extern "C" void moe_gemm_gguf_decode_silu(
    const float *inputs,
    const void *gate_weights,
    const void *up_weights,
    const void *down_weights,
    const int32_t *sorted_token_ids,
    const int32_t *expert_ids,
    const float *topk_weights,
    float *outputs,
    int num_experts,
    int topk,
    int hidden_size,
    int intermediate_size,
    int gate_up_quant_type,
    int down_quant_type,
    cudaStream_t stream) {
    // This path is deliberately specialized for MiniMax's one-token top-8
    // decode. Other shapes continue through the general MoE launchers.
    if (topk != 8) {
        return;
    }
    const int hidden_padded = pad(hidden_size, MATRIX_ROW_PADDING);
    const int intermediate_padded = pad(intermediate_size, MATRIX_ROW_PADDING);

    const size_t input_q8_bytes =
        (size_t)hidden_padded / QK8_1 * sizeof(block_q8_1);
    const size_t projection_bytes =
        (size_t)topk * intermediate_size * sizeof(float);
    const size_t down_input_q8_bytes =
        (size_t)topk * intermediate_padded / QK8_1 * sizeof(block_q8_1);
    char *scratch = nullptr;
    cudaMallocAsync(
        &scratch,
        input_q8_bytes + 2 * projection_bytes + down_input_q8_bytes,
        stream);
    block_q8_1 *input_q8_1 = reinterpret_cast<block_q8_1 *>(scratch);
    float *gate_outputs = reinterpret_cast<float *>(scratch + input_q8_bytes);
    float *up_outputs = reinterpret_cast<float *>(
        scratch + input_q8_bytes + projection_bytes);
    block_q8_1 *down_input_q8_1 = reinterpret_cast<block_q8_1 *>(
        scratch + input_q8_bytes + 2 * projection_bytes);

    quantize_q8_1<<<
        dim3(ceil_div(hidden_padded, CUDA_QUANTIZE_BLOCK_SIZE), 1, 1),
        dim3(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1),
        0,
        stream>>>(inputs, input_q8_1, hidden_size, hidden_padded);

    constexpr int gate_warps = 2;
    const dim3 gate_grid(ceil_div(intermediate_size, gate_warps), topk, 2);
    const dim3 gate_block(WARP_SIZE, gate_warps);
    switch (gate_up_quant_type) {
        case 0:
            LAUNCH_GATE_UP_DECODE(
                QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ,
                vec_dot_q8_0_q8_1);
            break;
        case 1:
            LAUNCH_GATE_UP_DECODE(
                QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ,
                vec_dot_q4_K_q8_1);
            break;
        case 2:
            LAUNCH_GATE_UP_DECODE(
                QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ,
                vec_dot_q2_K_q8_1);
            break;
        case 3:
            LAUNCH_GATE_UP_DECODE(
                QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ,
                vec_dot_q3_K_q8_1);
            break;
        case 4:
            LAUNCH_GATE_UP_DECODE(
                QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ,
                vec_dot_q5_K_q8_1);
            break;
        case 5:
            LAUNCH_GATE_UP_DECODE(
                QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ,
                vec_dot_q6_K_q8_1);
            break;
    }

    quantize_silu_mul_q8_1<<<
        dim3(ceil_div(intermediate_padded, CUDA_QUANTIZE_BLOCK_SIZE), topk, 1),
        dim3(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1),
        0,
        stream>>>(gate_outputs, up_outputs, down_input_q8_1, topk,
                  intermediate_size, intermediate_padded);

    const dim3 down_grid(hidden_size, 1, 1);
    const dim3 down_block(WARP_SIZE, topk, 1);
    switch (down_quant_type) {
        case 0:
            LAUNCH_DOWN_DECODE(
                QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ,
                vec_dot_q8_0_q8_1);
            break;
        case 1:
            LAUNCH_DOWN_DECODE(
                QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ,
                vec_dot_q4_K_q8_1);
            break;
        case 2:
            LAUNCH_DOWN_DECODE(
                QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ,
                vec_dot_q2_K_q8_1);
            break;
        case 3:
            LAUNCH_DOWN_DECODE(
                QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ,
                vec_dot_q3_K_q8_1);
            break;
        case 4:
            LAUNCH_DOWN_DECODE(
                QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ,
                vec_dot_q5_K_q8_1);
            break;
        case 5:
            LAUNCH_DOWN_DECODE(
                QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ,
                vec_dot_q6_K_q8_1);
            break;
    }

    cudaFreeAsync(scratch, stream);
}
