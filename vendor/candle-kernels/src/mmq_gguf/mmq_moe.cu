// Quantized MoE prefill using llama.cpp-style MMQ kernels.
//
// Inputs are first quantized to the block_q8_1_mmq layout in expert-sorted
// route order. The MMQ kernels consume the compact rows together with the
// expert segment bounds and scatter results back to flattened token/top-k
// order through sorted_token_ids.

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

extern "C" void launch_mmq_quantize_q8_1_moe_DS4(
    const void *x_f32, const int32_t *sorted_ids, int id_divisor, void *vy,
    int64_t ne00, int64_t s01, int64_t ne0, int64_t ne1, void *stream);

extern "C" void launch_mmq_gguf_q4_k_moe(
    void *tmp_fixup, const void *x, const void *y_q8_1_mmq,
    const int32_t *ids_dst, const int32_t *expert_bounds, void *dst,
    int64_t ncols_x, int64_t nrows_x, int64_t ncols_y,
    int64_t stride_row_x, int64_t stride_col_dst,
    int num_experts, int64_t ncols_max, int use_stream_k,
    int cc, int nsm, int64_t smpbo, int warp_size, void *stream);

extern "C" void launch_mmq_gguf_q5_k_moe(
    void *tmp_fixup, const void *x, const void *y_q8_1_mmq,
    const int32_t *ids_dst, const int32_t *expert_bounds, void *dst,
    int64_t ncols_x, int64_t nrows_x, int64_t ncols_y,
    int64_t stride_row_x, int64_t stride_col_dst,
    int num_experts, int64_t ncols_max, int use_stream_k,
    int cc, int nsm, int64_t smpbo, int warp_size, void *stream);

static __global__ void find_expert_bounds(
    const int32_t * __restrict__ expert_ids,
    int32_t * __restrict__ expert_bounds,
    int size_m,
    int num_experts) {
    const int expert = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert > num_experts) {
        return;
    }

    // expert_ids is already sorted. One lower_bound per expert avoids an
    // atomic histogram and a separate prefix scan.
    int low = 0;
    int high = size_m;
    while (low < high) {
        const int mid = low + (high - low) / 2;
        if (expert_ids[mid] < expert) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    expert_bounds[expert] = low;
}

static __global__ void scale_moe_rows(
    float * __restrict__ output,
    const float * __restrict__ topk_weights,
    int64_t elements,
    int size_n) {
    const int64_t index = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        output[index] *= topk_weights[index / size_n];
    }
}

extern "C" void moe_gemm_mmq_gguf(
    const float *input,
    const void *weights,
    const int32_t *sorted_token_ids,
    const int32_t *expert_ids,
    const float *topk_weights,
    float *output,
    int num_experts,
    int topk,
    int size_m,
    int size_n,
    int size_k,
    int quant_type,
    cudaStream_t stream) {
    // This path currently covers the types used by MiniMax-M2.7's experts.
    if (quant_type != 1 && quant_type != 4) { // Q4_K and Q5_K
        return;
    }

    constexpr int matrix_row_padding = 512;
    constexpr int qk_k = 256;
    constexpr size_t q8_1_mmq_bytes_per_128_values = 144;
    constexpr int max_mmq_x = 128;
    constexpr int mmq_y = 128;

    const int k_padded = ((size_k + matrix_row_padding - 1) / matrix_row_padding) * matrix_row_padding;
    const int num_tokens = size_m / topk;
    const int id_divisor = topk_weights == nullptr ? topk : 1;

    void *input_q8_1 = nullptr;
    int32_t *expert_bounds = nullptr;
    void *tmp_fixup = nullptr;

    const size_t q8_data_bytes = (size_t)size_m * (k_padded / 128) * q8_1_mmq_bytes_per_128_values;
    // MMQ loads a full route tile even when the final expert segment is partial.
    const size_t q8_padding_bytes = (size_t)max_mmq_x * q8_1_mmq_bytes_per_128_values;
    cudaMallocAsync(&input_q8_1, q8_data_bytes + q8_padding_bytes, stream);
    cudaMemsetAsync((char *)input_q8_1 + q8_data_bytes, 0, q8_padding_bytes, stream);
    cudaMallocAsync((void **)&expert_bounds, (size_t)(num_experts + 1) * sizeof(int32_t), stream);

    constexpr int bounds_threads = 256;
    const int bounds_blocks = (num_experts + 1 + bounds_threads - 1) / bounds_threads;
    find_expert_bounds<<<bounds_blocks, bounds_threads, 0, stream>>>(
        expert_ids, expert_bounds, size_m, num_experts);

    launch_mmq_quantize_q8_1_moe_DS4(
        input, sorted_token_ids, id_divisor, input_q8_1,
        size_k, size_k, k_padded, size_m, (void *)stream);

    int device = 0;
    int major = 0;
    int minor = 0;
    int nsm = 0;
    int smpbo = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, device);
    cudaDeviceGetAttribute(&smpbo, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    const int cc = major * 100 + minor * 10;

    // On Volta and newer, this kernel's device-side scheduler is always the
    // stream-k implementation. Launching it with a conventional 3-D tile grid
    // would duplicate work because blockIdx.y/z are intentionally unused.
    constexpr int use_stream_k = 1;
    const size_t fixup_bytes = (size_t)nsm * max_mmq_x * mmq_y * sizeof(float);
    cudaMallocAsync(&tmp_fixup, fixup_bytes, stream);

    const int64_t stride_row_x = size_k / qk_k;
    if (quant_type == 1) {
        launch_mmq_gguf_q4_k_moe(
            tmp_fixup, weights, input_q8_1, sorted_token_ids, expert_bounds, output,
            size_k, size_n, size_m, stride_row_x, size_n,
            num_experts, num_tokens, use_stream_k,
            cc, nsm, smpbo, 32, (void *)stream);
    } else {
        launch_mmq_gguf_q5_k_moe(
            tmp_fixup, weights, input_q8_1, sorted_token_ids, expert_bounds, output,
            size_k, size_n, size_m, stride_row_x, size_n,
            num_experts, num_tokens, use_stream_k,
            cc, nsm, smpbo, 32, (void *)stream);
    }

    if (topk_weights != nullptr) {
        constexpr int threads = 256;
        const int64_t elements = (int64_t)size_m * size_n;
        const int blocks = (int)((elements + threads - 1) / threads);
        scale_moe_rows<<<blocks, threads, 0, stream>>>(
            output, topk_weights, elements, size_n);
    }

    if (tmp_fixup != nullptr) {
        cudaFreeAsync(tmp_fixup, stream);
    }
    cudaFreeAsync(expert_bounds, stream);
    cudaFreeAsync(input_q8_1, stream);
}
