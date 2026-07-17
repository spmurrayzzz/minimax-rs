#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdint>

namespace {

constexpr int HEAD_DIM = 128;
constexpr int QUERY_HEADS = 48;
constexpr int KV_HEADS = 8;
constexpr int GQA_GROUP = QUERY_HEADS / KV_HEADS;
constexpr int WARPS = 4;
constexpr int THREADS = WARPS * 32;
constexpr int PARTIAL_STRIDE = HEAD_DIM + 2;
constexpr int MAX_SPLITS = 128;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return __shfl_sync(0xffffffff, value, 0);
}

// Each block owns one query position, one KV head, and one contiguous causal
// sequence split. Its four warps process disjoint key positions and share each
// K/V row across all six GQA query heads. The block emits one online-softmax
// partial per query head.
__global__ __launch_bounds__(THREADS, 2) void gqa_split_f16_128(
    const half *__restrict__ q,
    const half *__restrict__ k,
    const half *__restrict__ v,
    float *__restrict__ partials,
    int past_len,
    int q_head_stride,
    int q_seq_stride,
    int k_head_stride,
    int v_head_stride,
    int num_splits,
    float scale) {
    const int split = blockIdx.x;
    const int kv_head = blockIdx.y;
    const int query = blockIdx.z;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int tid = threadIdx.x;

    const int causal_len = past_len + query + 1;
    const int split_begin = static_cast<int>((static_cast<int64_t>(causal_len) * split) / num_splits);
    const int split_end = static_cast<int>((static_cast<int64_t>(causal_len) * (split + 1)) / num_splits);

    float2 q_reg[GQA_GROUP][2];
#pragma unroll
    for (int group = 0; group < GQA_GROUP; ++group) {
        const int q_head = kv_head * GQA_GROUP + group;
        const half2 *q_ptr = reinterpret_cast<const half2 *>(
            q + q_head * q_head_stride + query * q_seq_stride);
        q_reg[group][0] = __half22float2(q_ptr[lane]);
        q_reg[group][1] = __half22float2(q_ptr[32 + lane]);
    }

    float running_max[GQA_GROUP];
    float running_sum[GQA_GROUP];
    float2 accumulator[GQA_GROUP][2];
#pragma unroll
    for (int group = 0; group < GQA_GROUP; ++group) {
        running_max[group] = -FLT_MAX;
        running_sum[group] = 0.0f;
        accumulator[group][0] = make_float2(0.0f, 0.0f);
        accumulator[group][1] = make_float2(0.0f, 0.0f);
    }

    const half *k_head = k + kv_head * k_head_stride;
    const half *v_head = v + kv_head * v_head_stride;
    for (int pos = split_begin + warp; pos < split_end; pos += WARPS) {
        const half2 *k_row = reinterpret_cast<const half2 *>(k_head + pos * HEAD_DIM);
        const float2 k0 = __half22float2(k_row[lane]);
        const float2 k1 = __half22float2(k_row[32 + lane]);

        float scores[GQA_GROUP];
#pragma unroll
        for (int group = 0; group < GQA_GROUP; ++group) {
            float dot = q_reg[group][0].x * k0.x + q_reg[group][0].y * k0.y;
            dot += q_reg[group][1].x * k1.x + q_reg[group][1].y * k1.y;
            scores[group] = warp_sum(dot) * scale;
        }

        const half2 *v_row = reinterpret_cast<const half2 *>(v_head + pos * HEAD_DIM);
        const float2 v0 = __half22float2(v_row[lane]);
        const float2 v1 = __half22float2(v_row[32 + lane]);
#pragma unroll
        for (int group = 0; group < GQA_GROUP; ++group) {
            const float next_max = fmaxf(running_max[group], scores[group]);
            const float old_scale = __expf(running_max[group] - next_max);
            const float weight = __expf(scores[group] - next_max);
            running_sum[group] = running_sum[group] * old_scale + weight;
            accumulator[group][0].x = accumulator[group][0].x * old_scale + weight * v0.x;
            accumulator[group][0].y = accumulator[group][0].y * old_scale + weight * v0.y;
            accumulator[group][1].x = accumulator[group][1].x * old_scale + weight * v1.x;
            accumulator[group][1].y = accumulator[group][1].y * old_scale + weight * v1.y;
            running_max[group] = next_max;
        }
    }

    __shared__ float warp_max[WARPS][GQA_GROUP];
    __shared__ float warp_sum_value[WARPS][GQA_GROUP];
    __shared__ float warp_acc[WARPS][GQA_GROUP][HEAD_DIM];
    __shared__ float combine_weight[GQA_GROUP][WARPS];
    __shared__ float block_max[GQA_GROUP];
    __shared__ float block_sum[GQA_GROUP];

#pragma unroll
    for (int group = 0; group < GQA_GROUP; ++group) {
        if (lane == 0) {
            warp_max[warp][group] = running_max[group];
            warp_sum_value[warp][group] = running_sum[group];
        }
        warp_acc[warp][group][2 * lane] = accumulator[group][0].x;
        warp_acc[warp][group][2 * lane + 1] = accumulator[group][0].y;
        warp_acc[warp][group][64 + 2 * lane] = accumulator[group][1].x;
        warp_acc[warp][group][64 + 2 * lane + 1] = accumulator[group][1].y;
    }
    __syncthreads();

    if (tid < GQA_GROUP) {
        const int group = tid;
        float max_value = -FLT_MAX;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) {
            max_value = fmaxf(max_value, warp_max[w][group]);
        }
        float sum_value = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) {
            const float weight = warp_sum_value[w][group] == 0.0f
                                     ? 0.0f
                                     : __expf(warp_max[w][group] - max_value);
            combine_weight[group][w] = weight;
            sum_value += warp_sum_value[w][group] * weight;
        }
        block_max[group] = max_value;
        block_sum[group] = sum_value;
    }
    __syncthreads();

    const int q_head_base = kv_head * GQA_GROUP;
#pragma unroll
    for (int group = 0; group < GQA_GROUP; ++group) {
        float value = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) {
            value += warp_acc[w][group][tid] * combine_weight[group][w];
        }
        float *partial = partials +
            ((query * num_splits + split) * QUERY_HEADS + q_head_base + group) * PARTIAL_STRIDE;
        partial[2 + tid] = value;
        if (tid == 0) {
            partial[0] = block_max[group];
            partial[1] = block_sum[group];
        }
    }
}

// Merge sequence splits using the same numerically-stable online-softmax
// representation: (maximum, exponential sum, unnormalised weighted value).
__global__ __launch_bounds__(THREADS, 2) void gqa_combine_f16_128(
    const float *__restrict__ partials,
    half *__restrict__ output,
    int query_len,
    int num_splits) {
    const int q_head = blockIdx.x;
    const int query = blockIdx.y;
    const int dim = threadIdx.x;
    const float *query_partials =
        partials + query * num_splits * QUERY_HEADS * PARTIAL_STRIDE;

    __shared__ float weights[MAX_SPLITS];
    __shared__ float denominators[MAX_SPLITS];
    __shared__ float global_max;
    __shared__ float global_sum;

    if (dim == 0) {
        float max_value = -FLT_MAX;
        for (int split = 0; split < num_splits; ++split) {
            const float *partial = query_partials + (split * QUERY_HEADS + q_head) * PARTIAL_STRIDE;
            max_value = fmaxf(max_value, partial[0]);
        }
        global_max = max_value;
    }
    __syncthreads();

    if (dim < num_splits) {
        const float *partial = query_partials + (dim * QUERY_HEADS + q_head) * PARTIAL_STRIDE;
        const float weight = partial[1] == 0.0f ? 0.0f : __expf(partial[0] - global_max);
        weights[dim] = weight;
        denominators[dim] = partial[1] * weight;
    }
    __syncthreads();

    if (dim == 0) {
        float sum_value = 0.0f;
        for (int split = 0; split < num_splits; ++split) {
            sum_value += denominators[split];
        }
        global_sum = sum_value;
    }
    __syncthreads();

    float value = 0.0f;
    for (int split = 0; split < num_splits; ++split) {
        const float *partial = query_partials + (split * QUERY_HEADS + q_head) * PARTIAL_STRIDE;
        value += partial[2 + dim] * weights[split];
    }
    output[(q_head * query_len + query) * HEAD_DIM + dim] = __float2half_rn(value / global_sum);
}

} // namespace

extern "C" void launch_gqa_f16_128(
    const void *q,
    const void *k,
    const void *v,
    void *partials,
    void *output,
    int query_len,
    int past_len,
    int q_head_stride,
    int q_seq_stride,
    int k_head_stride,
    int v_head_stride,
    int num_splits,
    float scale,
    void *stream) {
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const dim3 split_grid(num_splits, KV_HEADS, query_len);
    gqa_split_f16_128<<<split_grid, THREADS, 0, cuda_stream>>>(
        static_cast<const half *>(q),
        static_cast<const half *>(k),
        static_cast<const half *>(v),
        static_cast<float *>(partials),
        past_len,
        q_head_stride,
        q_seq_stride,
        k_head_stride,
        v_head_stride,
        num_splits,
        scale);
    const dim3 combine_grid(QUERY_HEADS, query_len);
    gqa_combine_f16_128<<<combine_grid, THREADS, 0, cuda_stream>>>(
        static_cast<const float *>(partials),
        static_cast<half *>(output),
        query_len,
        num_splits);
}
