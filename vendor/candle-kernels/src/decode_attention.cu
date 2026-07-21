#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include "mmq_gguf/mmq_mma.cuh"

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

// Tensor-core decode path used for long contexts. llama.cpp makes the same
// vector-to-MMA transition for FP16 GQA once the KV cache reaches 8192 tokens.
// A block owns one KV head and one sequence split. It evaluates all six query
// heads together in 128-key tiles, cutting the per-key online-softmax work in
// half and replacing scalar QK/PV products with FP16 MMA and FP32 accumulation.
constexpr int MMA_M = 16;
constexpr int MMA_N = 16;
constexpr int MMA_K = 16;
constexpr int MMA_KEY_TILE = 128;

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    return __shfl_sync(0xffffffff, value, 0);
}

__device__ __forceinline__ void mma_softmax_tile(
    float *__restrict__ matrix,
    half *__restrict__ probabilities,
    float *__restrict__ warp_maxima,
    float *__restrict__ warp_sums,
    float *__restrict__ online_max,
    float *__restrict__ online_sum,
    float *__restrict__ old_scales,
    int valid_keys,
    float scale) {
    const int tid = threadIdx.y * 32 + threadIdx.x;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;

    float scores[GQA_GROUP];
#pragma unroll
    for (int group = 0; group < GQA_GROUP; ++group) {
        scores[group] = tid < valid_keys ? matrix[group * MMA_KEY_TILE + tid] * scale : -FLT_MAX;
        const float maximum = warp_max(scores[group]);
        if (lane == 0) {
            warp_maxima[warp * GQA_GROUP + group] = maximum;
        }
    }
    __syncthreads();

    if (warp == 0 && lane < GQA_GROUP) {
        float tile_max = -FLT_MAX;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) {
            tile_max = fmaxf(tile_max, warp_maxima[w * GQA_GROUP + lane]);
        }
        const float next_max = fmaxf(online_max[lane], tile_max);
        old_scales[lane] = online_sum[lane] == 0.0f ? 0.0f : __expf(online_max[lane] - next_max);
        online_max[lane] = next_max;
    }
    __syncthreads();

#pragma unroll
    for (int group = 0; group < GQA_GROUP; ++group) {
        const float weight = tid < valid_keys ? __expf(scores[group] - online_max[group]) : 0.0f;
        probabilities[group * MMA_KEY_TILE + tid] = __float2half_rn(weight);
        const float sum = warp_sum(weight);
        if (lane == 0) {
            warp_sums[warp * GQA_GROUP + group] = sum;
        }
    }
    __syncthreads();

    if (warp == 0 && lane < GQA_GROUP) {
        float tile_sum = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) {
            tile_sum += warp_sums[w * GQA_GROUP + lane];
        }
        online_sum[lane] = online_sum[lane] * old_scales[lane] + tile_sum;
    }
    __syncthreads();
}

__device__ __forceinline__ void mma_qk_tile(
    const half *__restrict__ q_tile,
    const half *__restrict__ k_head,
    float *__restrict__ matrix,
    int key_begin) {
    using namespace nvcuda;
    const int warp = threadIdx.y;

#pragma unroll
    for (int key_column = warp * MMA_N; key_column < MMA_KEY_TILE; key_column += WARPS * MMA_N) {
        wmma::fragment<wmma::accumulator, MMA_M, MMA_N, MMA_K, float> score;
        wmma::fill_fragment(score, 0.0f);
#pragma unroll
        for (int dim = 0; dim < HEAD_DIM; dim += MMA_K) {
            wmma::fragment<wmma::matrix_a, MMA_M, MMA_N, MMA_K, half, wmma::row_major> q_fragment;
            wmma::fragment<wmma::matrix_b, MMA_M, MMA_N, MMA_K, half, wmma::col_major> k_fragment;
            wmma::load_matrix_sync(q_fragment, q_tile + dim, HEAD_DIM);
            wmma::load_matrix_sync(
                k_fragment,
                k_head + (key_begin + key_column) * HEAD_DIM + dim,
                HEAD_DIM);
            wmma::mma_sync(score, q_fragment, k_fragment, score);
        }
        wmma::store_matrix_sync(
            matrix + key_column,
            score,
            MMA_KEY_TILE,
            wmma::mem_row_major);
    }
}

__device__ __forceinline__ void mma_pv_tile(
    const half *__restrict__ probabilities,
    const half *__restrict__ v_head,
    float *__restrict__ matrix,
    int key_begin) {
    using namespace nvcuda;
    const int warp = threadIdx.y;

#pragma unroll
    for (int output_column = warp * MMA_N;
         output_column < HEAD_DIM;
         output_column += WARPS * MMA_N) {
        wmma::fragment<wmma::accumulator, MMA_M, MMA_N, MMA_K, float> value;
        wmma::fill_fragment(value, 0.0f);
#pragma unroll
        for (int key = 0; key < MMA_KEY_TILE; key += MMA_K) {
            wmma::fragment<wmma::matrix_a, MMA_M, MMA_N, MMA_K, half, wmma::row_major> p_fragment;
            wmma::fragment<wmma::matrix_b, MMA_M, MMA_N, MMA_K, half, wmma::row_major> v_fragment;
            wmma::load_matrix_sync(p_fragment, probabilities + key, MMA_KEY_TILE);
            wmma::load_matrix_sync(
                v_fragment,
                v_head + (key_begin + key) * HEAD_DIM + output_column,
                HEAD_DIM);
            wmma::mma_sync(value, p_fragment, v_fragment, value);
        }
        wmma::store_matrix_sync(
            matrix + output_column,
            value,
            HEAD_DIM,
            wmma::mem_row_major);
    }
}

__global__ __launch_bounds__(THREADS, 2) void gqa_decode_split_mma_f16_128(
    const half *__restrict__ q,
    const half *__restrict__ k,
    const half *__restrict__ v,
    float *__restrict__ partials,
    int seq_len,
    int q_head_stride,
    int k_head_stride,
    int v_head_stride,
    int num_splits,
    float scale) {
    const int split = blockIdx.x;
    const int kv_head = blockIdx.y;
    const int tid = threadIdx.y * 32 + threadIdx.x;

    // Partition complete 128-key tiles rather than raw token counts. Raw
    // splits create a padded MMA tail in every block (up to 95 redundant tiles
    // per layer); tile-aligned splits leave only the sequence's final tail.
    const int tile_count = (seq_len + MMA_KEY_TILE - 1) / MMA_KEY_TILE;
    const int first_tile = static_cast<int>((static_cast<int64_t>(tile_count) * split) / num_splits);
    const int last_tile = static_cast<int>((static_cast<int64_t>(tile_count) * (split + 1)) / num_splits);
    const int split_begin = first_tile * MMA_KEY_TILE;
    const int split_end = min(seq_len, last_tile * MMA_KEY_TILE);
    const int q_head_base = kv_head * GQA_GROUP;
    const half *k_head = k + kv_head * k_head_stride;
    const half *v_head = v + kv_head * v_head_stride;

    __shared__ __align__(16) half q_tile[MMA_M * HEAD_DIM];
    __shared__ __align__(16) float matrix[MMA_M * HEAD_DIM];
    __shared__ __align__(16) half probabilities[MMA_M * MMA_KEY_TILE];
    __shared__ float warp_maxima[WARPS * GQA_GROUP];
    __shared__ float warp_sums[WARPS * GQA_GROUP];
    __shared__ float online_max[GQA_GROUP];
    __shared__ float online_sum[GQA_GROUP];
    __shared__ float old_scales[GQA_GROUP];

    for (int index = tid; index < MMA_M * HEAD_DIM; index += THREADS) {
        const int group = index / HEAD_DIM;
        const int dim = index % HEAD_DIM;
        q_tile[index] = group < GQA_GROUP
            ? q[(q_head_base + group) * q_head_stride + dim]
            : __float2half(0.0f);
        probabilities[index] = __float2half(0.0f);
    }
    if (tid < GQA_GROUP) {
        online_max[tid] = -FLT_MAX;
        online_sum[tid] = 0.0f;
    }

    float accumulator[GQA_GROUP];
#pragma unroll
    for (int group = 0; group < GQA_GROUP; ++group) {
        accumulator[group] = 0.0f;
    }
    __syncthreads();

    int key_begin = split_begin;
    for (; key_begin + MMA_KEY_TILE <= split_end; key_begin += MMA_KEY_TILE) {
        mma_qk_tile(q_tile, k_head, matrix, key_begin);
        __syncthreads();

        mma_softmax_tile(
            matrix,
            probabilities,
            warp_maxima,
            warp_sums,
            online_max,
            online_sum,
            old_scales,
            MMA_KEY_TILE,
            scale);

        mma_pv_tile(probabilities, v_head, matrix, key_begin);
        __syncthreads();
#pragma unroll
        for (int group = 0; group < GQA_GROUP; ++group) {
            accumulator[group] = accumulator[group] * old_scales[group]
                + matrix[group * HEAD_DIM + tid];
        }
        __syncthreads();
    }

    const int tail = split_end - key_begin;
    if (tail > 0) {
        // Production KV caches grow in 4096-token increments, so their head
        // stride guarantees a readable zero-padded 128-key tile. Exact-sized
        // standalone tensors use a scalar final tail to avoid an OOB load.
        const int readable_keys = min(k_head_stride, v_head_stride) / HEAD_DIM;
        if (key_begin + MMA_KEY_TILE <= readable_keys) {
            mma_qk_tile(q_tile, k_head, matrix, key_begin);
            __syncthreads();

            mma_softmax_tile(
                matrix,
                probabilities,
                warp_maxima,
                warp_sums,
                online_max,
                online_sum,
                old_scales,
                tail,
                scale);

            mma_pv_tile(probabilities, v_head, matrix, key_begin);
            __syncthreads();
#pragma unroll
            for (int group = 0; group < GQA_GROUP; ++group) {
                accumulator[group] = accumulator[group] * old_scales[group]
                    + matrix[group * HEAD_DIM + tid];
            }
        } else {
            if (tid < tail) {
                const half2 *k_row = reinterpret_cast<const half2 *>(k_head + (key_begin + tid) * HEAD_DIM);
#pragma unroll
                for (int group = 0; group < GQA_GROUP; ++group) {
                    const half2 *q_row = reinterpret_cast<const half2 *>(q_tile + group * HEAD_DIM);
                    float dot = 0.0f;
#pragma unroll
                    for (int dim = 0; dim < HEAD_DIM / 2; ++dim) {
                        const float2 q_value = __half22float2(q_row[dim]);
                        const float2 k_value = __half22float2(k_row[dim]);
                        dot += q_value.x * k_value.x + q_value.y * k_value.y;
                    }
                    matrix[group * MMA_KEY_TILE + tid] = dot;
                }
            }
            __syncthreads();

            mma_softmax_tile(
                matrix,
                probabilities,
                warp_maxima,
                warp_sums,
                online_max,
                online_sum,
                old_scales,
                tail,
                scale);

            float tail_value[GQA_GROUP];
#pragma unroll
            for (int group = 0; group < GQA_GROUP; ++group) {
                tail_value[group] = 0.0f;
            }
            for (int key = 0; key < tail; ++key) {
                const float value = __half2float(v_head[(key_begin + key) * HEAD_DIM + tid]);
#pragma unroll
                for (int group = 0; group < GQA_GROUP; ++group) {
                    tail_value[group] += __half2float(probabilities[group * MMA_KEY_TILE + key]) * value;
                }
            }
#pragma unroll
            for (int group = 0; group < GQA_GROUP; ++group) {
                accumulator[group] = accumulator[group] * old_scales[group] + tail_value[group];
            }
        }
    }

#pragma unroll
    for (int group = 0; group < GQA_GROUP; ++group) {
        float *partial = partials +
            (split * QUERY_HEADS + q_head_base + group) * PARTIAL_STRIDE;
        partial[2 + tid] = accumulator[group];
        if (tid == 0) {
            partial[0] = online_max[group];
            partial[1] = online_sum[group];
        }
    }
}

constexpr int N8_V_STRIDE = HEAD_DIM / 2 + 4;
constexpr int N8_GROUPS = 8;

__device__ __forceinline__ void stage_v_tile_n8(
    const half *__restrict__ source,
    half2 *__restrict__ destination,
    int valid_keys) {
    constexpr int chunks_per_row = HEAD_DIM * sizeof(half) / sizeof(uint4);
    const int tid = threadIdx.y * 32 + threadIdx.x;
    if (valid_keys == MMA_KEY_TILE) {
        const unsigned int shared_base = __cvta_generic_to_shared(destination);
        for (int index = tid; index < MMA_KEY_TILE * chunks_per_row; index += THREADS) {
            const int row = index / chunks_per_row;
            const int chunk = index % chunks_per_row;
            const unsigned int dst = shared_base
                + (row * N8_V_STRIDE * sizeof(half2))
                + chunk * sizeof(uint4);
            const half *src = source + row * HEAD_DIM
                + chunk * (sizeof(uint4) / sizeof(half));
            asm volatile(
                "cp.async.cg.shared.global.L2::64B [%0], [%1], 16;"
                :
                : "r"(dst), "l"(src));
        }
    } else {
        const uint4 zero = {0, 0, 0, 0};
        for (int index = tid; index < MMA_KEY_TILE * chunks_per_row; index += THREADS) {
            const int row = index / chunks_per_row;
            const int chunk = index % chunks_per_row;
            uint4 *dst = reinterpret_cast<uint4 *>(
                destination + row * N8_V_STRIDE + chunk * (sizeof(uint4) / sizeof(half2)));
            if (row < valid_keys) {
                const uint4 *src = reinterpret_cast<const uint4 *>(
                    source + row * HEAD_DIM + chunk * (sizeof(uint4) / sizeof(half)));
                *dst = *src;
            } else {
                *dst = zero;
            }
        }
    }
}

__device__ __forceinline__ void wait_for_v_tile_n8() {
    asm volatile("cp.async.wait_all;");
}

// Long-context decode specialized to an 8-column MMA tile. Six real query
// heads plus two zero columns fill m16n8 tensor-core instructions much more
// efficiently than the generic WMMA path's 16 query rows. Four warps share a
// 128-key V tile, retain softmax probabilities in registers, and combine only
// once at the end of the sequence split.
__global__ __launch_bounds__(THREADS, 2) void gqa_decode_split_mma_n8_f16_128(
    const half *__restrict__ q,
    const half *__restrict__ k,
    const half *__restrict__ v,
    float *__restrict__ partials,
    int seq_len,
    int q_head_stride,
    int k_head_stride,
    int v_head_stride,
    int num_splits,
    float scale) {
    using namespace ggml_cuda_mma;
    using KqA = tile<16, 8, half2>;
    using KqB = tile<8, 8, half2>;
    using KqC = tile<16, 8, float>;
    using PvA = tile<16, 8, half2>;
    using PvC = tile<16, 4, half2>;

    const int split = blockIdx.x;
    const int kv_head = blockIdx.y;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    const int tid = warp * 32 + lane;

    const int tile_count = (seq_len + MMA_KEY_TILE - 1) / MMA_KEY_TILE;
    const int first_tile = static_cast<int>((static_cast<int64_t>(tile_count) * split) / num_splits);
    const int last_tile = static_cast<int>((static_cast<int64_t>(tile_count) * (split + 1)) / num_splits);
    const int split_begin = first_tile * MMA_KEY_TILE;
    const int split_end = min(seq_len, last_tile * MMA_KEY_TILE);
    const int q_head_base = kv_head * GQA_GROUP;
    const half *k_head = k + kv_head * k_head_stride;
    const half *v_head = v + kv_head * v_head_stride;
    const int q_head_stride2 = q_head_stride / 2;
    const half2 *q_half2 = reinterpret_cast<const half2 *>(q);

    extern __shared__ __align__(16) half2 kv_tiles[];
    half2 *k_tile = kv_tiles;
    half2 *v_tile = k_tile + MMA_KEY_TILE * N8_V_STRIDE;
    __shared__ float warp_maxima[WARPS * N8_GROUPS];

    KqB q_fragments[HEAD_DIM / 16];
#pragma unroll
    for (int dim = 0; dim < HEAD_DIM / 2; dim += 8) {
        KqB &fragment = q_fragments[dim / 8];
#pragma unroll
        for (int index = 0; index < fragment.ne; ++index) {
            const int group = fragment.get_i(index);
            const int offset = fragment.get_j(index);
            fragment.x[index] = group < GQA_GROUP
                ? q_half2[(q_head_base + group) * q_head_stride2 + dim + offset]
                : make_half2(0.0f, 0.0f);
        }
    }

    PvC value_fragments[HEAD_DIM / 16];
    float running_max[2] = {-FLT_MAX, -FLT_MAX};
    float running_sum[2] = {0.0f, 0.0f};

    if (split_begin < split_end) {
        stage_v_tile_n8(
            k_head + split_begin * HEAD_DIM,
            k_tile,
            min(MMA_KEY_TILE, split_end - split_begin));
    }
    for (int key_begin = split_begin; key_begin < split_end; key_begin += MMA_KEY_TILE) {
        const int valid_keys = min(MMA_KEY_TILE, split_end - key_begin);
        wait_for_v_tile_n8();
        __syncthreads();
        stage_v_tile_n8(v_head + key_begin * HEAD_DIM, v_tile, valid_keys);

        KqC scores[2];
#pragma unroll
        for (int key_fragment = 0; key_fragment < 2; ++key_fragment) {
            const int warp_key = warp * 32 + key_fragment * 16;
#pragma unroll
            for (int dim = 0; dim < HEAD_DIM / 2; dim += 8) {
                KqA k_fragment;
                load_ldmatrix(
                    k_fragment,
                    k_tile + warp_key * N8_V_STRIDE + dim,
                    N8_V_STRIDE);
                mma(scores[key_fragment], k_fragment, q_fragments[dim / 8]);
            }
        }

        float next_max[2] = {running_max[0], running_max[1]};
#pragma unroll
        for (int key_fragment = 0; key_fragment < 2; ++key_fragment) {
#pragma unroll
            for (int index = 0; index < scores[key_fragment].ne; ++index) {
                const int key = warp * 32 + key_fragment * 16
                    + scores[key_fragment].get_i(index);
                const int column = index % 2;
                if (key < valid_keys) {
                    next_max[column] = fmaxf(
                        next_max[column],
                        scores[key_fragment].x[index] * scale);
                }
            }
        }
#pragma unroll
        for (int offset = 16; offset >= 4; offset >>= 1) {
            next_max[0] = fmaxf(next_max[0], __shfl_xor_sync(0xffffffff, next_max[0], offset));
            next_max[1] = fmaxf(next_max[1], __shfl_xor_sync(0xffffffff, next_max[1], offset));
        }

        if (lane < 4) {
            warp_maxima[warp * N8_GROUPS + 2 * lane] = next_max[0];
            warp_maxima[warp * N8_GROUPS + 2 * lane + 1] = next_max[1];
        }
        __syncthreads();

        const int group0 = 2 * (lane % 4);
        const int group1 = group0 + 1;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) {
            next_max[0] = fmaxf(next_max[0], warp_maxima[w * N8_GROUPS + group0]);
            next_max[1] = fmaxf(next_max[1], warp_maxima[w * N8_GROUPS + group1]);
        }
        const float old_scale0 = running_sum[0] == 0.0f
            ? 0.0f
            : __expf(running_max[0] - next_max[0]);
        const float old_scale1 = running_sum[1] == 0.0f
            ? 0.0f
            : __expf(running_max[1] - next_max[1]);
        running_max[0] = next_max[0];
        running_max[1] = next_max[1];

        const half2 old_scale = make_half2(old_scale0, old_scale1);
#pragma unroll
        for (int output_fragment = 0; output_fragment < HEAD_DIM / 16; ++output_fragment) {
#pragma unroll
            for (int index = 0; index < value_fragments[output_fragment].ne; ++index) {
                value_fragments[output_fragment].x[index] =
                    __hmul2(value_fragments[output_fragment].x[index], old_scale);
            }
        }

        float sum_add[2] = {0.0f, 0.0f};
#pragma unroll
        for (int key_fragment = 0; key_fragment < 2; ++key_fragment) {
#pragma unroll
            for (int index = 0; index < scores[key_fragment].ne; ++index) {
                const int key = warp * 32 + key_fragment * 16
                    + scores[key_fragment].get_i(index);
                const int column = index % 2;
                const float weight = key < valid_keys
                    ? __expf(scores[key_fragment].x[index] * scale - next_max[column])
                    : 0.0f;
                scores[key_fragment].x[index] = weight;
                sum_add[column] += weight;
            }
        }
        running_sum[0] = running_sum[0] * old_scale0 + sum_add[0];
        running_sum[1] = running_sum[1] * old_scale1 + sum_add[1];

        wait_for_v_tile_n8();
        __syncthreads();
        const int next_key_begin = key_begin + MMA_KEY_TILE;
        if (next_key_begin < split_end) {
            stage_v_tile_n8(
                k_head + next_key_begin * HEAD_DIM,
                k_tile,
                min(MMA_KEY_TILE, split_end - next_key_begin));
        }

        tile<16, 4, half2> probabilities_half0;
        tile<16, 4, half2> probabilities_half1;
#pragma unroll
        for (int index = 0; index < scores[0].ne; index += 2) {
            probabilities_half0.x[index / 2] =
                make_half2(scores[0].x[index], scores[0].x[index + 1]);
            probabilities_half1.x[index / 2] =
                make_half2(scores[1].x[index], scores[1].x[index + 1]);
        }
        KqB probabilities0;
        KqB probabilities1;
        probabilities0.x[0] = ggml_cuda_movmatrix(probabilities_half0.x[0]);
        probabilities0.x[1] = ggml_cuda_movmatrix(probabilities_half0.x[1]);
        probabilities1.x[0] = ggml_cuda_movmatrix(probabilities_half1.x[0]);
        probabilities1.x[1] = ggml_cuda_movmatrix(probabilities_half1.x[1]);
#pragma unroll
        for (int output_fragment = 0; output_fragment < HEAD_DIM / 16; ++output_fragment) {
            PvA v_fragment;
            load_ldmatrix_trans(
                v_fragment,
                v_tile + (warp * 32) * N8_V_STRIDE + output_fragment * 8,
                N8_V_STRIDE);
            mma(value_fragments[output_fragment], v_fragment, probabilities0);
            load_ldmatrix_trans(
                v_fragment,
                v_tile + (warp * 32 + 16) * N8_V_STRIDE + output_fragment * 8,
                N8_V_STRIDE);
            mma(value_fragments[output_fragment], v_fragment, probabilities1);
        }
    }

    // The next iteration's leading barrier protects V before it is reused, so
    // only the final iteration needs an explicit barrier before v_tile becomes
    // the warp-combine scratch buffer.
    __syncthreads();
#pragma unroll
    for (int offset = 16; offset >= 4; offset >>= 1) {
        running_sum[0] += __shfl_xor_sync(0xffffffff, running_sum[0], offset);
        running_sum[1] += __shfl_xor_sync(0xffffffff, running_sum[1], offset);
    }

    half *warp_values = reinterpret_cast<half *>(v_tile);
#pragma unroll
    for (int output_fragment = 0; output_fragment < HEAD_DIM / 16; ++output_fragment) {
#pragma unroll
        for (int index = 0; index < value_fragments[output_fragment].ne; ++index) {
            const int dim = output_fragment * 16 + value_fragments[output_fragment].get_i(index);
            const int group_pair = value_fragments[output_fragment].get_j(index);
            const float2 values = __half22float2(value_fragments[output_fragment].x[index]);
            warp_values[((warp * N8_GROUPS + 2 * group_pair) * HEAD_DIM) + dim] =
                __float2half_rn(values.x);
            warp_values[((warp * N8_GROUPS + 2 * group_pair + 1) * HEAD_DIM) + dim] =
                __float2half_rn(values.y);
        }
    }

    float *warp_sums = reinterpret_cast<float *>(
        warp_values + WARPS * N8_GROUPS * HEAD_DIM);
    float *warp_final_max = warp_sums + WARPS * N8_GROUPS;
    if (lane < 4) {
        warp_sums[warp * N8_GROUPS + 2 * lane] = running_sum[0];
        warp_sums[warp * N8_GROUPS + 2 * lane + 1] = running_sum[1];
        warp_final_max[warp * N8_GROUPS + 2 * lane] = running_max[0];
        warp_final_max[warp * N8_GROUPS + 2 * lane + 1] = running_max[1];
    }
    __syncthreads();

#pragma unroll
    for (int group = 0; group < GQA_GROUP; ++group) {
        float value = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) {
            value += __half2float(warp_values[(w * N8_GROUPS + group) * HEAD_DIM + tid]);
        }
        float *partial = partials +
            (split * QUERY_HEADS + q_head_base + group) * PARTIAL_STRIDE;
        partial[2 + tid] = value;
        if (tid == group) {
            float sum = 0.0f;
#pragma unroll
            for (int w = 0; w < WARPS; ++w) {
                sum += warp_sums[w * N8_GROUPS + group];
            }
            partial[0] = warp_final_max[group];
            partial[1] = sum;
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

extern "C" void launch_gqa_decode_mma_f16_128(
    const void *q,
    const void *k,
    const void *v,
    void *partials,
    void *output,
    int seq_len,
    int q_head_stride,
    int k_head_stride,
    int v_head_stride,
    int num_splits,
    float scale,
    int device,
    void *stream) {
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const dim3 split_grid(num_splits, KV_HEADS);
    const dim3 mma_block(32, WARPS);
    constexpr size_t n8_shared_bytes =
        2 * MMA_KEY_TILE * N8_V_STRIDE * sizeof(half2);
    // The opt-in shared-memory limit is a per-device setting. The server uses
    // two CUDA contexts, so cache configuration independently for each GPU.
    static bool n8_dynamic_shared_attempted[32] = {};
    static bool n8_dynamic_shared_configured[32] = {};
    bool use_n8 = false;
    if (device >= 0 && device < 32) {
        if (!n8_dynamic_shared_attempted[device]) {
            n8_dynamic_shared_configured[device] = cudaFuncSetAttribute(
                gqa_decode_split_mma_n8_f16_128,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                n8_shared_bytes) == cudaSuccess;
            n8_dynamic_shared_attempted[device] = true;
        }
        use_n8 = n8_dynamic_shared_configured[device];
    } else {
        use_n8 = cudaFuncSetAttribute(
            gqa_decode_split_mma_n8_f16_128,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            n8_shared_bytes) == cudaSuccess;
    }
    if (use_n8) {
        gqa_decode_split_mma_n8_f16_128<<<split_grid, mma_block, n8_shared_bytes, cuda_stream>>>(
            static_cast<const half *>(q),
            static_cast<const half *>(k),
            static_cast<const half *>(v),
            static_cast<float *>(partials),
            seq_len,
            q_head_stride,
            k_head_stride,
            v_head_stride,
            num_splits,
            scale);
    } else {
        // Retain the lower-shared-memory WMMA implementation as a safe
        // fallback when a CUDA device cannot opt in to the pipelined kernel.
        gqa_decode_split_mma_f16_128<<<split_grid, mma_block, 0, cuda_stream>>>(
            static_cast<const half *>(q),
            static_cast<const half *>(k),
            static_cast<const half *>(v),
            static_cast<float *>(partials),
            seq_len,
            q_head_stride,
            k_head_stride,
            v_head_stride,
            num_splits,
            scale);
    }
    gqa_combine_f16_128<<<QUERY_HEADS, THREADS, 0, cuda_stream>>>(
        static_cast<const float *>(partials),
        static_cast<half *>(output),
        1,
        num_splits);
}

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
