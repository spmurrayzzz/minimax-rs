// Stable expert grouping for MoE routes.
//
// Candle's generic bitonic argsort stores the whole flattened route list in
// dynamic shared memory. Top-8 routing therefore fails above 1024 tokens.
// CUB radix sort has no such block-local limit and directly produces the
// expert-sorted route IDs required by the MMQ kernels.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <cstdint>

static __global__ void route_iota(uint32_t *ids, int size) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        ids[index] = (uint32_t)index;
    }
}

extern "C" void moe_sort_routes(
    const uint32_t *expert_ids_unsorted,
    uint32_t *expert_ids_sorted,
    uint32_t *route_ids_sorted,
    int size,
    int end_bit,
    cudaStream_t stream) {
    uint32_t *route_ids = nullptr;
    void *temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    cudaMallocAsync((void **)&route_ids, (size_t)size * sizeof(uint32_t), stream);
    constexpr int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    route_iota<<<blocks, threads, 0, stream>>>(route_ids, size);

    cub::DeviceRadixSort::SortPairs(
        temp_storage, temp_storage_bytes,
        expert_ids_unsorted, expert_ids_sorted,
        route_ids, route_ids_sorted,
        size, 0, end_bit, stream);
    cudaMallocAsync(&temp_storage, temp_storage_bytes, stream);
    cub::DeviceRadixSort::SortPairs(
        temp_storage, temp_storage_bytes,
        expert_ids_unsorted, expert_ids_sorted,
        route_ids, route_ids_sorted,
        size, 0, end_bit, stream);

    cudaFreeAsync(temp_storage, stream);
    cudaFreeAsync(route_ids, stream);
}
