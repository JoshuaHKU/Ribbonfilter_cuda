#include "cuda_ribbonfilter.h"

// ... 构造、析构、LoadFromFileToGPU等实现略 ...

// 设备端hash函数（与CPU端一致，示例用简单乘法hash，可替换为实际Ribbon hash）
__device__ static uint64_t ribbon_hash(uint64_t key) {
    // 建议用实际RibbonFilter的hash算法
    uint64_t h = key * 0xc28f82822b650bedULL;
    return h;
}

// 设备端bswap64
__device__ static uint64_t bswap64(uint64_t x) {
    return  ((x & 0x00000000000000FFULL) << 56) |
            ((x & 0x000000000000FF00ULL) << 40) |
            ((x & 0x0000000000FF0000ULL) << 24) |
            ((x & 0x00000000FF000000ULL) << 8)  |
            ((x & 0x000000FF00000000ULL) >> 8)  |
            ((x & 0x0000FF0000000000ULL) >> 24) |
            ((x & 0x00FF000000000000ULL) >> 40) |
            ((x & 0xFF00000000000000ULL) >> 56);
}

// 设备端expected计算
template <typename CoeffType, uint32_t kNumColumns, uint32_t kMinPctOverhead, uint32_t kMilliBitsPerKey>
__device__ CoeffType DeviceGetExpected(uint64_t hash) {
    uint64_t a = hash * 0xc28f82822b650bedULL;
    uint64_t rr = bswap64(a);
    return rr & ((CoeffType{1} << kNumColumns) - 1);
}

// 设备端Contain实现
template <typename CoeffType, uint32_t kNumColumns, uint32_t kMinPctOverhead, uint32_t kMilliBitsPerKey>
__device__ bool CudaRibbonFilter<CoeffType, kNumColumns, kMinPctOverhead, kMilliBitsPerKey>::DeviceContain(
    const DeviceFilter* filter, uint64_t key) {
    // 1. hash
    uint64_t hash = ribbon_hash(key);

    // 2. segment_num
    size_t segment_num = hash % filter->num_starts;

    // 3. segment数据
    CoeffType segment = 0;
    if (segment_num * sizeof(CoeffType) < filter->data_len) {
        segment = ((CoeffType*)(filter->d_ptr))[segment_num];
    }

    // 4. expected
    CoeffType expected = DeviceGetExpected<CoeffType, kNumColumns, kMinPctOverhead, kMilliBitsPerKey>(hash);

    // 5. mask
    CoeffType mask = (CoeffType{1} << kNumColumns) - 1;

    // 6. 比较
    return (segment & mask) == (expected & mask);
}