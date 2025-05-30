#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <stdint.h>

// 仅支持uint64_t类型的Key和64位Hash，适用于RibbonFilter<uint64_t, 15, 0>
template <typename CoeffType, uint32_t kNumColumns, uint32_t kMinPctOverhead, uint32_t kMilliBitsPerKey = 7700>
class CudaRibbonFilter {
public:
    struct DeviceFilter {
        char* d_ptr;         // 过滤器主数据
        char* d_meta_ptr;    // 元数据
        size_t bytes;
        size_t meta_bytes;
        uint32_t log2_vshards;
        double kFractionalCols;
        size_t num_slots;
        size_t rec_count;
        size_t kCoeffBits;
        size_t upper_num_columns;
        size_t upper_start_block;
        size_t num_starts;
        size_t data_len;
    };

    CudaRibbonFilter() : d_filter_{nullptr, nullptr, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0} {}

    ~CudaRibbonFilter() {
        if (d_filter_.d_ptr) cudaFree(d_filter_.d_ptr);
        if (d_filter_.d_meta_ptr) cudaFree(d_filter_.d_meta_ptr);
    }

    // 从文件反序列化并上传到GPU
    void LoadFromFileToGPU(const std::string& filename) {
        // 1. 读取文件头和参数
        std::ifstream ifs(filename, std::ios::binary);
        if (!ifs) throw std::runtime_error("无法打开过滤器文件: " + filename);

        // 读取自定义头部（假设头部结构如下，实际请根据你的实现调整）
        struct FileHead {
            uint32_t magic;
            uint32_t version;
            uint32_t log2_vshards;
            double kFractionalCols;
            size_t num_slots;
            size_t rec_count;
            size_t meta_bytes;
            size_t bytes;
            size_t kCoeffBits;
            size_t upper_num_columns;
            size_t upper_start_block;
            size_t num_starts;
            size_t data_len;
        } fhead;

        ifs.read(reinterpret_cast<char*>(&fhead), sizeof(FileHead));
        if (!ifs) throw std::runtime_error("读取过滤器头部失败: " + filename);

        // 2. 读取元数据和主数据
        std::vector<char> meta_buf(fhead.meta_bytes);
        std::vector<char> data_buf(fhead.bytes);
        ifs.read(meta_buf.data(), fhead.meta_bytes);
        ifs.read(data_buf.data(), fhead.bytes);
        if (!ifs) throw std::runtime_error("读取过滤器数据失败: " + filename);

        // 3. 上传到GPU
        if (d_filter_.d_ptr) cudaFree(d_filter_.d_ptr);
        if (cudaMalloc(&d_filter_.d_ptr, fhead.bytes) != cudaSuccess)
            throw std::runtime_error("cudaMalloc d_ptr failed");
        cudaMemcpy(d_filter_.d_ptr, data_buf.data(), fhead.bytes, cudaMemcpyHostToDevice);

        if (d_filter_.d_meta_ptr) cudaFree(d_filter_.d_meta_ptr);
        if (cudaMalloc(&d_filter_.d_meta_ptr, fhead.meta_bytes) != cudaSuccess)
            throw std::runtime_error("cudaMalloc d_meta_ptr failed");
        cudaMemcpy(d_filter_.d_meta_ptr, meta_buf.data(), fhead.meta_bytes, cudaMemcpyHostToDevice);

        // 4. 记录参数
        d_filter_.bytes = fhead.bytes;
        d_filter_.meta_bytes = fhead.meta_bytes;
        d_filter_.log2_vshards = fhead.log2_vshards;
        d_filter_.kFractionalCols = fhead.kFractionalCols;
        d_filter_.num_slots = fhead.num_slots;
        d_filter_.rec_count = fhead.rec_count;
        d_filter_.kCoeffBits = fhead.kCoeffBits;
        d_filter_.upper_num_columns = fhead.upper_num_columns;
        d_filter_.upper_start_block = fhead.upper_start_block;
        d_filter_.num_starts = fhead.num_starts;
        d_filter_.data_len = fhead.data_len;
    }

    const DeviceFilter& GetDeviceFilter() const { return d_filter_; }

    // =========================
    // 设备端查询接口（__device__）
    // =========================

    // 仅支持kFixedNumColumns > 0的RibbonFilter
    __device__ static bool DeviceContain(const DeviceFilter* filter, uint64_t key) {
        // 1. 计算hash
        uint64_t hash = DeviceGetHash(key);

        // 2. 计算segment_num, num_columns, start_bit
        size_t kCoeffBits = filter->kCoeffBits;
        size_t num_starts = filter->num_starts;
        size_t upper_num_columns = filter->upper_num_columns;

        // 计算segment_num
        size_t segment_num = DeviceGetSegmentNum(hash, num_starts);

        // 计算num_columns和start_bit
        size_t num_columns = upper_num_columns;
        size_t start_bit = (segment_num % upper_num_columns) * kCoeffBits;

        // 3. 取出segment数据
        CoeffType segment = 0;
        if (segment_num * sizeof(CoeffType) < filter->data_len) {
            segment = ((CoeffType*)(filter->d_ptr))[segment_num];
        }

        // 4. 取出expected
        CoeffType expected = DeviceGetExpected(hash);

        // 5. 判断
        CoeffType mask = (CoeffType{1} << kNumColumns) - 1;
        return ((segment >> start_bit) & mask) == (expected & mask);
    }

private:
    DeviceFilter d_filter_;

    // 设备端hash函数（需与CPU端一致，这里用简单hash示例）
    __device__ static uint64_t DeviceGetHash(uint64_t key) {
        // 实际应与CPU端RibbonFilter的hash一致
        uint64_t h = key * 0xc28f82822b650bedULL;
        return h;
    }

    // 设备端计算segment_num
    __device__ static size_t DeviceGetSegmentNum(uint64_t hash, size_t num_starts) {
        // FastRangeGeneric
        return (hash % num_starts);
    }

    // 添加 bswap64
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

    __device__ static CoeffType DeviceGetExpected(uint64_t hash) {
        uint64_t a = hash * 0xc28f82822b650bedULL;
        uint64_t rr = bswap64(a);
        return rr & ((CoeffType{1} << kNumColumns) - 1);
    }
};