#include "cuda_ribbonfilter.h"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

using FilterType = CudaRibbonFilter<uint64_t, 15, 0>;

__global__ void gpu_query_kernel(const FilterType::DeviceFilter* d_filter, uint64_t query_key, bool* d_result) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *d_result = FilterType::DeviceContain(d_filter, query_key);
    }
}

int main() {
    // 1. 加载过滤器并上传到GPU
    FilterType cuda_filter;
    try {
        cuda_filter.LoadFromFileToGPU("../tests/test_ribbon_filter.bin");
    } catch (const std::exception& e) {
        std::cerr << "加载过滤器失败: " << e.what() << std::endl;
        return 1;
    }
    const auto& d_filter = cuda_filter.GetDeviceFilter();

    // 2. 分配device端过滤器结构体
    FilterType::DeviceFilter* d_filter_ptr;
    cudaMalloc(&d_filter_ptr, sizeof(FilterType::DeviceFilter));
    cudaMemcpy(d_filter_ptr, &d_filter, sizeof(FilterType::DeviceFilter), cudaMemcpyHostToDevice);

    // 3. 查询一个key
    uint64_t query_key = 12345;
    bool h_result = false;
    bool* d_result;
    cudaMalloc(&d_result, sizeof(bool));

    gpu_query_kernel<<<1, 1>>>(d_filter_ptr, query_key, d_result);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);

    std::cout << "DeviceContain(" << query_key << ") = " << (h_result ? "true" : "false") << std::endl;

    cudaFree(d_result);
    cudaFree(d_filter_ptr);

    return 0;
}