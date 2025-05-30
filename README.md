# Ribbonfilter_cuda
Separate the ribbon filter in rocksdb into an independent module, add serialization and deserialization functions; and further implement the deployment and query verification of the filter on CUDA architecture GPU

# ribbon_filter: A Probabilistic Membership Filter

`ribbon_filter` is a high-performance, probabilistic membership filter based on the Ribbon algorithm, offering higher efficiency than traditional Bloom filters. This C++ implementation is suitable for large-scale set deduplication, membership queries, and similar scenarios. The project includes a complete Ribbon Filter algorithm implementation, along with comprehensive tests and performance evaluation tools.

## Project Structure

- `src/ribbon/`: Core implementation of the Ribbon Filter algorithm, including data structures and logic.
- `tests/`: Test and benchmarking programs for the Ribbon Filter, covering batch insertion, queries, serialization/deserialization, and more.
- `README.md`: Project documentation (this file).

## Ribbon Filter Overview

Ribbon Filter is an efficient probabilistic set membership structure with the following features:

- **High Space Efficiency**: Achieves the same false positive rate as Bloom or Xor filters with less space.
- **Fast Query Performance**: Supports batch insertion and efficient queries, ideal for large-scale data.
- **Serialization Support**: Filter state can be saved to a file for persistence and distributed scenarios.

The core implementation is in `src/ribbon/ribbon_impl.h`, supporting flexible configuration (columns, overhead, etc.) to suit various needs.

## Main Implementation Class

The `FastRibbonFilter` class in `tests/FastRibbonFilter.h` is a high-level wrapper around the core Ribbon algorithm in `src/ribbon/ribbon_impl.h`. It provides a user-friendly interface for real-world applications, supporting:

- **Filter Construction**: Automatically calculates required space and initializes the filter based on the number of elements and parameters.
- **Batch Insertion**: Efficiently inserts a `std::vector<uint64_t>` of elements, suitable for large datasets.
- **Single Element Insertion**: Supports inserting individual elements for flexibility.
- **Membership Query**: The `Contain` method efficiently checks if an element is in the filter.
- **Serialization/Deserialization**: Saves and loads the complete filter state (including metadata and internal structure) for persistence and distributed deployment.
- **Space and Statistics**: Provides memory usage, inserted element count, and other statistics for analysis and tuning.

### Typical Performance

- **15 bits per element**: ~0.0036% false positive rate, ~2.1% space overhead.
- **13 bits per element**: ~0.0128% false positive rate, ~1.0% space overhead.

### FastRibbonFilter Usage Example

A typical workflow for constructing, inserting, querying, and serializing a Ribbon Filter:

```cpp
#include "FastRibbonFilter.h"
#include <vector>
#include <iostream>

int main() {
    // 1. Construct the filter for 1 million elements
    size_t add_count = 1000000;
    FastRibbonFilter<uint64_t, 15, 0> filter(add_count);

    // 2. Batch insert data
    std::vector<uint64_t> keys;
    for (uint64_t i = 0; i < add_count; ++i) {
        keys.push_back(i);
    }
    filter.AddAll(keys, 0, keys.size());

    // 3. Query for an element
    uint64_t query_key = 12345;
    if (filter.Contain(query_key)) {
        std::cout << "Key " << query_key << " may exist in the filter." << std::endl;
    } else {
        std::cout << "Key " << query_key << " definitely does not exist in the filter." << std::endl;
    }

    // 4. Serialize to file
    filter.SaveFilterToFile("my_ribbon_filter.bin");

    // 5. Deserialize from file
    FastRibbonFilter<uint64_t, 15, 0> loaded_filter(1); // Constructor parameter will be overwritten
    loaded_filter.LoadFilterFromFile("my_ribbon_filter.bin");

    // 6. Query the deserialized filter
    if (loaded_filter.Contain(query_key)) {
        std::cout << "After deserialization, key " << query_key << " may exist." << std::endl;
    }
}
```

With these interfaces, `FastRibbonFilter` meets the needs of high-performance deduplication, membership queries, and set operations at scale, with efficient persistence and recovery.

## Build Requirements

- C++11 or newer compiler (e.g., GCC, Clang)
- Linux recommended (some performance counters require Linux)
- x64 processor with AVX2 support (some algorithms can run without AVX2)

## Build Instructions

To build the test program, enter the `tests` directory and run:

```sh
cd ./ribbon_filter/tests
make clean
make all
```

To run a performance test with 10 million elements:

```sh
./test_rfilter 10000000
```

To compile a specific test program (e.g., `tests/fast_ribbonfilter_test.cc`):

```sh
g++ -std=c++11 -O3 -Wall -I./src -o tests/test_rfilter tests/fast_ribbonfilter_test.cc
```

> Note: Add `-I` options as needed for additional header paths.

## Test Program Overview

The `tests/` directory contains programs for validating Ribbon Filter functionality and performance.  
For example, `fast_ribbonfilter_test.cc` supports:

- Batch insertion of random elements
- Batch queries and false positive rate statistics
- Serialization (save to file) and deserialization (load from file)
- Detailed performance statistics

### Example Test Run

**Build:**
```sh
cd ./ribbon_filter/tests
make all
```

**Run (insert 1 million elements):**
```sh
./test_rfilter 200000000
```

The program outputs:

- Filter parameters and memory usage
- Batch insertion and query timings
- Serialization/deserialization timings and file size
- Recall, false positive rate, space per element, and other statistics

### Main Parameters

- `add_count`: Number of elements to insert (specified via command line)
- `filter_path`: Path to the serialized filter file (modifiable in source)

### Example Output

```
Starting test with 200,000,000 records...
Constructing FastRibbonFilter, main parameters:
        Total records: 200 million
        num_slots: 200838400
        meta_bytes: 262144
        kFractionalCols: 15.00
Filter memory usage: 376,572,000 bytes (0.35 GB)

Batch inserting elements... Saving filter to ./test_ribbon_filter.bin...
Time: 0.70 s, filter written to file, file size: 0.35 GB

Time: 0.11 s, loaded filter from ./test_ribbon_filter.bin,
        Filter memory usage: 0.35 GB, total 200 million records.
...
                                                    find    find    find    find    find  1*add+                       optimal  wasted million
                                     add  remove      0%     25%     50%     75%    100%  3*find      ε%  bits/item  bits/item  space%    keys
add    cycles:   0.0/key, instructions: (  0.0/key, -nan/cycle) cache misses:  0.00/key branch misses: 0.00/key effective frequency 0.00 GHz
0.00%  cycles:   0.0/key, instructions: (  0.0/key, -nan/cycle) cache misses:  0.00/key branch misses: 0.00/key effective frequency 0.00 GHz
0.25%  cycles:   0.0/key, instructions: (  0.0/key, -nan/cycle) cache misses:  0.00/key branch misses: 0.00/key effective frequency 0.00 GHz
0.50%  cycles:   0.0/key, instructions: (  0.0/key, -nan/cycle) cache misses:  0.00/key branch misses: 0.00/key effective frequency 0.00 GHz
0.75%  cycles:   0.0/key, instructions: (  0.0/key, -nan/cycle) cache misses:  0.00/key branch misses: 0.00/key effective frequency 0.00 GHz
1.00%  cycles:   0.0/key, instructions: (  0.0/key, -nan/cycle) cache misses:  0.00/key branch misses: 0.00/key effective frequency 0.00 GHz
                FastRibbonFilter  250.35    0.00  102.88  152.15  170.58  173.60  175.79  715.35  0.0028      15.07      15.15    -0.5 200.000

```

## References

- [Binary Fuse Filters: Fast and Smaller Than Xor Filters](http://arxiv.org/abs/2201.01174)
- [Xor Filters: Faster and Smaller Than Bloom and Cuckoo Filters](https://arxiv.org/abs/1912.08258)
- [Prefix Filter: Practically and Theoretically Better Than Bloom](https://arxiv.org/abs/2203.17139)

## Acknowledgements

Some code in this project is inspired by [FastFilter/fastfilter_cpp](https://github.com/FastFilter/fastfilter_cpp) and related open-source implementations.

For more details on the Ribbon Filter algorithm, parameter tuning, or advanced usage, please refer to the source code and comments in the `src/ribbon/` directory.

## Last Modified: 2025-05-30
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------



# ribbon_filter概率过滤器
`ribbon_filter` 是一种布隆过滤器，但比布隆过滤器更高效的，基于 Ribbon 算法的高性能近似成员查询过滤器（Probabilistic Membership Filter）C++ 实现，适用于大规模集合的快速判重、去重、集合查询等场景。该项目包含完整的 Ribbon Filter 算法实现，并提供了详细的测试与性能评测工具。

## 项目结构

- `src/ribbon/`：Ribbon Filter 算法核心实现，包含底层数据结构与算法逻辑。
- `tests/`：Ribbon Filter 的测试与性能评测程序，包括批量插入、查询、序列化/反序列化等功能的验证。
- `README.md`：项目说明文档（当前文件）。

## Ribbon Filter 简介

Ribbon Filter 是一种高效的概率型集合成员查询结构，具有如下特点：

- **空间效率高**：相比 Bloom Filter、Xor Filter 等，Ribbon Filter 能以更少的空间实现同等误判率。
- **查询速度快**：支持批量插入与高效查询，适合大规模数据场景。
- **支持序列化**：可将过滤器状态保存到文件，便于持久化与分布式场景下的加载。

核心实现位于 `src/ribbon/ribbon_impl.h`，支持灵活配置参数（如列数、溢出率等），可根据实际需求调整。

## 具体的实现类

`tests/FastRibbonFilter.h` 中的 `FastRibbonFilter` 类，是对 `src/ribbon/ribbon_impl.h` 中 Ribbon 算法核心的一个高层封装。该类为实际工程应用提供了简洁易用的接口，支持如下主要功能：

- **过滤器构建**：可根据指定的元素数量和参数，自动计算所需空间并初始化 Ribbon Filter。
- **批量插入**：支持将一个 `std::vector<uint64_t>` 的元素批量写入过滤器，效率高，适合大规模数据场景。
- **单元素插入**：支持单条数据的插入，便于灵活扩展。
- **成员查询**：提供 `Contain` 方法，可高效判断某个元素是否在过滤器中。
- **序列化与反序列化**：支持将过滤器完整状态（包括元数据和内部结构）保存到文件，并可从文件恢复，方便持久化和分布式部署。
- **空间与统计信息**：可获取过滤器实际占用内存、已插入元素数量等信息，便于性能分析和调优。

### 典型参数下的性能

- **每个元素占用 15 bits** 时，假阳性率约为 0.0036%，空间浪费率约为 2.1%。
- **每个元素占用 13 bits** 时，假阳性率约为 0.0128%，空间浪费率约为 1.0%。

### FastRibbonFilter 使用示例

下面是一个典型的使用流程，展示如何构建、插入、查询和序列化 Ribbon Filter：
//----------------------------------------------------------------------------------------------------------
```cpp
#include "FastRibbonFilter.h"
#include <vector>
#include <iostream>

int main() {
    // 1. 构建过滤器，假设要插入100万条数据
    size_t add_count = 1000000;
    FastRibbonFilter<uint64_t, 15, 0> filter(add_count);

    // 2. 批量插入数据
    std::vector<uint64_t> keys;
    for (uint64_t i = 0; i < add_count; ++i) {
        keys.push_back(i);
    }
    filter.AddAll(keys, 0, keys.size());

    // 3. 查询某个元素是否存在
    uint64_t query_key = 12345;
    if (filter.Contain(query_key)) {
        std::cout << "Key " << query_key << " 可能存在于过滤器中。" << std::endl;
    } else {
        std::cout << "Key " << query_key << " 一定不存在于过滤器中。" << std::endl;
    }

    // 4. 序列化到文件
    filter.SaveFilterToFile("my_ribbon_filter.bin");

    // 5. 从文件反序列化
    FastRibbonFilter<uint64_t, 15, 0> loaded_filter(1); // 构造参数会被覆盖
    loaded_filter.LoadFilterFromFile("my_ribbon_filter.bin");

    // 6. 查询反序列化后的过滤器
    if (loaded_filter.Contain(query_key)) {
        std::cout << "反序列化后，Key " << query_key << " 可能存在。" << std::endl;
    }
}
```
//----------------------------------------------------------------------------------------------------------
通过上述接口，`FastRibbonFilter` 能够满足大规模数据去重、判重、集合查询等高性能场景的需求，并支持高效的持久化与恢复。

## 编译环境要求

- C++11 及以上标准的编译器（如 GCC、Clang）
- Linux 推荐（部分性能计数器依赖 Linux）
- 支持 AVX2 指令集的 x64 处理器（部分算法可在无 AVX2 环境下运行）

## 编译方法

编译测试程序，可进入tests目录，执行如下命令：

$cd ./ribbon_filter/tests
$make clean
$make all
运行[./test_rfilter 10000000 ]命令，执行一个有1000万个元素的创建、序列化及反序列化、查询性能测试。

如需单独编译测试程序（以 `tests/fast_ribbonfilter_test.cc` 为例）：
g++ -std=c++11 -O3 -Wall -I./src -o tests/test_rfilter tests/fast_ribbonfilter_test.cc

> 注意：如有依赖其他头文件路径，请根据实际情况补充 `-I` 参数。

## 测试程序说明

`tests/` 目录下包含的测试程序，主要用于验证 Ribbon Filter 的功能与性能。  
以 `fast_ribbonfilter_test.cc` 为例，支持如下功能：

- 批量插入指定数量的随机元素
- 批量查询并统计误判率
- 支持过滤器的序列化（保存到文件）与反序列化（从文件加载）
- 输出详细的性能统计信息

### 测试程序运行示例
# 编译
$cd ./ribbon_filter/tests
$ake all

# 运行（插入100万条数据）
$./tests/test_rfilter 200000000

运行后，程序会输出如下信息：

- 过滤器参数与内存占用
- 批量插入与查询的耗时
- 过滤器序列化/反序列化的耗时与文件大小
- 查全率、误判率、每条数据的空间占用等统计信息

### 主要参数说明

- `add_count`：插入元素数量（命令行参数指定）
- `filter_path`：过滤器序列化文件路径（可在源码中修改）

### 典型输出示例
开始测试200000000条记录...
开始构造FastRibbonFilter,主要参数如下:
        记录总数: 20000万条
        num_slots: 200838400
        meta_bytes: 262144
        kFractionalCols: 15.00
过滤器占用空间: 376572000字节 (0.35 GB)

正在批量把元素写入过滤器....开始把数据保存到文件./test_ribbon_filter.bin...
耗时: 0.70 s,把过滤器写入文件,文件大小为:0.35 GB)


耗时: 0.11 s,把过滤器载入从文件./test_ribbon_filter.bin载入,
        过滤器占用内存:0.35 GB),共有20000.00万条记录.
                                                    find    find    find    find    find  1*add+                       optimal  wasted million
                                     add  remove      0%     25%     50%     75%    100%  3*find      ε%  bits/item  bits/item  space%    keys
add    cycles:   0.0/key, instructions: (  0.0/key, -nan/cycle) cache misses:  0.00/key branch misses: 0.00/key effective frequency 0.00 GHz
0.00%  cycles:   0.0/key, instructions: (  0.0/key, -nan/cycle) cache misses:  0.00/key branch misses: 0.00/key effective frequency 0.00 GHz
0.25%  cycles:   0.0/key, instructions: (  0.0/key, -nan/cycle) cache misses:  0.00/key branch misses: 0.00/key effective frequency 0.00 GHz
0.50%  cycles:   0.0/key, instructions: (  0.0/key, -nan/cycle) cache misses:  0.00/key branch misses: 0.00/key effective frequency 0.00 GHz
0.75%  cycles:   0.0/key, instructions: (  0.0/key, -nan/cycle) cache misses:  0.00/key branch misses: 0.00/key effective frequency 0.00 GHz
1.00%  cycles:   0.0/key, instructions: (  0.0/key, -nan/cycle) cache misses:  0.00/key branch misses: 0.00/key effective frequency 0.00 GHz
                FastRibbonFilter  250.35    0.00  102.88  152.15  170.58  173.60  175.79  715.35  0.0028      15.07      15.15    -0.5 200.000

## 参考文献

- [Binary Fuse Filters: Fast and Smaller Than Xor Filters](http://arxiv.org/abs/2201.01174)
- [Xor Filters: Faster and Smaller Than Bloom and Cuckoo Filters](https://arxiv.org/abs/1912.08258)
- [Prefix Filter: Practically and Theoretically Better Than Bloom](https://arxiv.org/abs/2203.17139)

## 致谢

本项目部分代码参考自 [FastFilter/fastfilter_cpp](https://github.com/FastFilter/fastfilter_cpp) 及相关开源实现。

如需进一步了解 Ribbon Filter 算法原理、参数调优或扩展用法，请参考 `src/ribbon/` 目录下的源码及注释。
## 最后修改日期[2025-05-30]
