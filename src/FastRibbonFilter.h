#ifndef FAST_RIBBON_FILTERAPI_H
#define FAST_RIBBON_FILTERAPI_H

#pragma once

#include <climits>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <stdexcept>
#include <stdio.h>
#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cassert>
#include <iomanip>
#include "xorfilter.h"
#include "xorfilter_plus.h"
#include "xorfilter_singleheader.h"
#include "ribbon_impl.h" // RibbonTS, InterleavedSoln, etc.
#define CONTAIN_ATTRIBUTES __attribute__((noinline))

using namespace std;
using namespace ribbon;

template <typename CoeffType, bool kHomog, uint32_t kNumColumns,
          bool kSmash = false>
struct RibbonTS {
  static constexpr bool kIsFilter = true;
  static constexpr bool kHomogeneous = kHomog;
  static constexpr bool kFirstCoeffAlwaysOne = true;
  static constexpr bool kUseSmash = kSmash;
  using CoeffRow = CoeffType;
  using Hash = uint64_t;
  using Key = uint64_t;
  using Seed = uint32_t;
  using Index = size_t;
  using ResultRow = uint32_t;
  static constexpr bool kAllowZeroStarts = false;
  static constexpr uint32_t kFixedNumColumns = kNumColumns;

  static Hash HashFn(const Hash &input, Seed raw_seed) {
    // return input;
    uint64_t h = input + raw_seed;
    h ^= h >> 33;
    h *= UINT64_C(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h *= UINT64_C(0xc4ceb9fe1a85ec53);
    h ^= h >> 33;
    return h;
  }
};
//------------------------------------------------------------------------------------------------------
// RibbonFilter文件头
struct RibbonFilterFileHead{
  size_t rec_count;
  uint32_t log2_vshards;
  double kFractionalCols;
  size_t num_slots;
  size_t bytes;
  size_t meta_bytes;
};
//------------------------------------------------------------------------------------------------------
//经测试，以下实现的这个RibbonFilter 是速度最快的
//当每个元素占用15bits时，假阳性率为0.0036%,空间浪费率为2.1%
//当每个元素占用13bits时，假阳性率为0.0128%,空间浪费率为1.0%
template <typename CoeffType, uint32_t kNumColumns, uint32_t kMinPctOverhead,
          uint32_t kMilliBitsPerKey = 7700>
class FastRibbonFilter {
  using TS = RibbonTS<CoeffType, /*kHomog*/ false, kNumColumns>;
  IMPORT_RIBBON_IMPL_TYPES(TS);

  static constexpr uint32_t kBitsPerVshard = 8;
  using BalancedBanding = ribbon::BalancedBanding<TS, kBitsPerVshard>;
  using BalancedHasher = ribbon::BalancedHasher<TS, kBitsPerVshard>;

  double kFractionalCols;
  uint32_t log2_vshards;
  size_t num_slots;
  size_t bytes;
  unique_ptr<char[]> ptr;
  InterleavedSoln soln;   //对应class SerializableInterleavedSolution 
  size_t meta_bytes;
  unique_ptr<char[]> meta_ptr;
  BalancedHasher hasher;

  size_t rec_count;  
private:
  //把实例的当前参数写入文件头结构体,此函数在序列化时调用
  void InitFileHead(RibbonFilterFileHead* fhead) const {
    fhead->bytes = bytes;
    fhead->meta_bytes = meta_bytes;
    fhead->rec_count = rec_count;
    fhead->log2_vshards = log2_vshards;
    fhead->kFractionalCols = kFractionalCols;
    fhead->num_slots = num_slots;
  }
  //用从文件载入的数据重新初始化过滤器,此函数在反序列化时调用
  void ReInitFilter(const RibbonFilterFileHead& fhead) {
    rec_count = fhead.rec_count;
    kFractionalCols = fhead.kFractionalCols;
    log2_vshards = fhead.log2_vshards;
    num_slots = fhead.num_slots;
    bytes = fhead.bytes;
    meta_bytes = fhead.meta_bytes;
    ptr.reset(new char[bytes]);
    meta_ptr.reset(new char[meta_bytes]);
    hasher = BalancedHasher(log2_vshards, meta_ptr.get());
  }
public:

  static double GetNumSlots(size_t add_count, uint32_t log2_vshards) {
    size_t add_per_vshard = add_count >> log2_vshards;
    double overhead;
    if (sizeof(CoeffType) == 8) {
      overhead = 0.0000055 * add_per_vshard;
    } else if (sizeof(CoeffType) == 4) {
      overhead = 0.00005 * add_per_vshard;
    } else if (sizeof(CoeffType) == 2) {
      overhead = 0.00010 * add_per_vshard;
    } else {
      assert(sizeof(CoeffType) == 16);
      overhead = 0.0000013 * add_per_vshard;
    }
    overhead = std::max(overhead, 0.01 * kMinPctOverhead);
    return InterleavedSoln::RoundUpNumSlots(
        (size_t)(add_count + overhead * add_count + add_per_vshard / 5));
  }

  FastRibbonFilter(size_t add_count)
    : kFractionalCols(kNumColumns == 0 ? kMilliBitsPerKey / 1000.0 : kNumColumns),
      log2_vshards((uint32_t)FloorLog2((add_count + add_count / 3 + add_count / 5) / (128 * sizeof(CoeffType)))),
      num_slots(GetNumSlots(add_count, log2_vshards)),
      bytes(static_cast<size_t>((num_slots * kFractionalCols + 7) / 8)),
      ptr(nullptr),
      soln(nullptr, 0),
      meta_bytes(BalancedHasher(log2_vshards, nullptr).GetMetadataLength()),
      meta_ptr(new char[meta_bytes]),
      hasher(log2_vshards, meta_ptr.get())
{
    rec_count = add_count;
    if (bytes == 0) {
        throw std::runtime_error("过滤器分配空间为0，参数设置有误！");
    }
    ptr.reset(new char[bytes]);
    soln = InterleavedSoln(ptr.get(), bytes);
    /*调试注释*/
    // std::cout << "开始构造FastRibbonFilter,主要参数如下:\n\t记录总数: " << add_count/10000 << "万条"<<std::endl;
    // std::cout << "\tnum_slots: " << num_slots << std::endl;
    // std::cout << "\tmeta_bytes: " << meta_bytes << std::endl;
    // std::cout << "\tkFractionalCols: " << kFractionalCols << std::endl;
    // std::cout << "过滤器占用空间: " << bytes << "字节 ("
    //           << std::fixed << std::setprecision(2)
    //           << (bytes / 1024.0 / 1024 / 1024) << " GB)" << std::endl;
  }
  //把vector对象传入的元素批量写入过滤器
  void AddAll(const vector<uint64_t> &keys, const size_t start, const size_t end) {
    BalancedBanding b(log2_vshards);
    b.BalancerAddRange(keys.begin() + start, keys.begin() + end);
    if (!b.Balance(num_slots)) {
      fprintf(stderr, "Failed!\n");
      return;
    }
    soln.BackSubstFrom(b);
    memcpy(meta_ptr.get(), b.GetMetadata(), b.GetMetadataLength());
  }
  //把单个元素写入过滤器
  void Add(const uint64_t &key){
    BalancedBanding b(log2_vshards);
    b.BalancerAdd(key);
  }
  //判断一个元素是否在过滤器中
  bool Contain(uint64_t key) const { return soln.FilterQuery(key, hasher); }
  //获取过滤器占用的内存大小
  size_t SizeInBytes() const { return bytes + meta_bytes; }
  //获取过滤器的记录总数
  size_t RecordCount() const { return rec_count; }

  // 序列化过滤器文件
  size_t SaveFilterToFile(const std::string& filename) const {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
      throw std::runtime_error("无法打开文件进行写入: " + filename);
    }
    RibbonFilterFileHead fhead;
    InitFileHead(&fhead);
    //写入文件头
    ofs.write(reinterpret_cast<const char*>(&fhead), sizeof(RibbonFilterFileHead));
    //写入元数据
    ofs.write(meta_ptr.get(), meta_bytes);
    //写入过滤器数据
    ofs.write(ptr.get(), bytes);
    if (!ofs) {
      throw std::runtime_error("向文件写入过滤器数据失败: " + filename);
    }    
    ofs.close();
    // 获取文件大小
    std::ifstream ifs(filename, std::ios::binary | std::ios::ate);
    size_t file_size = 0;
    if (ifs) {
      file_size = static_cast<size_t>(ifs.tellg());
      ifs.close();
    }    
    return file_size;
  }

  // 从文件反序列化到 meta_ptr
  void LoadFilterFromFile(const std::string& filename) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) {
      throw std::runtime_error("无法打开文件进行读取: " + filename);
    }
    RibbonFilterFileHead fhead;
    //先读入文件头
    ifs.read(reinterpret_cast<char*>(&fhead), sizeof(RibbonFilterFileHead));
    if (!ifs) {
      throw std::runtime_error("读取文件头数据失败: " + filename);
    }
    //用文件头数据得新初始化类成员
    ReInitFilter(fhead);
    //读入元数据
    ifs.read(meta_ptr.get(), meta_bytes);
    if (!ifs) {
      throw std::runtime_error("从文件读元数据失败: " + filename);
    }    
    //读入过滤器数据
    ifs.read(ptr.get(), bytes);
    if (!ifs) {
      throw std::runtime_error("从文件读过滤器数据失败: " + filename);
    }
    ifs.close();
    // 重置序列化的解题方案
    soln.ResetSerializableInterleavedSolution(ptr.get(), bytes);
  }
};
//---------------------------------------------------------------------------------------------------------------
// RibbonFilterAPI 主模板
template <typename T>
struct RibbonFilterAPI;

// RibbonFilterAPI 特化
template <typename CoeffType, uint32_t kNumColumns, uint32_t kMinPctOverhead, uint32_t kMilliBitsPerKey>
struct RibbonFilterAPI<FastRibbonFilter<CoeffType, kNumColumns, kMinPctOverhead, kMilliBitsPerKey>> {
  using RibbonTable = FastRibbonFilter<CoeffType, kNumColumns, kMinPctOverhead, kMilliBitsPerKey>;
  static RibbonTable ConstructFromAddCount(size_t add_count) {
    return RibbonTable(add_count);
  }
  static void Add(uint64_t key, RibbonTable *table) {
    table->Add(key);
  }
  static void AddAll(const vector<uint64_t> &keys, const size_t start,
                     const size_t end, RibbonTable *table) {
    table->AddAll(keys, start, end);
  }
  static void Remove(uint64_t, RibbonTable *) {
    throw std::runtime_error("Unsupported");
  }
  CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const RibbonTable *table) {
    return table->Contain(key);
  }
};
#endif