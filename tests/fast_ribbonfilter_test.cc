// This benchmark reports on the bulk insert and bulk query rates. It is invoked as:
//
//     ./bulk-insert-and-query.exe 158000
//
// That invocation will test each probabilistic membership container type with 158000
// randomly generated items. It tests bulk Add() from empty to full and Contain() on
// filters with varying rates of expected success. For instance, at 75%, three out of
// every four values passed to Contain() were earlier Add()ed.
//
// Example usage:
//
// for alg in `seq 0 1 14`; do for num in `seq 10 10 200`; do ./bulk-insert-and-query.exe ${num}000000 ${alg}; done; done > results.txt
#include "random.h"
#include "timing.h"
#ifdef __linux__
#include "linux-perf-events.h"
#endif
#include "FastRibbonFilter.h"

using MyRibbonFilter = FastRibbonFilter<uint64_t, 15, 0>;
// The number of items sampled when determining the lookup performance
const size_t MAX_SAMPLE_SIZE = 10 * 1000 * 1000;

// The statistics gathered for each table type:
struct Statistics {
  size_t add_count;
  double nanos_per_add;
  double nanos_per_remove;
  // key: percent of queries that were expected to be positive
  map<int, double> nanos_per_finds;
  double false_positive_probabilty;
  double bits_per_item;
};


// Output for the first row of the table of results. type_width is the maximum number of
// characters of the description of any table type, and find_percent_count is the number
// of different lookup statistics gathered for each table. This function assumes the
// lookup expected positive probabiilties are evenly distributed, with the first being 0%
// and the last 100%.
string StatisticsTableHeader(int type_width, const std::vector<double> &found_probabilities) {
  ostringstream os;

  os << string(type_width, ' ');
  os << setw(8) << right << "";
  os << setw(8) << right << "";
  for (size_t i = 0; i < found_probabilities.size(); ++i) {
    os << setw(8) << "find";
  }
  os << setw(8) << "1*add+";
  os << setw(8) << "" << setw(11) << "" << setw(11)
     << "optimal" << setw(8) << "wasted" << setw(8) << "million" << endl;

  os << string(type_width, ' ');
  os << setw(8) << right << "add";
  os << setw(8) << right << "remove";
  for (double prob : found_probabilities) {
    os << setw(8 - 1) << static_cast<int>(prob * 100.0) << '%';
  }
  os << setw(8) << "3*find";
  os << setw(9) << "ε%" << setw(11) << "bits/item" << setw(11)
     << "bits/item" << setw(8) << "space%" << setw(8) << "keys";
  return os.str();
}

// Overloading the usual operator<< as used in "std::cout << foo", but for Statistics
template <class CharT, class Traits>
basic_ostream<CharT, Traits>& operator<<(
    basic_ostream<CharT, Traits>& os, const Statistics& stats) {
  os << fixed << setprecision(2) << setw(8) << right
     << stats.nanos_per_add;
  double add_and_find = 0;
  os << fixed << setprecision(2) << setw(8) << right
     << stats.nanos_per_remove;
  for (const auto& fps : stats.nanos_per_finds) {
    os << setw(8) << fps.second;
    add_and_find += fps.second;
  }
  add_and_find = add_and_find * 3 / stats.nanos_per_finds.size();
  add_and_find += stats.nanos_per_add;
  os << setw(8) << add_and_find;

  // we get some nonsensical result for very small fpps
  if(stats.false_positive_probabilty > 0.0000001) {
    const auto minbits = log2(1 / stats.false_positive_probabilty);
    os << setw(8) << setprecision(4) << stats.false_positive_probabilty * 100
       << setw(11) << setprecision(2) << stats.bits_per_item << setw(11) << minbits
       << setw(8) << setprecision(1) << 100 * (stats.bits_per_item / minbits - 1)
       << " " << setw(7) << setprecision(3) << (stats.add_count / 1000000.);
  } else {
    os << setw(8) << setprecision(4) << stats.false_positive_probabilty * 100
       << setw(11) << setprecision(2) << stats.bits_per_item << setw(11) << 64
       << setw(8) << setprecision(1) << 0
       << " " << setw(7) << setprecision(3) << (stats.add_count / 1000000.);
  }
  return os;
}


// assuming that first1,last1 and first2, last2 are sorted,
// this tries to find out how many of first1,last1 can be
// found in first2, last2, this includes duplicates
template<class InputIt1, class InputIt2>
size_t match_size_iter(InputIt1 first1, InputIt1 last1,
                          InputIt2 first2, InputIt2 last2) {
    size_t answer = 0;
    while (first1 != last1 && first2 != last2) {
        if (*first1 < *first2) {
            ++first1;
        } else  if (*first2 < *first1) {
            ++first2;
        } else {
            answer ++;
            ++first1;
        }
    }
    return answer;
}

template<class InputIt>
size_t count_distinct(InputIt first, InputIt last) {
    if(last  == first) return 0;
    size_t answer = 1;
    auto val = *first;
    first++;

    while (first != last) {
      if(val != *first) ++answer;
      first++;
    }
    return answer;
}

size_t match_size(vector<uint64_t> a,  vector<uint64_t> b, size_t * distincta, size_t * distinctb) {
  // could obviously be accelerated with a Bloom filter
  // But this is surprisingly fast!
  vector<uint64_t> result;
  std::sort(a.begin(), a.end());
  std::sort(b.begin(), b.end());
  if(distincta != NULL) *distincta  = count_distinct(a.begin(), a.end());
  if(distinctb != NULL) *distinctb  = count_distinct(b.begin(), b.end());
  return match_size_iter(a.begin(), a.end(),b.begin(), b.end());
}

bool has_duplicates(vector<uint64_t> a) {
  std::sort(a.begin(), a.end());
  return count_distinct(a.begin(), a.end()) < a.size();
}
struct samples {
  double found_probability;
  std::vector<uint64_t> to_lookup_mixed;
  size_t true_match;
  size_t actual_sample_size;
};

uint64_t reverseBitsSlow(uint64_t v) {
    // r will be reversed bits of v; first get LSB of v
    uint64_t r = v & 1;
    int s = sizeof(v) * CHAR_BIT - 1; // extra shift needed at end
    for (v >>= 1; v; v >>= 1) {
        r <<= 1;
        r |= v & 1;
        s--;
    }
    r <<= s; // shift when v's highest bits are zero
    return r;
}

void parse_comma_separated(char * c, std::set<int> & answer ) {
    std::stringstream ss(c);
    int i;
    while (ss >> i) {
        answer.insert(i);
        if (ss.peek() == ',') {
            ss.ignore();
        } else if (ss.peek() == '-') {
            ss.ignore();
            int j;
            ss >> j;
            for (i++; i <= j; i++) {
                answer.insert(i);
            }
        }
    }
}
typedef struct samples samples_t;
// --------------------------------------------------------------------------------------------------------------------------
//测试平衡型RibbonFilter
int TestBalanceRibbonFilter(int argc, char * argv[]){

  // Parameter Parsing ----------------------------------------------------------
  const char * add_count_str;
  if (argc < 2) {
    cout << "使用方法: " << argv[0] << " <过滤器元素的个数> [<algorithmId> [<seed>]]" << endl;
    cout << " 过滤器元素的个数: number of keys, we recommend at least 100000000" << endl;
    add_count_str = "10000000";  
  } else {
    add_count_str = argv[1];
  }
  stringstream input_string(add_count_str);
  size_t add_count;
  input_string >> add_count;
  if (input_string.fail()) {
    cerr << "Invalid number: " << add_count_str << endl;
    return 2;
  }
  //int algorithmId = -1; // -1 is just the default
  int seed = -1;

  size_t actual_sample_size = MAX_SAMPLE_SIZE;
  if (actual_sample_size > add_count) {
    actual_sample_size = add_count;
  }

  // Generating Samples ----------------------------------------------------------

  vector<uint64_t> to_add = seed == -1 ?
      GenerateRandom64Fast(add_count, rand()) :
      GenerateRandom64Fast(add_count, seed);
  vector<uint64_t> to_lookup = seed == -1 ?
      GenerateRandom64Fast(actual_sample_size, rand()) :
      GenerateRandom64Fast(actual_sample_size, seed + add_count);

  if (seed >= 0 && seed < 64) {
    // 0-64 are special seeds
    uint rotate = seed;
    cout << "Using sequential ordering rotated by " << rotate << endl;
    for(uint64_t i = 0; i < to_add.size(); i++) {
        to_add[i] = xorfilter::rotl64(i, rotate);
    }
    for(uint64_t i = 0; i < to_lookup.size(); i++) {
        to_lookup[i] = xorfilter::rotl64(i + to_add.size(), rotate);
    }
  } else if (seed >= 64 && seed < 128) {
    // 64-127 are special seeds
    uint rotate = seed - 64;
    cout << "Using sequential ordering rotated by " << rotate << " and reversed bits " << endl;
    for(uint64_t i = 0; i < to_add.size(); i++) {
        to_add[i] = reverseBitsSlow(xorfilter::rotl64(i, rotate));
    }
    for(uint64_t i = 0; i < to_lookup.size(); i++) {
        to_lookup[i] = reverseBitsSlow(xorfilter::rotl64(i + to_add.size(), rotate));
    }
  }

  assert(to_lookup.size() == actual_sample_size);
  size_t distinct_lookup;
  size_t distinct_add;
  std::cout << "checking match size... " << std::flush;
  size_t intersectionsize = match_size(to_lookup, to_add, &distinct_lookup, & distinct_add);
  std::cout << "\r                       \r" << std::flush;

  if(intersectionsize > 0) {
    cout << "WARNING: Out of the lookup table, "<< intersectionsize<< " ("<<intersectionsize * 100.0 / to_lookup.size() << "%) of values are present in the filter." << endl;
  }

  if(distinct_lookup != to_lookup.size()) {
    cout << "WARNING: Lookup contains "<< (to_lookup.size() - distinct_lookup)<<" duplicates." << endl;
  }
  if(distinct_add != to_add.size()) {
    cout << "WARNING: Filter contains "<< (to_add.size() - distinct_add) << " duplicates." << endl;
  }

  if (actual_sample_size > to_lookup.size()) {
    std::cerr << "actual_sample_size = "<< actual_sample_size << std::endl;
    throw out_of_range("to_lookup must contain at least actual_sample_size values");
  }

  std::vector<samples_t> mixed_sets;

  const std::vector<double> found_probabilities = { 0.0, 0.25, 0.5, 0.75, 1.0 };

  for (const double found_probability : found_probabilities) {
    std::cout << "generating samples with probability " << found_probability <<" ... " << std::flush;

    struct samples thisone;
    thisone.found_probability = found_probability;
    thisone.actual_sample_size = actual_sample_size;
    uint64_t mixingseed = seed == -1 ? random() : seed;
    // seed could be 0 (incremental numbers, or random() might return 0), which we can't use
    if (seed == 0) seed = 1;
    thisone.to_lookup_mixed = DuplicateFreeMixIn(&to_lookup[0], &to_lookup[actual_sample_size], &to_add[0],
        &to_add[add_count], found_probability, mixingseed);
    assert(thisone.to_lookup_mixed.size() == actual_sample_size);
    thisone.true_match = match_size(thisone.to_lookup_mixed,to_add, NULL, NULL);
    double trueproba =  thisone.true_match /  static_cast<double>(actual_sample_size) ;
    double bestpossiblematch = fabs(round(found_probability * actual_sample_size) / static_cast<double>(actual_sample_size) - found_probability);
    double tolerance = bestpossiblematch > 0.01 ? bestpossiblematch : 0.01;
    double probadiff = fabs(trueproba - found_probability);
    if(probadiff >= tolerance) {
      cerr << "WARNING: You claim to have a find proba. of " << found_probability << " but actual is " << trueproba << endl;
      return EXIT_FAILURE;
    }
    mixed_sets.push_back(thisone);
    std::cout << "\r                                                                                         \r"  << std::flush;
  }
  std::cout << "将写入过滤器的元素集to_add 占用内存空间: "
          << (to_add.size() * sizeof(uint64_t)) << " 字节 ("
          << std::fixed << std::setprecision(2)
          << (to_add.size() * sizeof(uint64_t) / 1024.0 / 1024) << " MB, "
          << (to_add.size() * sizeof(uint64_t) / 1024.0 / 1024 / 1024) << " GB)"
          << std::endl;  
  std::cout << "用于校验的元素集mixed_sets 占用内存空间: "
          << (mixed_sets.size() * sizeof(uint64_t)) << " 字节 ("
          << std::fixed << std::setprecision(2)
          << (mixed_sets.size() * sizeof(uint64_t) / 1024.0 / 1024) << " MB, "
          << (mixed_sets.size() * sizeof(uint64_t) / 1024.0 / 1024 / 1024) << " GB)"
          << std::endl;  
  // Begin benchmark ----------------------------------------------------------
  constexpr int NAME_WIDTH = 32;
  cout <<"开始测试"<< add_count << "条记录..." << endl;  
  MyRibbonFilter filter = RibbonFilterAPI<MyRibbonFilter>::ConstructFromAddCount(add_count);
  Statistics result;
#ifdef __linux__
  vector<int> evts;
  evts.push_back(PERF_COUNT_HW_CPU_CYCLES);
  evts.push_back(PERF_COUNT_HW_INSTRUCTIONS);
  evts.push_back(PERF_COUNT_HW_CACHE_MISSES);
  evts.push_back(PERF_COUNT_HW_BRANCH_MISSES);
  LinuxEvents<PERF_TYPE_HARDWARE> unified(evts);
  vector<unsigned long long> results;
  results.resize(evts.size());
  cout << endl;
  unified.start();
#else
   std::cout << "-" << std::flush;
#endif
  bool batchedadd = true;
  bool remove = false;
  // Add values until failure or until we run out of values to add:
  if(batchedadd) {
    std::cout << "正在批量把元素写入过滤器...." << std::flush;
  } else {
    std::cout << "1-by-1 add" << std::flush;
  }  
  auto start_time = NowNanos();     //记录开始时间
  if(batchedadd) {
    //把全部批量记录写入filter
    RibbonFilterAPI<MyRibbonFilter>::AddAll(to_add, 0, add_count, &filter);
  }
  else {
    //把记录逐条写入filter
    for (size_t added = 0; added < add_count; ++added) {
      filter.Add(to_add[added]);
      //RibbonFilterAPI<MyRibbonFilter>::Add(to_add[added], &filter);
    }
  }
  //测试把过滤器写入文件
  
  string filter_path = "./test_ribbon_filter.bin";  //"~/dict/ribbon_filter/test_ribbon_filter.bin";
  std::cout << "\n开始把数据保存到文件" << filter_path << "..." << std::flush;
  auto save_start = NowNanos();
  size_t filter_size = filter.SaveFilterToFile(filter_path);
  auto save_end = NowNanos();
  std::cout << "\n耗时: " << std::fixed << std::setprecision(2)
          << (save_end - save_start) / 1e9 << "秒,把过滤器写入文件,文件大小为:"
          << std::fixed << std::setprecision(2)
          << (filter_size / 1024.0 / 1024 / 1024) << " GB)\n" << std::endl;
  //测试从文件载入过滤器
  save_start = NowNanos();  
  filter.LoadFilterFromFile(filter_path);
  save_end = NowNanos();
  std::cout << "\n耗时: " << std::fixed << std::setprecision(2)
          << (save_end - save_start) / 1e9 << "秒,把过滤器从文件" << filter_path << "载入,\n\t过滤器占用内存:"
          << std::fixed << std::setprecision(2) << (filter.SizeInBytes() / 1024.0 / 1024 / 1024) << " GB),共有"
          << std::fixed << std::setprecision(2) << (filter.RecordCount()/10000.0) <<"万条记录.\n" << std::endl;

  //打印性能测试统计表的表头
  cout << StatisticsTableHeader(NAME_WIDTH, found_probabilities) << endl;  
  auto time = NowNanos() - start_time;
  std::cout << "\r             \r" << std::flush;
#ifdef __linux__
  unified.end(results);
  printf("add    ");
  printf("cycles: %5.1f/key, instructions: (%5.1f/key, %4.2f/cycle) cache misses: %5.2f/key branch misses: %4.2f/key effective frequency %4.2f GHz\n",
    results[0]*1.0/add_count,
    results[1]*1.0/add_count ,
    results[1]*1.0/results[0],
    results[2]*1.0/add_count,
    results[3]*1.0/add_count,
    results[0]*1.0/time);
#else
  std::cout << "." << std::flush;
#endif

  // sanity check:
  for (size_t added = 0; added < add_count; ++added) {
    assert(RibbonFilterAPI<MyRibbonFilter>::Contain(to_add[added], &filter) == 1);
  }

  result.add_count = add_count;
  result.nanos_per_add = static_cast<double>(time) / add_count;
  result.bits_per_item = static_cast<double>(CHAR_BIT * filter.SizeInBytes()) / add_count;
  size_t found_count = 0;

  for (auto t :  mixed_sets) {
    const double found_probability = t.found_probability;
    const auto to_lookup_mixed =  t.to_lookup_mixed ;
    size_t true_match = t.true_match ;

#ifdef __linux__
    unified.start();
#else
    std::cout << "-" << std::flush;
#endif
    const auto start_time = NowNanos();
    found_count = 0;
    for (const auto v : to_lookup_mixed) {
      found_count += RibbonFilterAPI<MyRibbonFilter>::Contain(v, &filter);
    }
    const auto lookup_time = NowNanos() - start_time;
#ifdef __linux__
    unified.end(results);
    printf("%3.2f%%  ",found_probability);
    printf("cycles: %5.1f/key, instructions: (%5.1f/key, %4.2f/cycle) cache misses: %5.2f/key branch misses: %4.2f/key effective frequency %4.2f GHz\n",
      results[0]*1.0/to_lookup_mixed.size(),
      results[1]*1.0/to_lookup_mixed.size(),
      results[1]*1.0/results[0],
      results[2]*1.0/to_lookup_mixed.size(),
      results[3] * 1.0/to_lookup_mixed.size(),
      results[0]*1.0/lookup_time);
#else
    std::cout << "." << std::flush;
#endif

    if (found_count < true_match) {
           cerr << "ERROR: Expected to find at least " << true_match << " found " << found_count << endl;
           cerr << "ERROR: This is a potential bug!" << endl;
    }
    result.nanos_per_finds[100 * found_probability] =
        static_cast<double>(lookup_time) / t.actual_sample_size;
    if (0.0 == found_probability) {
      ////////////////////////////
      // This is obviously technically wrong!!! The assumption is that there is no overlap between the random
      // queries and the random content. This is likely true if your 64-bit values were generated randomly,
      // but not true in general.
      ///////////////////////////
      // result.false_positive_probabilty =
      //    found_count / static_cast<double>(to_lookup_mixed.size());
      if(t.to_lookup_mixed.size() == intersectionsize) {
        cerr << "WARNING: fpp is probably meaningless! " << endl;
      }
      result.false_positive_probabilty = (found_count  - intersectionsize) / static_cast<double>(to_lookup_mixed.size() - intersectionsize);
    }
  }

  // Remove
  result.nanos_per_remove = 0;
  if (remove) {
    std::cout << "1-by-1 remove" << std::flush;
#ifdef __linux__
    unified.start();
#else
    std::cout << "-" << std::flush;
#endif
    start_time = NowNanos();
    for (size_t added = 0; added < add_count; ++added) {
      RibbonFilterAPI<MyRibbonFilter>::Remove(to_add[added], &filter);
    }
    time = NowNanos() - start_time;
    result.nanos_per_remove = static_cast<double>(time) / add_count;
#ifdef __linux__
    unified.end(results);
    printf("remove ");
    printf("cycles: %5.1f/key, instructions: (%5.1f/key, %4.2f/cycle) cache misses: %5.2f/key branch misses: %4.2f/key effective frequency %4.2f GHz\n",
      results[0]*1.0/add_count,
      results[1]*1.0/add_count ,
      results[1]*1.0/results[0],
      results[2]*1.0/add_count,
      results[3]*1.0/add_count,
      results[0]*1.0/time);
#else
    std::cout << "." << std::flush;
#endif
  }

#ifndef __linux__
  std::cout << "\r             \r" << std::flush;
#endif
    cout << setw(NAME_WIDTH) << "FastRibbonFilter" << result << endl;
  return 0;
}
// --------------------------------------------------------------------------------------------------------------------------
int main(int argc, char * argv[]) {
  TestBalanceRibbonFilter(argc,argv);
}

