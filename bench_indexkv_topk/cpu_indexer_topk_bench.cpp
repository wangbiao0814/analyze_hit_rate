// Standalone CPU benchmark for the DeepSeek-V3.2 DSA indexer score + top-k.
//
// The score computed for every historical token s is:
//   score[s] = sum_h weight[h] * relu(dot(q[h, :], index_k[s, :]))
//
// q and index_k are stored as BF16. Dot products accumulate in FP32.  On Arm,
// the benchmark uses SVE BF16 BFDOT when available, then NEON BF16 BFDOT, and
// finally a portable scalar BF16-to-FP32 implementation.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__ARM_FEATURE_SVE_BF16)
#include <arm_sve.h>
#elif defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
#include <arm_neon.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
  std::int64_t seq_len = 128 * 1024;
  int heads = 64;
  int dim = 128;
  int topk = 2048;
  int threads = 0;
  int warmup = 3;
  int iters = 10;
  bool check = false;
};

struct Entry {
  float score;
  std::int32_t index;
};

struct TimingStats {
  double min_ms;
  double median_ms;
  double mean_ms;
};

[[noreturn]] void usage(const char* argv0, const std::string& error = {}) {
  if (!error.empty()) {
    std::cerr << "error: " << error << "\n\n";
  }
  std::cerr
      << "Usage: " << argv0 << " [options]\n"
      << "  --seq-len N   historical tokens (default: 131072)\n"
      << "  --heads N     index heads (default: 64)\n"
      << "  --dim N       index head dimension (default: 128)\n"
      << "  --topk N      selected token count (default: 2048)\n"
      << "  --threads N   OpenMP threads (default: runtime maximum)\n"
      << "  --warmup N    warmup iterations (default: 3)\n"
      << "  --iters N     measured iterations (default: 10)\n"
      << "  --check       compare the first 256 scores with scalar code\n"
      << "  --help        show this message\n";
  std::exit(error.empty() ? 0 : 2);
}

template <typename T>
T parse_integer(const char* text, const char* name) {
  std::size_t consumed = 0;
  const std::string value(text);
  long long parsed = 0;
  try {
    parsed = std::stoll(value, &consumed);
  } catch (const std::exception&) {
    usage("cpu_indexer_topk_bench", std::string("invalid ") + name + ": " + value);
  }
  if (consumed != value.size() || parsed <= 0 ||
      parsed > static_cast<long long>(std::numeric_limits<T>::max())) {
    usage("cpu_indexer_topk_bench", std::string("invalid ") + name + ": " + value);
  }
  return static_cast<T>(parsed);
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    auto require_value = [&](const char* name) -> const char* {
      if (++i >= argc) {
        usage(argv[0], std::string("missing value for ") + name);
      }
      return argv[i];
    };

    if (arg == "--seq-len") {
      options.seq_len = parse_integer<std::int64_t>(require_value("--seq-len"), "seq-len");
    } else if (arg == "--heads") {
      options.heads = parse_integer<int>(require_value("--heads"), "heads");
    } else if (arg == "--dim") {
      options.dim = parse_integer<int>(require_value("--dim"), "dim");
    } else if (arg == "--topk") {
      options.topk = parse_integer<int>(require_value("--topk"), "topk");
    } else if (arg == "--threads") {
      options.threads = parse_integer<int>(require_value("--threads"), "threads");
    } else if (arg == "--warmup") {
      options.warmup = parse_integer<int>(require_value("--warmup"), "warmup");
    } else if (arg == "--iters") {
      options.iters = parse_integer<int>(require_value("--iters"), "iters");
    } else if (arg == "--check") {
      options.check = true;
    } else if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
    } else {
      usage(argv[0], "unknown option: " + arg);
    }
  }
  if (options.topk > options.seq_len) {
    options.topk = static_cast<int>(options.seq_len);
  }
  if (options.seq_len > std::numeric_limits<std::int32_t>::max()) {
    usage(argv[0], "seq-len must fit in the int32 top-k index type");
  }
  return options;
}

inline std::uint16_t fp32_to_bf16(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  // Round to nearest, ties to even.
  const std::uint32_t rounding_bias = 0x7fffu + ((bits >> 16u) & 1u);
  return static_cast<std::uint16_t>((bits + rounding_bias) >> 16u);
}

inline float bf16_to_fp32(std::uint16_t value) {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16u;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

inline std::uint64_t splitmix64(std::uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30u)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27u)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31u);
}

inline float deterministic_float(std::uint64_t index) {
  // A deterministic value in [-0.25, 0.25]. It avoids a shared RNG during the
  // parallel first-touch initialization of the large index-K allocation.
  const std::uint32_t mantissa = static_cast<std::uint32_t>(splitmix64(index) >> 40u);
  return (static_cast<float>(mantissa) / static_cast<float>(1u << 24u) - 0.5f) * 0.5f;
}

float dot_bf16_scalar(const std::uint16_t* lhs, const std::uint16_t* rhs, int dim) {
  float sum = 0.0f;
  for (int d = 0; d < dim; ++d) {
    sum += bf16_to_fp32(lhs[d]) * bf16_to_fp32(rhs[d]);
  }
  return sum;
}

#if defined(__ARM_FEATURE_SVE_BF16)
float dot_bf16_native(const std::uint16_t* lhs, const std::uint16_t* rhs, int dim) {
  svfloat32_t accum = svdup_n_f32(0.0f);
  const std::uint64_t lanes = svcnth();
  for (std::uint64_t d = 0; d < static_cast<std::uint64_t>(dim); d += lanes) {
    const svbool_t predicate = svwhilelt_b16(d, static_cast<std::uint64_t>(dim));
    const svbfloat16_t lhs_vec = svld1_bf16(
        predicate, reinterpret_cast<const bfloat16_t*>(lhs + d));
    const svbfloat16_t rhs_vec = svld1_bf16(
        predicate, reinterpret_cast<const bfloat16_t*>(rhs + d));
    accum = svbfdot_f32(accum, lhs_vec, rhs_vec);
  }
  return svaddv_f32(svptrue_b32(), accum);
}
#elif defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
float dot_bf16_native(const std::uint16_t* lhs, const std::uint16_t* rhs, int dim) {
  float32x4_t accum = vdupq_n_f32(0.0f);
  int d = 0;
  for (; d + 8 <= dim; d += 8) {
    const bfloat16x8_t lhs_vec =
        vld1q_bf16(reinterpret_cast<const bfloat16_t*>(lhs + d));
    const bfloat16x8_t rhs_vec =
        vld1q_bf16(reinterpret_cast<const bfloat16_t*>(rhs + d));
    accum = vbfdotq_f32(accum, lhs_vec, rhs_vec);
  }
  float sum = vaddvq_f32(accum);
  for (; d < dim; ++d) {
    sum += bf16_to_fp32(lhs[d]) * bf16_to_fp32(rhs[d]);
  }
  return sum;
}
#else
float dot_bf16_native(const std::uint16_t* lhs, const std::uint16_t* rhs, int dim) {
  return dot_bf16_scalar(lhs, rhs, dim);
}
#endif

const char* kernel_name() {
#if defined(__ARM_FEATURE_SVE_BF16)
  return "Arm SVE BF16 BFDOT";
#elif defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
  return "Arm NEON BF16 BFDOT";
#else
  return "portable scalar BF16";
#endif
}

int runtime_threads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

void initialize_inputs(std::vector<std::uint16_t>& q,
                       std::vector<std::uint16_t>& index_k,
                       std::vector<float>& weights) {
#pragma omp parallel for schedule(static)
  for (std::int64_t i = 0; i < static_cast<std::int64_t>(index_k.size()); ++i) {
    index_k[static_cast<std::size_t>(i)] =
        fp32_to_bf16(deterministic_float(static_cast<std::uint64_t>(i) + 1));
  }

  for (std::size_t i = 0; i < q.size(); ++i) {
    q[i] = fp32_to_bf16(deterministic_float(i + 0x100000000ULL));
  }
  for (std::size_t i = 0; i < weights.size(); ++i) {
    // Indexer head gates are not constrained to be positive.
    weights[i] = deterministic_float(i + 0x200000000ULL) * 4.0f;
  }
}

void compute_scores(const std::vector<std::uint16_t>& q,
                    const std::vector<std::uint16_t>& index_k,
                    const std::vector<float>& weights,
                    std::vector<float>& scores,
                    int heads,
                    int dim) {
#pragma omp parallel for schedule(static)
  for (std::int64_t token = 0; token < static_cast<std::int64_t>(scores.size()); ++token) {
    const std::uint16_t* key = index_k.data() + token * dim;
    float score = 0.0f;
    for (int head = 0; head < heads; ++head) {
      const std::uint16_t* query = q.data() + static_cast<std::int64_t>(head) * dim;
      const float dot = dot_bf16_native(query, key, dim);
      score += weights[head] * std::max(dot, 0.0f);
    }
    scores[static_cast<std::size_t>(token)] = score;
  }
}

bool better_entry(const Entry& lhs, const Entry& rhs) {
  if (lhs.score != rhs.score) {
    return lhs.score > rhs.score;
  }
  return lhs.index < rhs.index;
}

std::vector<Entry> parallel_topk(const std::vector<float>& scores, int topk) {
  const int max_threads = runtime_threads();
  std::vector<std::vector<Entry>> local_candidates(max_threads);

#pragma omp parallel
  {
#ifdef _OPENMP
    const int thread_id = omp_get_thread_num();
    const int thread_count = omp_get_num_threads();
#else
    const int thread_id = 0;
    const int thread_count = 1;
#endif
    const std::int64_t begin =
        static_cast<std::int64_t>(scores.size()) * thread_id / thread_count;
    const std::int64_t end =
        static_cast<std::int64_t>(scores.size()) * (thread_id + 1) / thread_count;
    auto& local = local_candidates[thread_id];
    local.reserve(static_cast<std::size_t>(end - begin));
    for (std::int64_t i = begin; i < end; ++i) {
      local.push_back({scores[static_cast<std::size_t>(i)], static_cast<std::int32_t>(i)});
    }
    if (static_cast<int>(local.size()) > topk) {
      std::nth_element(local.begin(), local.begin() + topk, local.end(), better_entry);
      local.resize(topk);
    }
  }

  std::size_t candidate_count = 0;
  for (const auto& local : local_candidates) {
    candidate_count += local.size();
  }
  std::vector<Entry> result;
  result.reserve(candidate_count);
  for (auto& local : local_candidates) {
    result.insert(result.end(), local.begin(), local.end());
  }
  if (static_cast<int>(result.size()) > topk) {
    std::nth_element(result.begin(), result.begin() + topk, result.end(), better_entry);
    result.resize(topk);
  }
  std::sort(result.begin(), result.end(), better_entry);
  return result;
}

TimingStats summarize(std::vector<double> samples) {
  if (samples.empty()) {
    throw std::runtime_error("cannot summarize an empty timing sample");
  }
  const double mean =
      std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
  std::sort(samples.begin(), samples.end());
  const std::size_t middle = samples.size() / 2;
  const double median = samples.size() % 2 == 0
                            ? (samples[middle - 1] + samples[middle]) * 0.5
                            : samples[middle];
  return {samples.front(), median, mean};
}

double elapsed_ms(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

void check_scores(const std::vector<std::uint16_t>& q,
                  const std::vector<std::uint16_t>& index_k,
                  const std::vector<float>& weights,
                  const std::vector<float>& scores,
                  int heads,
                  int dim) {
  const std::size_t check_tokens = std::min<std::size_t>(256, scores.size());
  double max_abs_error = 0.0;
  double max_rel_error = 0.0;
  for (std::size_t token = 0; token < check_tokens; ++token) {
    const std::uint16_t* key = index_k.data() + token * dim;
    float reference = 0.0f;
    for (int head = 0; head < heads; ++head) {
      const float dot = dot_bf16_scalar(q.data() + static_cast<std::int64_t>(head) * dim,
                                        key,
                                        dim);
      reference += weights[head] * std::max(dot, 0.0f);
    }
    const double abs_error = std::abs(static_cast<double>(scores[token]) - reference);
    const double rel_error = abs_error / std::max(1.0e-6, std::abs(static_cast<double>(reference)));
    max_abs_error = std::max(max_abs_error, abs_error);
    max_rel_error = std::max(max_rel_error, rel_error);
  }
  std::cout << "check_tokens=" << check_tokens << " max_abs_error=" << max_abs_error
            << " max_rel_error=" << max_rel_error << "\n";
  if (max_abs_error > 5.0e-3 && max_rel_error > 5.0e-3) {
    throw std::runtime_error("SVE/NEON score check failed");
  }
}

void check_topk(const std::vector<float>& scores,
                const std::vector<Entry>& actual,
                int topk) {
  std::vector<Entry> expected;
  expected.reserve(scores.size());
  for (std::size_t i = 0; i < scores.size(); ++i) {
    expected.push_back({scores[i], static_cast<std::int32_t>(i)});
  }
  std::partial_sort(expected.begin(), expected.begin() + topk, expected.end(), better_entry);
  expected.resize(topk);
  if (actual.size() != expected.size()) {
    throw std::runtime_error("parallel top-k returned the wrong number of entries");
  }
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (actual[i].index != expected[i].index || actual[i].score != expected[i].score) {
      throw std::runtime_error("parallel top-k differs from serial partial_sort");
    }
  }
  std::cout << "topk_check=pass\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
#ifdef _OPENMP
    omp_set_dynamic(0);
    if (options.threads > 0) {
      omp_set_num_threads(options.threads);
    }
#else
    if (options.threads > 1) {
      std::cerr << "warning: binary was built without OpenMP; using one thread\n";
    }
#endif

    const std::size_t q_elements =
        static_cast<std::size_t>(options.heads) * options.dim;
    const std::size_t k_elements =
        static_cast<std::size_t>(options.seq_len) * options.dim;
    std::vector<std::uint16_t> q(q_elements);
    std::vector<std::uint16_t> index_k(k_elements);
    std::vector<float> weights(options.heads);
    std::vector<float> scores(static_cast<std::size_t>(options.seq_len));

    initialize_inputs(q, index_k, weights);

    std::cout << "kernel=" << kernel_name() << "\n";
    std::cout << "threads=" << runtime_threads();
#if defined(__ARM_FEATURE_SVE_BF16)
    std::cout << " sve_bits=" << svcntb() * 8;
#endif
    std::cout << "\n";
    std::cout << "shape: q=[" << options.heads << ',' << options.dim
              << "] index_k=[" << options.seq_len << ',' << options.dim
              << "] topk=" << options.topk << "\n";
    std::cout << "index_k_size_mib=" << std::fixed << std::setprecision(2)
              << (index_k.size() * sizeof(std::uint16_t) / 1048576.0) << "\n";

    std::vector<Entry> topk_result;
    for (int i = 0; i < options.warmup; ++i) {
      compute_scores(q, index_k, weights, scores, options.heads, options.dim);
      topk_result = parallel_topk(scores, options.topk);
    }

    std::vector<double> score_samples;
    std::vector<double> topk_samples;
    std::vector<double> total_samples;
    score_samples.reserve(options.iters);
    topk_samples.reserve(options.iters);
    total_samples.reserve(options.iters);

    for (int i = 0; i < options.iters; ++i) {
      const auto total_begin = Clock::now();
      const auto score_begin = total_begin;
      compute_scores(q, index_k, weights, scores, options.heads, options.dim);
      const auto score_end = Clock::now();
      topk_result = parallel_topk(scores, options.topk);
      const auto topk_end = Clock::now();
      score_samples.push_back(elapsed_ms(score_begin, score_end));
      topk_samples.push_back(elapsed_ms(score_end, topk_end));
      total_samples.push_back(elapsed_ms(total_begin, topk_end));
    }

    if (options.check) {
      check_scores(q, index_k, weights, scores, options.heads, options.dim);
      check_topk(scores, topk_result, options.topk);
    }

    const TimingStats score_stats = summarize(score_samples);
    const TimingStats topk_stats = summarize(topk_samples);
    const TimingStats total_stats = summarize(total_samples);
    const double flops = 2.0 * options.seq_len * options.heads * options.dim;
    const double effective_gflops = flops / (score_stats.median_ms * 1.0e6);
    const double unique_k_gib = index_k.size() * sizeof(std::uint16_t) /
                                static_cast<double>(1ULL << 30);
    const double unique_k_gib_per_s = unique_k_gib / (score_stats.median_ms / 1000.0);

    auto print_stats = [](const char* name, const TimingStats& stats) {
      std::cout << name << "_ms: min=" << stats.min_ms << " median=" << stats.median_ms
                << " mean=" << stats.mean_ms << "\n";
    };
    print_stats("score", score_stats);
    print_stats("topk", topk_stats);
    print_stats("total", total_stats);
    std::cout << "score_effective_gflops=" << effective_gflops << "\n";
    std::cout << "unique_index_k_gib_per_s=" << unique_k_gib_per_s << "\n";

    double checksum = 0.0;
    std::cout << "top_indices:";
    for (std::size_t i = 0; i < std::min<std::size_t>(10, topk_result.size()); ++i) {
      std::cout << ' ' << topk_result[i].index;
      checksum += static_cast<double>(topk_result[i].score) * (i + 1);
    }
    std::cout << "\nchecksum=" << checksum << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "fatal: " << error.what() << "\n";
    return 1;
  }
}
