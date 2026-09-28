// Standalone CPU benchmark for the DeepSeek-V3.2 DSA indexer score.
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
#include <pthread.h>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

#if defined(__ARM_FEATURE_SVE_BF16)
#include <arm_sve.h>
#endif

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
  std::int64_t seq_len = 128 * 1024;
  int heads = 64;
  int dim = 128;
  int threads = 0;
  int warmup = 3;
  int iters = 10;
  bool check = false;
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
      << "  --threads N   pthread workers (default: online CPU count)\n"
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

#if defined(__ARM_FEATURE_SVE_BF16)
float weighted_relu_sum_native(const std::uint16_t* q,
                               const std::uint16_t* key,
                               const float* weights,
                               int heads,
                               int dim) {
  // SVE implementations can expose at most 2048 bits, or 64 FP32 lanes.
  alignas(256) float dots[64];
  const std::uint64_t lanes = svcntw();
  svfloat32_t score = svdup_n_f32(0.0f);

  for (std::uint64_t head = 0; head < static_cast<std::uint64_t>(heads); head += lanes) {
    const std::uint64_t active =
        std::min(lanes, static_cast<std::uint64_t>(heads) - head);
    for (std::uint64_t lane = 0; lane < active; ++lane) {
      dots[lane] = dot_bf16_native(q + (head + lane) * dim, key, dim);
    }

    const svbool_t predicate = svwhilelt_b32(std::uint64_t{0}, active);
    const svfloat32_t dot_vec = svld1_f32(predicate, dots);
    const svfloat32_t weight_vec = svld1_f32(predicate, weights + head);
    const svfloat32_t relu_vec =
        svmax_f32_x(predicate, dot_vec, svdup_n_f32(0.0f));
    score = svmla_f32_m(predicate, score, weight_vec, relu_vec);
  }

  return svaddv_f32(svptrue_b32(), score);
}
#elif defined(__aarch64__)

#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
float32x4_t dot_bf16_4heads_native(const std::uint16_t* q,
                                   const std::uint16_t* key,
                                   int dim) {
  float32x4_t accum0 = vdupq_n_f32(0.0f);
  float32x4_t accum1 = vdupq_n_f32(0.0f);
  float32x4_t accum2 = vdupq_n_f32(0.0f);
  float32x4_t accum3 = vdupq_n_f32(0.0f);
  int d = 0;
  for (; d + 8 <= dim; d += 8) {
    const bfloat16x8_t key_vec =
        vld1q_bf16(reinterpret_cast<const bfloat16_t*>(key + d));
    accum0 = vbfdotq_f32(
        accum0,
        vld1q_bf16(reinterpret_cast<const bfloat16_t*>(q + d)),
        key_vec);
    accum1 = vbfdotq_f32(
        accum1,
        vld1q_bf16(reinterpret_cast<const bfloat16_t*>(q + dim + d)),
        key_vec);
    accum2 = vbfdotq_f32(
        accum2,
        vld1q_bf16(reinterpret_cast<const bfloat16_t*>(q + 2 * dim + d)),
        key_vec);
    accum3 = vbfdotq_f32(
        accum3,
        vld1q_bf16(reinterpret_cast<const bfloat16_t*>(q + 3 * dim + d)),
        key_vec);
  }

  // Pairwise reductions leave one dot product in each FP32 lane.
  const float32x4_t pair01 = vpaddq_f32(accum0, accum1);
  const float32x4_t pair23 = vpaddq_f32(accum2, accum3);
  float32x4_t dots = vpaddq_f32(pair01, pair23);
  if (d < dim) {
    alignas(16) float dot_lanes[4];
    vst1q_f32(dot_lanes, dots);
    for (; d < dim; ++d) {
      const float key_value = bf16_to_fp32(key[d]);
      dot_lanes[0] += bf16_to_fp32(q[d]) * key_value;
      dot_lanes[1] += bf16_to_fp32(q[dim + d]) * key_value;
      dot_lanes[2] += bf16_to_fp32(q[2 * dim + d]) * key_value;
      dot_lanes[3] += bf16_to_fp32(q[3 * dim + d]) * key_value;
    }
    dots = vld1q_f32(dot_lanes);
  }
  return dots;
}
#endif

float weighted_relu_sum_native(const std::uint16_t* q,
                               const std::uint16_t* key,
                               const float* weights,
                               int heads,
                               int dim) {
  const float32x4_t zero = vdupq_n_f32(0.0f);
  float32x4_t score = zero;
  int head = 0;
  for (; head + 4 <= heads; head += 4) {
#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
    const float32x4_t dot_vec =
        dot_bf16_4heads_native(q + static_cast<std::int64_t>(head) * dim, key, dim);
#else
    alignas(16) float dot_lanes[4];
    for (int lane = 0; lane < 4; ++lane) {
      dot_lanes[lane] = dot_bf16_native(
          q + static_cast<std::int64_t>(head + lane) * dim, key, dim);
    }
    const float32x4_t dot_vec = vld1q_f32(dot_lanes);
#endif
    score = vfmaq_f32(score,
                      vld1q_f32(weights + head),
                      vmaxq_f32(dot_vec, zero));
  }

  if (head < heads) {
    alignas(16) float dot_lanes[4] = {};
    alignas(16) float weight_lanes[4] = {};
    for (int lane = 0; head + lane < heads; ++lane) {
      dot_lanes[lane] = dot_bf16_native(
          q + static_cast<std::int64_t>(head + lane) * dim, key, dim);
      weight_lanes[lane] = weights[head + lane];
    }
    score = vfmaq_f32(score,
                      vld1q_f32(weight_lanes),
                      vmaxq_f32(vld1q_f32(dot_lanes), zero));
  }

  return vaddvq_f32(score);
}
#else
float weighted_relu_sum_native(const std::uint16_t* q,
                               const std::uint16_t* key,
                               const float* weights,
                               int heads,
                               int dim) {
  float score = 0.0f;
  for (int head = 0; head < heads; ++head) {
    const float dot =
        dot_bf16_native(q + static_cast<std::int64_t>(head) * dim, key, dim);
    score += weights[head] * std::max(dot, 0.0f);
  }
  return score;
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

int online_cpu_count() {
  const long count = sysconf(_SC_NPROCESSORS_ONLN);
  return count > 0 && count <= std::numeric_limits<int>::max()
             ? static_cast<int>(count)
             : 1;
}

void initialize_inputs(std::vector<std::uint16_t>& q,
                       std::vector<std::uint16_t>& index_k,
                       std::vector<float>& weights) {
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

class PthreadScoreExecutor {
 public:
  PthreadScoreExecutor(const std::vector<std::uint16_t>& q,
                       const std::vector<std::uint16_t>& index_k,
                       const std::vector<float>& weights,
                       std::vector<float>& scores,
                       int heads,
                       int dim,
                       int thread_count)
      : q_(q.data()),
        index_k_(index_k.data()),
        weights_(weights.data()),
        scores_(scores.data()),
        seq_len_(scores.size()),
        heads_(heads),
        dim_(dim),
        thread_count_(thread_count) {
    if (thread_count_ <= 0) {
      throw std::invalid_argument("pthread worker count must be positive");
    }
    threads_.resize(static_cast<std::size_t>(thread_count_));
    contexts_.resize(static_cast<std::size_t>(thread_count_));
    int error = pthread_mutex_init(&mutex_, nullptr);
    if (error != 0) {
      throw_pthread_error("pthread_mutex_init", error);
    }
    error = pthread_cond_init(&start_condition_, nullptr);
    if (error != 0) {
      pthread_mutex_destroy(&mutex_);
      throw_pthread_error("pthread_cond_init(start)", error);
    }
    error = pthread_cond_init(&done_condition_, nullptr);
    if (error != 0) {
      pthread_cond_destroy(&start_condition_);
      pthread_mutex_destroy(&mutex_);
      throw_pthread_error("pthread_cond_init(done)", error);
    }

    int created = 0;
    for (; created < thread_count_; ++created) {
      contexts_[static_cast<std::size_t>(created)] = {this, created};
      error = pthread_create(&threads_[static_cast<std::size_t>(created)],
                             nullptr,
                             &PthreadScoreExecutor::worker_entry,
                             &contexts_[static_cast<std::size_t>(created)]);
      if (error != 0) {
        stop_and_join(created);
        pthread_cond_destroy(&done_condition_);
        pthread_cond_destroy(&start_condition_);
        pthread_mutex_destroy(&mutex_);
        throw_pthread_error("pthread_create", error);
      }
    }
  }

  PthreadScoreExecutor(const PthreadScoreExecutor&) = delete;
  PthreadScoreExecutor& operator=(const PthreadScoreExecutor&) = delete;

  ~PthreadScoreExecutor() {
    stop_and_join(thread_count_);
    pthread_cond_destroy(&done_condition_);
    pthread_cond_destroy(&start_condition_);
    pthread_mutex_destroy(&mutex_);
  }

  void compute_scores() {
    pthread_mutex_lock(&mutex_);
    completed_workers_ = 0;
    ++generation_;
    pthread_cond_broadcast(&start_condition_);
    while (completed_workers_ != thread_count_) {
      pthread_cond_wait(&done_condition_, &mutex_);
    }
    pthread_mutex_unlock(&mutex_);
  }

  int thread_count() const { return thread_count_; }

 private:
  struct WorkerContext {
    PthreadScoreExecutor* executor = nullptr;
    int thread_id = 0;
  };

  [[noreturn]] static void throw_pthread_error(const char* operation, int error) {
    throw std::runtime_error(std::string(operation) + ": " + std::strerror(error));
  }

  static void* worker_entry(void* argument) {
    auto* context = static_cast<WorkerContext*>(argument);
    context->executor->worker_loop(context->thread_id);
    return nullptr;
  }

  std::size_t token_boundary(int boundary) const {
    // Equivalent to floor(seq_len * boundary / thread_count), without making
    // the multiplication overflow for very large sequence lengths.
    const std::size_t quotient = seq_len_ / static_cast<std::size_t>(thread_count_);
    const std::size_t remainder = seq_len_ % static_cast<std::size_t>(thread_count_);
    const std::size_t position = static_cast<std::size_t>(boundary);
    return quotient * position + remainder * position / thread_count_;
  }

  void worker_loop(int thread_id) {
    std::uint64_t observed_generation = 0;
    pthread_mutex_lock(&mutex_);
    while (true) {
      while (!stop_ && observed_generation == generation_) {
        pthread_cond_wait(&start_condition_, &mutex_);
      }
      if (stop_) {
        pthread_mutex_unlock(&mutex_);
        return;
      }
      observed_generation = generation_;
      pthread_mutex_unlock(&mutex_);

      const std::size_t begin = token_boundary(thread_id);
      const std::size_t end = token_boundary(thread_id + 1);
      for (std::size_t token = begin; token < end; ++token) {
        const std::uint16_t* key = index_k_ + token * dim_;
        scores_[token] =
            weighted_relu_sum_native(q_, key, weights_, heads_, dim_);
      }

      pthread_mutex_lock(&mutex_);
      ++completed_workers_;
      if (completed_workers_ == thread_count_) {
        pthread_cond_signal(&done_condition_);
      }
    }
  }

  void stop_and_join(int created_threads) {
    pthread_mutex_lock(&mutex_);
    stop_ = true;
    pthread_cond_broadcast(&start_condition_);
    pthread_mutex_unlock(&mutex_);
    for (int i = 0; i < created_threads; ++i) {
      pthread_join(threads_[static_cast<std::size_t>(i)], nullptr);
    }
  }

  const std::uint16_t* q_;
  const std::uint16_t* index_k_;
  const float* weights_;
  float* scores_;
  std::size_t seq_len_;
  int heads_;
  int dim_;
  int thread_count_;
  std::vector<pthread_t> threads_;
  std::vector<WorkerContext> contexts_;
  pthread_mutex_t mutex_{};
  pthread_cond_t start_condition_{};
  pthread_cond_t done_condition_{};
  std::uint64_t generation_ = 0;
  int completed_workers_ = 0;
  bool stop_ = false;
};

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

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const int thread_count =
        options.threads > 0 ? options.threads : online_cpu_count();

    const std::size_t q_elements =
        static_cast<std::size_t>(options.heads) * options.dim;
    const std::size_t k_elements =
        static_cast<std::size_t>(options.seq_len) * options.dim;
    std::vector<std::uint16_t> q(q_elements);
    std::vector<std::uint16_t> index_k(k_elements);
    std::vector<float> weights(options.heads);
    std::vector<float> scores(static_cast<std::size_t>(options.seq_len));

    initialize_inputs(q, index_k, weights);
    PthreadScoreExecutor score_executor(
        q, index_k, weights, scores, options.heads, options.dim, thread_count);

    std::cout << "kernel=" << kernel_name() << "\n";
    std::cout << "threads=" << score_executor.thread_count();
#if defined(__ARM_FEATURE_SVE_BF16)
    std::cout << " sve_bits=" << svcntb() * 8;
#endif
    std::cout << "\n";
    std::cout << "shape: q=[" << options.heads << ',' << options.dim
              << "] index_k=[" << options.seq_len << ',' << options.dim << "]\n";
    std::cout << "index_k_size_mib=" << std::fixed << std::setprecision(2)
              << (index_k.size() * sizeof(std::uint16_t) / 1048576.0) << "\n";

    for (int i = 0; i < options.warmup; ++i) {
      score_executor.compute_scores();
    }

    std::vector<double> score_samples;
    score_samples.reserve(options.iters);

    for (int i = 0; i < options.iters; ++i) {
      const auto score_begin = Clock::now();
      score_executor.compute_scores();
      const auto score_end = Clock::now();
      score_samples.push_back(elapsed_ms(score_begin, score_end));
    }

    if (options.check) {
      check_scores(q, index_k, weights, scores, options.heads, options.dim);
    }

    const TimingStats score_stats = summarize(score_samples);
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
    std::cout << "score_effective_gflops=" << effective_gflops << "\n";
    std::cout << "unique_index_k_gib_per_s=" << unique_k_gib_per_s << "\n";

    const double checksum =
        std::accumulate(scores.begin(), scores.end(), 0.0);
    std::cout << "checksum=" << checksum << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "fatal: " << error.what() << "\n";
    return 1;
  }
}
