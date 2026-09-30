#pragma once
// Shared BF16 arithmetic for the single-buffer and NUMA indexer benchmarks.
#include "indexer_features.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#if INDEXER_HAS_SVE_BF16
#include <arm_sve.h>
#endif
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace indexer {

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

inline float dot_bf16_scalar(const std::uint16_t* lhs, const std::uint16_t* rhs, int dim) {
  float sum = 0.0f;
  for (int d = 0; d < dim; ++d) {
    sum += bf16_to_fp32(lhs[d]) * bf16_to_fp32(rhs[d]);
  }
  return sum;
}

#if INDEXER_HAS_SVE_BF16
inline float dot_bf16_native(const std::uint16_t* lhs, const std::uint16_t* rhs, int dim) {
  svfloat32_t accum0 = svdup_n_f32(0.0f);
  svfloat32_t accum1 = svdup_n_f32(0.0f);
  svfloat32_t accum2 = svdup_n_f32(0.0f);
  svfloat32_t accum3 = svdup_n_f32(0.0f);
  const std::uint64_t lanes = svcnth();
  const std::uint64_t elements = static_cast<std::uint64_t>(dim);
  std::uint64_t d = 0;
  for (; d + 4 * lanes <= elements; d += 4 * lanes) {
    const svbool_t predicate = svptrue_b16();
    accum0 = svbfdot_f32(
        accum0,
        svld1_bf16(predicate, reinterpret_cast<const bfloat16_t*>(lhs + d)),
        svld1_bf16(predicate, reinterpret_cast<const bfloat16_t*>(rhs + d)));
    accum1 = svbfdot_f32(
        accum1,
        svld1_bf16(predicate, reinterpret_cast<const bfloat16_t*>(lhs + d + lanes)),
        svld1_bf16(predicate, reinterpret_cast<const bfloat16_t*>(rhs + d + lanes)));
    accum2 = svbfdot_f32(
        accum2,
        svld1_bf16(predicate, reinterpret_cast<const bfloat16_t*>(lhs + d + 2 * lanes)),
        svld1_bf16(predicate, reinterpret_cast<const bfloat16_t*>(rhs + d + 2 * lanes)));
    accum3 = svbfdot_f32(
        accum3,
        svld1_bf16(predicate, reinterpret_cast<const bfloat16_t*>(lhs + d + 3 * lanes)),
        svld1_bf16(predicate, reinterpret_cast<const bfloat16_t*>(rhs + d + 3 * lanes)));
  }

  svfloat32_t accum = svadd_f32_x(
      svptrue_b32(),
      svadd_f32_x(svptrue_b32(), accum0, accum1),
      svadd_f32_x(svptrue_b32(), accum2, accum3));
  for (; d < elements; d += lanes) {
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
inline float dot_bf16_native(const std::uint16_t* lhs, const std::uint16_t* rhs, int dim) {
  float32x4_t accum0 = vdupq_n_f32(0.0f);
  float32x4_t accum1 = vdupq_n_f32(0.0f);
  int d = 0;
  for (; d + 16 <= dim; d += 16) {
    accum0 = vbfdotq_f32(
        accum0,
        vld1q_bf16(reinterpret_cast<const bfloat16_t*>(lhs + d)),
        vld1q_bf16(reinterpret_cast<const bfloat16_t*>(rhs + d)));
    accum1 = vbfdotq_f32(
        accum1,
        vld1q_bf16(reinterpret_cast<const bfloat16_t*>(lhs + d + 8)),
        vld1q_bf16(reinterpret_cast<const bfloat16_t*>(rhs + d + 8)));
  }
  float32x4_t accum = vaddq_f32(accum0, accum1);
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
inline float dot_bf16_native(const std::uint16_t* lhs, const std::uint16_t* rhs, int dim) {
  return dot_bf16_scalar(lhs, rhs, dim);
}
#endif

#if INDEXER_HAS_SVE_BF16
inline float weighted_relu_sum_native(const std::uint16_t* q,
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
inline float32x4_t dot_bf16_4heads_native(const std::uint16_t* q,
                                   const std::uint16_t* key,
                                   int dim) {
  float32x4_t accum0 = vdupq_n_f32(0.0f);
  float32x4_t accum1 = vdupq_n_f32(0.0f);
  float32x4_t accum2 = vdupq_n_f32(0.0f);
  float32x4_t accum3 = vdupq_n_f32(0.0f);
  float32x4_t accum4 = vdupq_n_f32(0.0f);
  float32x4_t accum5 = vdupq_n_f32(0.0f);
  float32x4_t accum6 = vdupq_n_f32(0.0f);
  float32x4_t accum7 = vdupq_n_f32(0.0f);
  int d = 0;
  for (; d + 16 <= dim; d += 16) {
    const bfloat16x8_t key_vec0 =
        vld1q_bf16(reinterpret_cast<const bfloat16_t*>(key + d));
    const bfloat16x8_t key_vec1 =
        vld1q_bf16(reinterpret_cast<const bfloat16_t*>(key + d + 8));
    accum0 = vbfdotq_f32(
        accum0,
        vld1q_bf16(reinterpret_cast<const bfloat16_t*>(q + d)),
        key_vec0);
    accum1 = vbfdotq_f32(
        accum1,
        vld1q_bf16(reinterpret_cast<const bfloat16_t*>(q + dim + d)),
        key_vec0);
    accum2 = vbfdotq_f32(
        accum2,
        vld1q_bf16(reinterpret_cast<const bfloat16_t*>(q + 2 * dim + d)),
        key_vec0);
    accum3 = vbfdotq_f32(
        accum3,
        vld1q_bf16(reinterpret_cast<const bfloat16_t*>(q + 3 * dim + d)),
        key_vec0);
    accum4 = vbfdotq_f32(
        accum4,
        vld1q_bf16(reinterpret_cast<const bfloat16_t*>(q + d + 8)),
        key_vec1);
    accum5 = vbfdotq_f32(
        accum5,
        vld1q_bf16(reinterpret_cast<const bfloat16_t*>(q + dim + d + 8)),
        key_vec1);
    accum6 = vbfdotq_f32(
        accum6,
        vld1q_bf16(reinterpret_cast<const bfloat16_t*>(q + 2 * dim + d + 8)),
        key_vec1);
    accum7 = vbfdotq_f32(
        accum7,
        vld1q_bf16(reinterpret_cast<const bfloat16_t*>(q + 3 * dim + d + 8)),
        key_vec1);
  }

  accum0 = vaddq_f32(accum0, accum4);
  accum1 = vaddq_f32(accum1, accum5);
  accum2 = vaddq_f32(accum2, accum6);
  accum3 = vaddq_f32(accum3, accum7);
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

inline float weighted_relu_sum_native(const std::uint16_t* q,
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
inline float weighted_relu_sum_native(const std::uint16_t* q,
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

inline const char* kernel_name() {
#if INDEXER_HAS_SVE_BF16
  return "Arm SVE BF16 BFDOT";
#elif defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
  return "Arm NEON BF16 BFDOT";
#else
  return "portable scalar BF16";
#endif
}

}  // namespace indexer
