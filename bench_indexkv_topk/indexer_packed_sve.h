#pragma once
#include "indexer_bf16_kernels.h"
#include <cstddef>

namespace indexer {

// [head, dim] -> [ceil(dim/2), head, 2]. Each FP32 BFDOT lane then
// accumulates a complete head; no horizontal reduction is needed per head.
inline void pack_query(const std::uint16_t* q, std::uint16_t* packed,
                       int heads, int dim) {
  const std::size_t pairs = (static_cast<std::size_t>(dim) + 1) / 2;
  for (std::size_t pair = 0; pair < pairs; ++pair) {
    for (int head = 0; head < heads; ++head) {
      const std::size_t src = static_cast<std::size_t>(head) * dim + 2 * pair;
      const std::size_t dst = (pair * heads + head) * 2;
      packed[dst] = q[src];
      packed[dst + 1] = 2 * pair + 1 < static_cast<std::size_t>(dim) ? q[src + 1] : 0;
    }
  }
}

#if defined(__ARM_FEATURE_SVE_BF16)
inline std::uint32_t key_pair(const std::uint16_t* key, std::size_t d, int dim) {
  // memcpy handles unaligned pairs and the last odd dimension without overread.
  std::uint32_t pair = 0;
  std::memcpy(&pair, key + d, d + 1 < static_cast<std::size_t>(dim) ? 4 : 2);
  return pair;
}

inline void score_four_packed_sve(const std::uint16_t* packed_q,
                                  const std::uint16_t* keys,
                                  const float* weights, float* scores,
                                  int heads, int dim) {
  const svfloat32_t zero = svdup_n_f32(0);
  svfloat32_t score0 = zero, score1 = zero, score2 = zero, score3 = zero;
  const std::uint64_t lanes = svcntw();
  const std::uint16_t* k0 = keys;
  const std::uint16_t* k1 = k0 + dim;
  const std::uint16_t* k2 = k1 + dim;
  const std::uint16_t* k3 = k2 + dim;
  for (std::uint64_t head = 0; head < static_cast<std::uint64_t>(heads); head += lanes) {
    const svbool_t ph = svwhilelt_b32(head, static_cast<std::uint64_t>(heads));
    const svbool_t pq = svwhilelt_b16(head * 2, static_cast<std::uint64_t>(heads) * 2);
    svfloat32_t dot0 = zero, dot1 = zero, dot2 = zero, dot3 = zero;
    for (std::size_t d = 0; d < static_cast<std::size_t>(dim); d += 2) {
      const svbfloat16_t q = svld1_bf16(pq, reinterpret_cast<const bfloat16_t*>(
          packed_q + (d / 2 * heads + head) * 2));
      dot0 = svbfdot_f32(dot0, q, svreinterpret_bf16_u32(svdup_n_u32(key_pair(k0, d, dim))));
      dot1 = svbfdot_f32(dot1, q, svreinterpret_bf16_u32(svdup_n_u32(key_pair(k1, d, dim))));
      dot2 = svbfdot_f32(dot2, q, svreinterpret_bf16_u32(svdup_n_u32(key_pair(k2, d, dim))));
      dot3 = svbfdot_f32(dot3, q, svreinterpret_bf16_u32(svdup_n_u32(key_pair(k3, d, dim))));
    }
    const svfloat32_t w = svld1_f32(ph, weights + head);
    score0 = svmla_f32_m(ph, score0, w, svmax_f32_x(ph, dot0, zero));
    score1 = svmla_f32_m(ph, score1, w, svmax_f32_x(ph, dot1, zero));
    score2 = svmla_f32_m(ph, score2, w, svmax_f32_x(ph, dot2, zero));
    score3 = svmla_f32_m(ph, score3, w, svmax_f32_x(ph, dot3, zero));
  }
  scores[0] = svaddv_f32(svptrue_b32(), score0);
  scores[1] = svaddv_f32(svptrue_b32(), score1);
  scores[2] = svaddv_f32(svptrue_b32(), score2);
  scores[3] = svaddv_f32(svptrue_b32(), score3);
}
#endif

inline void score_range(const std::uint16_t* q, const std::uint16_t* packed_q,
                        const std::uint16_t* keys, const float* weights,
                        float* scores, std::size_t tokens, int heads, int dim,
                        bool use_packed) {
  std::size_t token = 0;
#if defined(__ARM_FEATURE_SVE_BF16)
  if (use_packed) {
    for (; token + 4 <= tokens; token += 4) {
      score_four_packed_sve(packed_q, keys + token * dim, weights,
                            scores + token, heads, dim);
    }
  }
#else
  (void)packed_q;
  (void)use_packed;
#endif
  for (; token < tokens; ++token) {
    scores[token] = weighted_relu_sum_native(q, keys + token * dim, weights, heads, dim);
  }
}

}  // namespace indexer
