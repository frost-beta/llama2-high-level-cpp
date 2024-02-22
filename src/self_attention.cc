#include "src/self_attention.h"

#include <complex>
#include <vector>

using namespace frost;

namespace {

constexpr
std::array<float, kLayersSize * kEmbeddingSize * kHeadsSize * kHeadDimension>
kAttentionQueryWeights = {
#include "wq.inc"
};

constexpr
std::array<float, kLayersSize * kEmbeddingSize * kKVHeadsSize * kHeadDimension>
kAttentionKeyWeights = {
#include "wk.inc"
};

constexpr
std::array<float, kLayersSize * kEmbeddingSize * kKVHeadsSize * kHeadDimension>
kAttentionValueWeights = {
#include "wv.inc"
};

constexpr
std::array<float, kLayersSize * kEmbeddingSize * kEmbeddingSize>
kAttentionOutputWeights = {
#include "wo.inc"
};

// Implement RoPE (Rotary Position Embedding) with complex numbers, which
// essentially transforms the embeddings into points on complex-plane, then use
// another complex number with magnitue of 1 to rotate it, and convert the
// points back to embeddings.
template<template<typename, size_t> typename S, typename T, size_t N>
void ApplyRotaryEmbeddings(size_t position, TensorBase<S, T, N>* x) {
  static_assert(N % 2 == 0);
  for (size_t i = 0; i < N; i += 2) {
    std::complex<T> sibling((*x)[i], (*x)[i + 1]);
    float theta = std::pow(10000.f, -1.f * i / kHeadDimension);
    std::complex<T> frequency = std::polar(1.f, position * theta);
    std::complex<T> rotated = sibling * frequency;
    (*x)[i] = rotated.real();
    (*x)[i + 1] = rotated.imag();
  }
}

}  // namespace

SelfAttention::SelfAttention(int layer)
    : wq_(kAttentionQueryWeights,
          layer * kEmbeddingSize * kHeadsSize * kHeadDimension),
      wk_(kAttentionKeyWeights,
          layer * kEmbeddingSize * kKVHeadsSize * kHeadDimension),
      wv_(kAttentionValueWeights,
          layer * kEmbeddingSize * kKVHeadsSize * kHeadDimension),
      wo_(kAttentionOutputWeights,
          layer * kEmbeddingSize * kEmbeddingSize) {}

TensorF<kEmbeddingSize> SelfAttention::Forward(TensorF<kEmbeddingSize> x,
                                               size_t position) {
  // The kHeadsSize is how many heads an attention layer has, the kHeadDimension
  // is the size of partial embedding that a head is responsible for.
  static_assert(kHeadDimension == kEmbeddingSize / kHeadsSize);
  // Compute queries for all heads at the |position|.
  TensorF<kHeadsSize * kHeadDimension> queries = MatrixProduct(wq_, x);

  // In grouped attentions, the keys and values have less heads than queries,
  static_assert(kHeadsSize % kKVHeadsSize == 0);
  // Compute keys and values for all heads at the |position| and remember the
  // results to cache.
  MutableTensorViewF<kKVHeadsSize * kHeadDimension> keys =
      keys_cache_[position];
  MatrixProductTo(wk_, x, &keys);
  MutableTensorViewF<kKVHeadsSize * kHeadDimension> values =
      values_cache_[position];
  MatrixProductTo(wv_, x, &values);

  // Reshape the vectors to multi-dimensional tensors to ease computation.
  auto xq = queries.ViewAs<kHeadsSize, kHeadDimension>();
  auto xk = keys_cache_.ViewAs<kSequenceSize, kKVHeadsSize, kHeadDimension>();
  auto xv = values_cache_.ViewAs<kSequenceSize, kKVHeadsSize, kHeadDimension>();

  // For each query and value at each head, apply RoPE positional encoding.
  for (size_t i = 0; i < kHeadsSize; ++i) {
    MutableTensorViewF<kHeadDimension> each = xq[i];
    ApplyRotaryEmbeddings(position, &each);
  }
  for (size_t i = 0; i < kKVHeadsSize; ++i) {
    MutableTensorViewF<kHeadDimension> each = xk[position][i];
    ApplyRotaryEmbeddings(position, &each);
  }

  // Compute grouped attention.
  for (size_t head = 0; head < kHeadsSize; ++head) {
    TensorViewF<kHeadDimension> query = xq[head];

    // Calculate scores for all positions in this head.
    std::vector<float> scores(position + 1);
    for (size_t past = 0; past <= position; ++past) {
      // Multiple heads share the same keys/values in grouped attention.
      size_t key_index = head / (kHeadsSize / kKVHeadsSize);
      TensorViewF<kHeadDimension> key = xk[past][key_index];
      scores[past] = DotProduct(query, key) / std::sqrt(kHeadDimension);
    }
    // Make the scores sum up to 1.
    Softmax(scores.begin(), scores.end());

    // Write the weighted value to x.
    MutableTensorViewF<kHeadDimension> output(x, head * kHeadDimension);
    std::fill(output.begin(), output.end(), 0);
    for (size_t past = 0; past <= position; ++past) {
      size_t value_index = head / (kHeadsSize / kKVHeadsSize);
      TensorViewF<kHeadDimension> value = xv[past][value_index];
      float score = scores[past];
      for (size_t i = 0; i < kHeadDimension; ++i) {
        output[i] += value[i] * score;
      }
    }
  }

  return MatrixProduct(wo_, x);
}
