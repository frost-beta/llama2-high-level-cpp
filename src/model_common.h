#pragma once

#include <algorithm>
#include <cmath>

#include "model_config.h"  // generated header
#include "src/tensor.h"

using frost::Tensor;
using frost::TensorF;
using frost::TensorViewF;

// In multi-head attention, each head is only responsible for transforming a
// part of the embedding, and kHeadDimension is the size of the part.
constexpr
size_t kHeadDimension = kEmbeddingSize / kHeadsSize;

// Re-scale the scalars of |x| with Root Mean Square Normalization, so the scalars
// won't be too large or too small.
template<size_t N>
TensorF<N> RMSNormalize(TensorViewF<N> x, TensorViewF<N> weights) {
  float sum_of_squres = 0;
  for (size_t i = 0; i < N; ++i)
    sum_of_squres += x[i] * x[i];
  // The constant is used by LLaMa2 to prevent running sqrt(0).
  float rms = std::sqrt(sum_of_squres / N + 1e-5f);
  TensorF<N> result;
  for (size_t i = 0; i < N; ++i)
    result[i] = weights[i] * x[i] / rms;
  return result;
}

// Convert a vector of scalars to a probability distribution.
template<typename Iter>
void Softmax(Iter first, Iter last) {
  using T = std::remove_reference_t<decltype(*first)>;
  T max_val = *std::max_element(first, last);
  T sum = T();
  for (Iter it = first; it != last; ++it) {
    *it = std::exp(*it - max_val);
    sum += *it;
  }
  for (Iter it = first; it != last; ++it) {
    *it /= sum;
  }
}
