#include "src/feed_forward.h"

namespace {

constexpr
std::array<float, kLayersSize * kHiddenDim * kEmbeddingSize> kFFNWeights1 = {
#include "w1.inc"
};

constexpr
std::array<float, kLayersSize * kEmbeddingSize * kHiddenDim> kFFNWeights2 = {
#include "w2.inc"
};

constexpr
std::array<float, kLayersSize * kHiddenDim * kEmbeddingSize> kFFNWeights3 = {
#include "w3.inc"
};

// The swish activation function.
// What it does is to convert negative elements to a number between (-1, 0)
// while keeping positive values close to what they were.
// With the activation function the linear tranformation becomes non-linear and
// the neutral networks becomes deeper.
template<size_t N>
void Swish(TensorF<N>* x) {
  for (auto& val : *x)
    val /= 1.f + std::exp(-val);
}

}  // namespace

FeedForward::FeedForward(int layer)
    : w1_(kFFNWeights1, layer * kHiddenDim * kEmbeddingSize),
      w2_(kFFNWeights2, layer * kEmbeddingSize * kHiddenDim),
      w3_(kFFNWeights3, layer * kHiddenDim * kEmbeddingSize) {}

TensorF<kEmbeddingSize> FeedForward::Forward(TensorF<kEmbeddingSize> x) const {
  // Compute a "gate" hidden state with swish activation.
  TensorF<kHiddenDim> gate = MatrixProduct(w1_, x);
  Swish(&gate);
  // Compute another hidden state.
  TensorF<kHiddenDim> h = MatrixProduct(w3_, x);
  // Multiply the elements of hidden state with the gates, intuitively this
  // controls how data in attention are filtered.
  for (size_t i = 0; i < kHiddenDim; ++i)
    h[i] *= gate[i];
  // Convert the hidden state into embedding.
  return MatrixProduct(w2_, h);
}
