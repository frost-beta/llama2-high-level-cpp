#include "src/decoder.h"

namespace {

constexpr
std::array<float, kLayersSize * kEmbeddingSize> kNormAttWeights = {
#include "rms_att_weight.inc"
};

constexpr
std::array<float, kLayersSize * kEmbeddingSize> kNormFFNWeights = {
#include "rms_ffn_weight.inc"
};

}  // namespace

Decoder::Decoder(int layer)
    : attention_(layer),
      feed_forward_(layer),
      attention_norm_(kNormAttWeights, layer * kEmbeddingSize),
      feed_forward_norm_(kNormFFNWeights, layer * kEmbeddingSize) {}

TensorF<kEmbeddingSize> Decoder::Forward(TensorViewF<kEmbeddingSize> x,
                                         size_t position) {
  TensorF<kEmbeddingSize> h = attention_.Forward(
      RMSNormalize(x, attention_norm_), position);

  // Residual block.
  for (size_t j = 0; j < kEmbeddingSize; ++j)
    h[j] += x[j];

  TensorF<kEmbeddingSize> result = feed_forward_.Forward(
      RMSNormalize(h.View(), feed_forward_norm_));

  // Residual block.
  for (size_t j = 0; j < kEmbeddingSize; ++j)
    result[j] += h[j];

  return result;
}
