#include "src/embedding.h"
#include "src/transformer.h"

namespace {

// The weights are stored as std::array literal.
// Note that though it is very tempting to store the weights as Tensor directly,
// doing so would crash clang.
constexpr std::array<float, kEmbeddingSize> kNormOutWeights = {
#include "rms_out_weight.inc"
};

// Helper to constructor decoders with layer numbers, i.e.
// return std::array<Decoder, 3>{Decoder(0), Decoder(1), Decoder(2)};
template<size_t... N>
constexpr auto MakeDecoders(std::index_sequence<N...>) {
  return std::array<Decoder, sizeof...(N)>{Decoder(N)...};
}

}  // namespace

Transformer::Transformer()
    : decoders_(MakeDecoders(std::make_index_sequence<kLayersSize>())),
      norm_weights_(kNormOutWeights) {}

TensorF<kTokensSize> Transformer::Forward(TensorF<kEmbeddingSize> x,
                                          size_t position) {
  // Feed the embedding through encoder blocks.
  for (size_t i = 0; i < kLayersSize; ++i)
    x = decoders_[i].Forward(x, position);
  // Normalize the result and convert it to logits, which is a vector with each
  // element representing how likely its index might be the next token.
  x = RMSNormalize(x.View(), norm_weights_);
  return EmbeddingToTokenLogits(x);
}
