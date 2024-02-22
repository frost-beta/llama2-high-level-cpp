#include "src/decoder.h"

class Transformer {
 public:
  Transformer();

  TensorF<kTokensSize> Forward(TensorF<kEmbeddingSize> x, size_t position);

 private:
  // The model layers.
  std::array<Decoder, kLayersSize> decoders_;

  // The model weights.
  const TensorViewF<kEmbeddingSize> norm_weights_;
};
