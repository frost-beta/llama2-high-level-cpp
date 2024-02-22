#include "src/embedding.h"

namespace {

// The raw model weights.
constexpr
std::array<float, kTokensSize * kEmbeddingSize> kTokenEmbeddingTable = {
#include "token_embedding_table.inc"
};

}  // namespace

TensorF<kEmbeddingSize> Encode(int token) {
  CHECK(token >= 0 && token < kTokensSize);
  TensorF<kEmbeddingSize> result;
  std::copy(kTokenEmbeddingTable.begin() + token * kEmbeddingSize,
            kTokenEmbeddingTable.begin() + (token + 1) * kEmbeddingSize,
            result.begin());
  return result;
}

TensorF<kTokensSize> EmbeddingToTokenLogits(TensorViewF<kEmbeddingSize> x) {
  return MatrixProduct(
      TensorViewF<kTokensSize, kEmbeddingSize>(kTokenEmbeddingTable),
      x);
}
