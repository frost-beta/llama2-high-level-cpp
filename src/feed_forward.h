#include "src/model_common.h"

// The FeedForward layer implements a SwiGLU (Swish Gated Linear Unit).
class FeedForward {
 public:
  explicit FeedForward(int layer);

  TensorF<kEmbeddingSize> Forward(TensorF<kEmbeddingSize> x) const;

 private:
  // The model weights.
  const TensorViewF<kHiddenDim, kEmbeddingSize> w1_;
  const TensorViewF<kEmbeddingSize, kHiddenDim> w2_;
  const TensorViewF<kHiddenDim, kEmbeddingSize> w3_;
};
