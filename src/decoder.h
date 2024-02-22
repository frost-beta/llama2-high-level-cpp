#include "src/feed_forward.h"
#include "src/self_attention.h"

class Decoder {
 public:
  explicit Decoder(int layer);

  TensorF<kEmbeddingSize> Forward(TensorViewF<kEmbeddingSize> x,
                                  size_t position);

 private:
  // The model layers.
  SelfAttention attention_;
  const FeedForward feed_forward_;

  // The model weights.
  const TensorViewF<kEmbeddingSize> attention_norm_;
  const TensorViewF<kEmbeddingSize> feed_forward_norm_;
};
