#include "src/model_common.h"

class SelfAttention {
 public:
  explicit SelfAttention(int layer);

  TensorF<kEmbeddingSize> Forward(TensorF<kEmbeddingSize> x, size_t position);

 private:
  // The model weights.
  const TensorViewF<kEmbeddingSize, kHeadsSize * kHeadDimension> wq_;
  const TensorViewF<kEmbeddingSize, kKVHeadsSize * kHeadDimension> wk_;
  const TensorViewF<kEmbeddingSize, kKVHeadsSize * kHeadDimension> wv_;
  const TensorViewF<kEmbeddingSize, kEmbeddingSize> wo_;

  // Computed keys and values.
  TensorF<kSequenceSize, kKVHeadsSize * kHeadDimension> keys_cache_;
  TensorF<kSequenceSize, kKVHeadsSize * kHeadDimension> values_cache_;
};
