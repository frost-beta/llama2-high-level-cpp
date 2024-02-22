#pragma once

#include "src/model_common.h"

// Convert a token to embedding.
TensorF<kEmbeddingSize> Encode(int token);

// The weights used for encoding embeddings is also used for decoding.
TensorF<kTokensSize> EmbeddingToTokenLogits(TensorViewF<kEmbeddingSize> x);
