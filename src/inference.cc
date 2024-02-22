#include <chrono>
#include <iostream>
#include <random>

#include "src/embedding.h"
#include "src/transformer.h"
#include "third_party/sentencepiece/src/sentencepiece_processor.h"

namespace {

// Return an index of element using top-p algorithm.
template<size_t N>
int SampleTopP(TensorViewF<N> probabilities, float p) {
  static_assert(N > 2);
  // Ignore the probability if it is less than cutoff.
  float cutoff = (1.f - p) / (N - 1);
  // Sort the elements of probabilities into a new vector.
  std::vector<std::pair<float, size_t>> sorted;
  for (size_t i = 0; i < N; i++) {
    if (probabilities[i] >= cutoff)
      sorted.push_back({probabilities[i], i});
  }
  std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
    return a.first > b.first;
  });
  CHECK_GT(sorted.size(), 0);

  // Calculate cumulative probabilities.
  float total = 0.f;
  size_t index = sorted.size() - 1;
  for (size_t i = 0; i < sorted.size(); i++) {
    total += sorted[i].first;
    if (total >= p) {
      index = i;
      break;
    }
  }

  // Get a random number between 0 and 1.
  static std::random_device device;
  static std::default_random_engine engine(device());
  std::uniform_real_distribution<float> uniform_dist(0.f, 1.f);
  float random = uniform_dist(engine);

  // Cumulative distribution function.
  float r = random * total;
  float cdf = 0.f;
  for (size_t i = 0; i <= index; i++) {
    cdf += sorted[i].first;
    if (r < cdf)
      return sorted[i].second;
  }
  return sorted[index].second;
}

}  // namespace

int main(int argc, const char *argv[]) {
  // Load llama2 sentencepiece model.
  sentencepiece::SentencePieceProcessor processor;
  const auto status = processor.Load("assets/tokenizer.model");
  if (!status.ok()) {
    std::cerr << "Failed to load tokenizer: " << status.ToString() << std::endl;
    return 1;
  }
  if (processor.GetPieceSize() != kTokensSize) {
    return 2;
  }

  Transformer transformer;

  // Get the token for a single character "i", the character itself does not
  // have any meaning. See Decode code below for more.
  std::vector<int> dummy;
  processor.Encode("i", &dummy);

  auto start_time = std::chrono::high_resolution_clock::now();

  // Start feeding tokens into transformer, the first token is always BOS.
  int token = processor.bos_id();
  size_t position = 0;
  while (position < kSequenceSize) {
    // Encode the token into an embedding and feed it to transformer.
    TensorF<kTokensSize> logits = transformer.Forward(Encode(token), position);
    Softmax(logits.begin(), logits.end());

    // Sample the result to predict the next token.
    token = SampleTopP(logits.View(), 0.9);

    // End of sequence.
    if (token == processor.eos_id() || token == processor.bos_id())
      break;

    // Decode the token into text.
    std::string piece;
    std::string_view result;
    if (position == 0) {
      processor.Decode({token}, &piece);
      result = piece;
    } else {
      // When decoding, sentencepiece strips the leading spaces for the first
      // token, which results in all spaces getting removed when decoding token
      // by token. We work around this by prepending a dummy token and skipping
      // the first character in the result.
      processor.Decode({dummy[0], token}, &piece);
      result = std::string_view(piece.begin() + 1, piece.end());
    }
    std::cout << result << std::flush;

    position++;
  }
  std::cout << std::endl;

  // Count time used for token generation.
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float> elapsed = end_time - start_time;
  CHECK_GT(position, 0);
  std::cout << "achieved tok/s: " << (position / elapsed.count()) << std::endl;

  return 0;
}
