// Copied from original_llama2_run.c to export the weights from .bin files.

#include <string>
#include <vector>

#include <stdio.h>
#include <stdlib.h>

struct Config {
  int dim;
  int hidden_dim;
  int n_layers;
  int n_heads;
  int n_kv_heads;
  int vocab_size;
  int seq_len;
};

std::vector<float> ReadFloats(FILE* file, size_t count) {
  std::vector<float> floats(count);
  size_t ret = fread(floats.data(), sizeof(float), count, file);
  if (ret != count)
    exit(3);
  return floats;
}

void WriteFloats(std::string_view filename, const std::vector<float>& floats) {
  FILE* out = fopen(filename.data(), "w");
  for (float f : floats)
    fprintf(out, "%f, ", f);
  fprintf(out, "\n");
  fclose(out);
}

int main(int argc, const char* argv[]) {
  if (argc != 3)
    return 1;

  std::string bin = argv[1];
  std::string dir = argv[2];
  FILE* file = fopen(bin.c_str(), "rb");

  Config config;
  if (fread(&config, sizeof(Config), 1, file) != 1)
    return 2;

  FILE* config_h = fopen((dir + "/model_config.h").c_str(), "w");
  fprintf(config_h, "constexpr int kHiddenDim = %d;\n", config.hidden_dim);
  fprintf(config_h, "constexpr int kLayersSize = %d;\n", config.n_layers);
  fprintf(config_h, "constexpr int kHeadsSize = %d;\n", config.n_heads);
  fprintf(config_h, "constexpr int kKVHeadsSize = %d;\n", config.n_kv_heads);
  fprintf(config_h, "constexpr int kEmbeddingSize = %d;\n", config.dim);
  fprintf(config_h, "constexpr int kSequenceSize = %d;\n", config.seq_len);
  fprintf(config_h, "constexpr int kTokensSize = %d;\n", config.vocab_size);
  fclose(config_h);

  int head_dimension = config.dim / config.n_heads;

  WriteFloats(dir + "/token_embedding_table.inc",
              ReadFloats(file, config.vocab_size * config.dim));
  WriteFloats(dir + "/rms_att_weight.inc",
              ReadFloats(file, config.n_layers * config.dim));
  WriteFloats(dir + "/wq.inc",
              ReadFloats(file, config.n_layers * config.dim *
                               config.n_heads * head_dimension));
  WriteFloats(dir + "/wk.inc",
              ReadFloats(file, config.n_layers * config.dim *
                               config.n_kv_heads * head_dimension));
  WriteFloats(dir + "/wv.inc",
              ReadFloats(file, config.n_layers * config.dim *
                               config.n_kv_heads * head_dimension));
  WriteFloats(dir + "/wo.inc",
              ReadFloats(file, config.n_layers * config.dim * config.dim));
  WriteFloats(dir + "/rms_ffn_weight.inc",
              ReadFloats(file, config.n_layers * config.dim));
  WriteFloats(dir + "/w1.inc",
              ReadFloats(file, config.n_layers *
                               config.dim * config.hidden_dim));
  WriteFloats(dir + "/w2.inc",
              ReadFloats(file, config.n_layers *
                               config.hidden_dim * config.dim));
  WriteFloats(dir + "/w3.inc",
              ReadFloats(file, config.n_layers *
                               config.dim * config.hidden_dim));
  WriteFloats(dir + "/rms_out_weight.inc",
              ReadFloats(file, config.dim));

  fclose(file);

  return 0;
}
