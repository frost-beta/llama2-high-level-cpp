# Inference Llama2 with High-Level C++

This is a educational project demonstrating how to inference a Llama2 model with
vanilla C++20.

The only dependency is [SentencePiece](https://github.com/google/sentencepiece)
which is the tokenizer used by Llama2. While writing a tokenizer from scratch
would help understand Llama2 better, I found it off target implementing the
details of SentencePiece.

## Inspirations

This project is highly inspired by the
[llama2.c](https://github.com/karpathy/llama2.c) project and simple PyTorch
Llama2 implementaions like
[hkproj/pytorch-llama](https://github.com/hkproj/pytorch-llama) and
[aju22/LLaMA2](https://github.com/aju22/LLaMA2).

I decided to write a new C++ implementation, because when reading the C
implemenation I was struggled to understand the code because of lack of
abstractions, while the PyTorch implementations were written in a way that uses
`if` and `for` as less as possible.

## Highlights

One difficulty I had understanding the existing Llama2 implementations was to
figure out the dimensions of the tensors being operated, so in this project
tensors are stored in `std::array` and `std::span` with a custom `Tensor`
wrapper implementing multi-dimensional operations, and you can immediately get
the information of tensor by looking at its type, like
`Tensor<float, kEmbeddingSize, kHeadsSize, kHeadDimension>`.

There is no weights loading code - they are written in text files and then
`#include`d in the source code, which makes it much easier to abstract the model
layers with minimal code.

There is almost no heap allocations in the code (except for a few places using
std containers which do it implicitly), weights are defined as globals and
temporary tensors are allocated on stack.

These decisions come with the downside that the code only works with tiny
models, larger ones will result in stack overflows. But I think they serve very
well for code readbilities.

## How to use

```bash
# Check out the code.
git clone --recursive https://github.com/frost-beta/llama2-high-level-cpp.git
cd llama2-high-level-cpp

# Download dependencies.
./scripts/bootstrap.py

# Build.
./scripts/build.py

# Run the model.
./out/Release/frost_run

# You can also run the original llama2.c code for comparisons.
# (Note that it does not work under Windows.)
./out/Release/original_llama2_run stories15M.bin
```

## Files

* `src` - The main code, start from the `inference.cc` file.
* `BUILD.gn` - Build rules.
* `assets` - Store the tokenizer weights.
* `scripts` - Scripts for building the project.
* `secondary` - Build rules for SentencePiece.
* `third_party` - Dependencies.
