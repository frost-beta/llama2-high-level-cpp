#pragma once

#include <array>
#include <cstdio>
#include <span>
#include <utility>

// Runtime checks.
#if !defined(CHECK)
#define UNLIKELY(expr) __builtin_expect(!!(expr), 0)
#define ERROR_AND_ABORT(expr)                                                 \
  do {                                                                        \
    if (!(expr)) {                                                            \
      fprintf(stderr, "Aborted: %s\n", #expr);                                \
      __builtin_trap();                                                       \
    }                                                                         \
  } while (0)
#define CHECK(expr)                                                           \
  do {                                                                        \
    if (UNLIKELY(!(expr))) {                                                  \
      ERROR_AND_ABORT(expr);                                                  \
    }                                                                         \
  } while (0)
#define CHECK_EQ(a, b) CHECK((a) == (b))
#define CHECK_GE(a, b) CHECK((a) >= (b))
#define CHECK_GT(a, b) CHECK((a) > (b))
#define CHECK_LE(a, b) CHECK((a) <= (b))
#define CHECK_LT(a, b) CHECK((a) < (b))
#define CHECK_NE(a, b) CHECK((a) != (b))
#endif

namespace frost {

// A multi-dimensional wrapper of std::span/std::array.
template<template<typename, size_t> typename S, typename T,
         size_t N1, size_t... N>
class TensorBase;

// Mutable version that has the ownership of the data.
template<typename T, size_t... N>
using Tensor = TensorBase<std::array, T, N...>;

// Immutable version that does not copy the data.
// Note that we are adding const to T, because the view is immutable, and it
// is required when the viewed array is a constexpr.
template<typename T, size_t... N>
using TensorView = TensorBase<std::span, const T, N...>;

// Muttable version of TensorView.
template<typename T, size_t... N>
using MutableTensorView = TensorBase<std::span, std::remove_const_t<T>, N...>;

// Aliases for numeric types.
// The "F" suffix means float.
template<size_t... N>
using TensorF = Tensor<float, N...>;
template<size_t... N>
using TensorViewF = TensorView<float, N...>;
template<size_t... N>
using MutableTensorViewF = MutableTensorView<float, N...>;

namespace helper {

// Compute the length needed to store a multi-dimensional tensor.
template<size_t N1, size_t... N>
constexpr size_t Multiplydimensions(std::index_sequence<N1, N...>) {
  size_t length = N1;
  ((length *= N), ...);
  return length;
}

// Get the return type for index operator of TensorBase.
template<template<typename, size_t> typename S, typename T, typename U>
struct GetResultOfIndex {
};

template<template<typename, size_t> typename S, typename T, size_t N1>
struct GetResultOfIndex<S, T, std::index_sequence<N1>> {
  // For 1-dimensional vector return the scalar value.
  using type = T&;
};

template<template<typename, size_t> typename S, typename T,
         size_t N1, size_t... N>
struct GetResultOfIndex<S, T, std::index_sequence<N1, N...>> {
  // For multi-dimensional tensors return sub-tensor.
  using type = TensorBase<std::span, T, N...>;
};

}  // namespace helper

template<template<typename, size_t> typename S, typename T,
         size_t N1, size_t... N>
class TensorBase {
 public:
  static constexpr size_t size = N1;
  static constexpr size_t storage_size =
      helper::Multiplydimensions(std::index_sequence<N1, N...>());

  // Default constructor, note that the data is NOT zero-intialized.
  constexpr TensorBase() {}

  // Create from std::array or std::span from |offset|.
  template<template<typename, size_t> typename SourceStorage,
           typename SourceType, size_t SourceSize>
  constexpr TensorBase(SourceStorage<SourceType, SourceSize>& source,
                       size_t offset = 0)
      : data_(source.begin() + offset, storage_size) {
    CHECK_LE(offset + storage_size, SourceSize);
  }

  // Another version that takes const reference.
  template<template<typename, size_t> typename SourceStorage,
           typename SourceType, size_t SourceSize>
  constexpr TensorBase(const SourceStorage<SourceType, SourceSize>& source,
                       size_t offset = 0)
      : data_(source.begin() + offset, storage_size) {
    CHECK_LE(offset + storage_size, SourceSize);
    // A std::span is immutable when taking from a const source.
    if constexpr (std::is_same_v<S<T, storage_size>,
                                 std::span<T, storage_size>>) {
      static_assert(std::is_const_v<T>);
    }
  }

  // Create from tensor from |offset|.
  template<template<typename, size_t> typename SourceS,
           typename SourceT, size_t... SourceN>
  constexpr TensorBase(TensorBase<SourceS, SourceT, SourceN...>& source,
                       size_t offset = 0)
      : data_(source.data_.begin() + offset, storage_size) {
    CHECK_LE(offset + storage_size, source.storage_size);
  }

  // Another version that takes const reference.
  template<template<typename, size_t> typename SourceS,
           typename SourceT, size_t... SourceN>
  constexpr TensorBase(const TensorBase<SourceS, SourceT, SourceN...>& source,
                       size_t offset = 0)
      : data_(source.data_.begin() + offset, storage_size) {
    CHECK_LE(offset + storage_size, source.storage_size);
    // A std::span is immutable when taking from a const source.
    if constexpr (std::is_same_v<S<T, storage_size>,
                                 std::span<T, storage_size>>) {
      static_assert(std::is_const_v<T>);
    }
  }

  // Supports copy and move.
  constexpr TensorBase(const TensorBase& other) = default;
  constexpr TensorBase(TensorBase&& other) = default;
  TensorBase& operator=(TensorBase& other) = default;
  TensorBase& operator=(TensorBase&& other) = default;

  // Return a view with new shape.
  template<size_t D1, size_t... D>
  constexpr auto ViewAs() {
    TensorBase<std::span, T, D1, D...> result(data_, 0);
    static_assert(storage_size == result.storage_size);
    return result;
  }

  // Const version of View() that returns an immutable view.
  template<size_t D1, size_t... D>
  constexpr auto ViewAs() const {
    TensorBase<std::span, const T, D1, D...> result(data_, 0);
    static_assert(storage_size == result.storage_size);
    return result;
  }

  // Helper to a view of same dimensions.
  constexpr TensorView<T, N1, N...> View() const {
    return TensorView<T, N1, N...>(*this, 0);
  }

  // The returned type of tensor[x] depends on the dimensions of the tensor.
  using IndexResultType =
      typename helper::GetResultOfIndex<
          S, T, std::index_sequence<N1, N...>>::type;

  constexpr IndexResultType operator[] (size_t i) {
    if constexpr (sizeof...(N) == 0) {
      return data_[i];
    } else {
      return IndexResultType(data_, storage_size / N1 * i);
    }
  }

  constexpr auto operator[] (size_t i) const {
    if constexpr (sizeof...(N) == 0) {
      return data_[i];
    } else {
      // Add const to T in the returned tensor.
      return TensorView<const T, N...>(data_, storage_size / N1 * i);
    }
  }

  // How to do iteration for multi-dimensional tensors? I have no idea for now.
  constexpr auto begin() {
    static_assert(sizeof...(N) == 0);
    return data_.begin();
  }

  constexpr auto end() {
    static_assert(sizeof...(N) == 0);
    return data_.end();
  }

 protected:
  // Allow accessing private data in other tensors.
  template<template<typename, size_t> typename, typename, size_t, size_t...>
  friend class TensorBase;

  S<T, storage_size> data_;
};

// Compute product of NxM matrix and M vector.
template<template<typename, size_t> typename S1,
         template<typename, size_t> typename S2,
         typename T1, typename T2,
         size_t N, size_t M>
auto MatrixProduct(const TensorBase<S1, T1, N, M>& left,
                   const TensorBase<S2, T2, M>& right) {
  TensorBase<std::array, std::remove_const_t<T1>, N> product;
  MatrixProductTo(left, right, &product);
  return product;
}

template<template<typename, size_t> typename S1,
         template<typename, size_t> typename S2,
         template<typename, size_t> typename S3,
         typename T1, typename T2, typename T3,
         size_t N, size_t M>
void MatrixProductTo(const TensorBase<S1, T1, N, M>& left,
                     const TensorBase<S2, T2, M>& right,
                     TensorBase<S3, T3, N>* out) {
  for (size_t i = 0; i < N; ++i) {
    (*out)[i] = 0;
    for (size_t j = 0; j < M; ++j) {
      (*out)[i] += left[i][j] * right[j];
    }
  }
}

// Compute dot product of 2 vectors with same length.
template<template<typename, size_t> typename S1,
         template<typename, size_t> typename S2,
         typename T1, typename T2, size_t N>
constexpr auto DotProduct(const TensorBase<S1, T1, N>& left,
                        const TensorBase<S2, T2, N>& right) {
  using T = std::remove_const_t<T1>;
  T result = T();
  for (size_t i = 0; i < N; ++i)
    result += left[i] * right[i];
  return result;
}

}  // namespace frost
