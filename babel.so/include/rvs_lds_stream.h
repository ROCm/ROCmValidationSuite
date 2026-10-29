/********************************************************************************
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * MIT LICENSE:
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to do
 * so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

// STREAM kernels that stage every operand through LDS. HBM <-> LDS traffic uses
// either the async-to/from-LDS instructions (tdm/asyncCopy.h) or the Tensor Data
// Mover (tdm/tdmCopy.h); the arithmetic only reads and writes LDS.
//
// Copy is the libraries' block-collective HBM -> LDS -> HBM round trip. The other
// kernels give every warp a private LDS window split into two halves: while one
// half is being computed on and stored, the next chunk loads into the other.
// Async loads and stores retire in order only among their own kind, so each phase
// starts with a full drain, which settles both the RAW (loaded chunk) and the WAR
// (half being reloaded) dependency.

#pragma once

#include <hip/hip_runtime.h>
#include <cfloat>
#include <cstdint>

#include "include/Stream.h"
#include "include/tdm/asyncCopy.h"
#include "include/tdm/tdmCopy.h"

#define LDS_STREAM_SUPPORTED (ASYNC_COPY_SUPPORTED && TDM_SUPPORTED)

namespace lds_stream {

enum class Lib { Async, Tdm };

// Index order matches the Stream<T> kernels and rvs_stress.cpp's subtest labels.
enum Op : int { OP_READ = 0, OP_WRITE, OP_COPY, OP_MUL, OP_ADD, OP_TRIAD, OP_DOT, OP_COUNT };

// Every chunk is a whole number of full-warp async b128 instructions (32 lanes x 16B)
// and of 256B TDM rows, so neither library peels a head or tail.
constexpr uint32_t CHUNK_GRAIN = 512;
// Each block's Copy slice starts on this boundary.
constexpr size_t COPY_SLICE_ALIGN = 4096;

template <int OP>
struct OpTraits;
template <> struct OpTraits<OP_READ>  { static constexpr uint32_t inputs = 1; static constexpr bool reduce = true;  };
template <> struct OpTraits<OP_WRITE> { static constexpr uint32_t inputs = 0; static constexpr bool reduce = false; };
template <> struct OpTraits<OP_MUL>   { static constexpr uint32_t inputs = 1; static constexpr bool reduce = false; };
template <> struct OpTraits<OP_ADD>   { static constexpr uint32_t inputs = 2; static constexpr bool reduce = false; };
template <> struct OpTraits<OP_TRIAD> { static constexpr uint32_t inputs = 2; static constexpr bool reduce = false; };
template <> struct OpTraits<OP_DOT>   { static constexpr uint32_t inputs = 2; static constexpr bool reduce = true;  };

// LDS slots per warp window: two halves of `inputs` slots (results overwrite slot 0),
// or a single slot holding Write's constant.
template <int OP>
constexpr uint32_t windowSlots() {
  return OpTraits<OP>::inputs ? 2 * OpTraits<OP>::inputs : 1;
}

// Bytes per chunk for OP, or 0 if the block's LDS cannot hold one grain per slot.
template <int OP>
__host__ __device__ constexpr uint32_t chunkBytes(uint32_t ldsBytes, uint32_t nWarps) {
  return (ldsBytes / nWarps / windowSlots<OP>()) & ~(CHUNK_GRAIN - 1);
}

template <typename T>
using Kernel = void (*)(const T*, const T*, T*, T*, size_t, uint32_t);

#if LDS_STREAM_SUPPORTED

template <Lib L, CachePolicy cp>
__device__ inline void load(const uint8_t* global, uint8_t* lds, size_t bytes) {
  if constexpr (L == Lib::Async)
    asyncLoadToLDS<SyncPolicy::Async, cp, true>(global, lds, bytes);
  else
    tdm::asyncLoadToLDS<SyncPolicy::Async, cp, true>(global, lds, bytes);
}

template <Lib L, CachePolicy cp>
__device__ inline void store(const uint8_t* lds, uint8_t* global, size_t bytes) {
  if constexpr (L == Lib::Async)
    asyncStoreFromLDS<SyncPolicy::Async, cp, true>(lds, global, bytes);
  else
    tdm::asyncStoreFromLDS<SyncPolicy::Async, cp, true>(lds, global, bytes);
}

template <Lib L>
__device__ inline void drain() {
  if constexpr (L == Lib::Async)
    asyncWait<0>();
  else
    tdm::tdmWait();
}

// The async/TDM engines read LDS independently of the wave's DS queue, so the
// wave's own LDS writes (and reads of a half about to be reloaded) must retire first.
__device__ inline void waitLds() {
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
}

template <int OP, typename T, typename V>
__device__ inline V apply(V x0, V x1) {
  const T s = static_cast<T>(startScalar);
  if constexpr (OP == OP_READ) return x0;
  if constexpr (OP == OP_MUL) return s * x0;
  if constexpr (OP == OP_ADD) return x0 + x1;
  if constexpr (OP == OP_TRIAD) return x0 + s * x1;
  if constexpr (OP == OP_DOT) return x0 * x1;
}

#endif  // LDS_STREAM_SUPPORTED

// in0/in1 are the op's inputs in STREAM order (Mul: c; Add: a, b; Triad: b, c;
// Dot: a, b; Read: a); out is the array written (Read writes c only to defeat DCE).
// Dot leaves one partial sum per block in sums[blockIdx.x].
template <Lib L, int OP, CachePolicy LD, CachePolicy ST, typename T>
__launch_bounds__(1024) __global__
void stream_kernel(const T* __restrict in0, const T* __restrict in1, T* __restrict out,
                   T* __restrict sums, size_t n, uint32_t ldsBytes) {
#if LDS_STREAM_SUPPORTED
  extern __shared__ __align__(CHUNK_GRAIN) uint8_t lds[];
  constexpr uint32_t NIN = OpTraits<OP>::inputs;
  constexpr uint32_t VEC = 16 / sizeof(T);
  using V = T __attribute__((ext_vector_type(VEC)));

  const uint32_t lane = __lane_id();
  const uint32_t warp = threadIdx.x / warpSize;
  const uint32_t nWarps = blockDim.x / warpSize;
  const size_t chunk = chunkBytes<OP>(ldsBytes, nWarps);
  uint8_t* win = lds + static_cast<size_t>(warp) * windowSlots<OP>() * chunk;

  const size_t bytes = n * sizeof(T);
  const size_t nChunks = (bytes + chunk - 1) / chunk;
  const size_t firstChunk = static_cast<size_t>(blockIdx.x) * nWarps + warp;
  const size_t chunkStride = static_cast<size_t>(gridDim.x) * nWarps;
  auto chunkLen = [&](size_t k) { return bytes - k * chunk < chunk ? bytes - k * chunk : chunk; };

  if constexpr (NIN == 0) {
    const V v = static_cast<T>(startC);
    for (uint32_t i = lane; i < chunk / sizeof(V); i += warpSize) reinterpret_cast<V*>(win)[i] = v;
    waitLds();
    uint32_t inFlight = 0;
    for (size_t k = firstChunk; k < nChunks; k += chunkStride) {
      store<L, ST>(win, reinterpret_cast<uint8_t*>(out) + k * chunk, chunkLen(k));
      if (++inFlight == 2) {
        drain<L>();
        inFlight = 0;
      }
    }
    drain<L>();
    return;
  } else {
    auto slot = [&](uint32_t half, uint32_t i) { return win + (half * NIN + i) * chunk; };
    auto issueLoads = [&](size_t k, uint32_t half) {
      load<L, LD>(reinterpret_cast<const uint8_t*>(in0) + k * chunk, slot(half, 0), chunkLen(k));
      if constexpr (NIN == 2)
        load<L, LD>(reinterpret_cast<const uint8_t*>(in1) + k * chunk, slot(half, 1), chunkLen(k));
    };

    V accV = static_cast<T>(0);
    T acc = 0;
    if (firstChunk < nChunks) issueLoads(firstChunk, 0);
    uint32_t half = 0;
    for (size_t k = firstChunk; k < nChunks; k += chunkStride, half ^= 1) {
      drain<L>();
      waitLds();
      if (k + chunkStride < nChunks) issueLoads(k + chunkStride, half ^ 1);

      const size_t len = chunkLen(k);
      V* x0 = reinterpret_cast<V*>(slot(half, 0));
      const V* x1 = reinterpret_cast<const V*>(slot(half, NIN - 1));
      const uint32_t nVec = static_cast<uint32_t>(len / sizeof(V));
      for (uint32_t i = lane; i < nVec; i += warpSize) {
        if constexpr (OpTraits<OP>::reduce) accV += apply<OP, T>(x0[i], x1[i]);
        else x0[i] = apply<OP, T>(x0[i], x1[i]);
      }
      T* s0 = reinterpret_cast<T*>(x0);
      const T* s1 = reinterpret_cast<const T*>(x1);
      for (uint32_t j = nVec * VEC + lane; j < len / sizeof(T); j += warpSize) {
        const V r = apply<OP, T>(static_cast<V>(s0[j]), static_cast<V>(s1[j]));
        if constexpr (OpTraits<OP>::reduce) acc += r[0];
        else s0[j] = r[0];
      }

      if constexpr (!OpTraits<OP>::reduce) {
        waitLds();
        store<L, ST>(slot(half, 0), reinterpret_cast<uint8_t*>(out) + k * chunk, len);
      }
    }
    drain<L>();

    if constexpr (OpTraits<OP>::reduce) {
      for (uint32_t i = 0; i < VEC; ++i) acc += accV[i];
      for (uint32_t off = warpSize / 2; off; off >>= 1) acc += __shfl_xor(acc, off);
      if constexpr (OP == OP_READ) {
        if (acc == static_cast<T>(FLT_MIN)) out[0] = acc;
      } else {
        T* partial = reinterpret_cast<T*>(lds);
        __syncthreads();
        if (lane == 0) partial[warp] = acc;
        __syncthreads();
        if (threadIdx.x == 0) {
          T total = 0;
          for (uint32_t w = 0; w < nWarps; ++w) total += partial[w];
          sums[blockIdx.x] = total;
        }
      }
    }
  }
#else
  __builtin_trap();
#endif
}

// c = a, as one HBM -> LDS -> HBM round trip per block over a contiguous slice.
template <Lib L, CachePolicy cp, typename T>
__launch_bounds__(1024) __global__
void copy_kernel(const T* __restrict in0, const T* __restrict, T* __restrict out, T* __restrict,
                 size_t n, uint32_t ldsBytes) {
#if LDS_STREAM_SUPPORTED
  extern __shared__ __align__(CHUNK_GRAIN) uint8_t lds[];
  const size_t bytes = n * sizeof(T);
  size_t per = (bytes + gridDim.x - 1) / gridDim.x;
  per = (per + COPY_SLICE_ALIGN - 1) / COPY_SLICE_ALIGN * COPY_SLICE_ALIGN;
  const size_t start = static_cast<size_t>(blockIdx.x) * per;
  if (start >= bytes) return;
  const size_t len = bytes - start < per ? bytes - start : per;
  const uint8_t* src = reinterpret_cast<const uint8_t*>(in0) + start;
  uint8_t* dst = reinterpret_cast<uint8_t*>(out) + start;
  if constexpr (L == Lib::Async)
    async::tdmCopy<cp>(dst, src, len, lds, ldsBytes);
  else
    tdm::tdmCopy<cp>(dst, src, len, lds, ldsBytes);
#else
  __builtin_trap();
#endif
}

template <Lib L, CachePolicy LD, CachePolicy ST, CachePolicy CP, typename T>
Kernel<T> kernel_for(int op) {
  switch (op) {
    case OP_READ:  return stream_kernel<L, OP_READ, LD, ST, T>;
    case OP_WRITE: return stream_kernel<L, OP_WRITE, LD, ST, T>;
    case OP_COPY:  return copy_kernel<L, CP, T>;
    case OP_MUL:   return stream_kernel<L, OP_MUL, LD, ST, T>;
    case OP_ADD:   return stream_kernel<L, OP_ADD, LD, ST, T>;
    case OP_TRIAD: return stream_kernel<L, OP_TRIAD, LD, ST, T>;
    default:       return stream_kernel<L, OP_DOT, LD, ST, T>;
  }
}

// Smallest LDS budget at which every op gets at least one chunk grain per slot.
inline uint32_t min_lds_bytes(uint32_t nWarps) {
  return nWarps * windowSlots<OP_DOT>() * CHUNK_GRAIN;
}

}  // namespace lds_stream
