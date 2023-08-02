/*
Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

#define PackedFloat_t   float4
#define WARP_SIZE       64
#define BLOCKSIZE       256
#define FLOATS_PER_PACK (sizeof(PackedFloat_t) / sizeof(float))
#define MEMSET_CHAR     75
#define MEMSET_VAL      13323083.0f

// Each subExecutor is provided with subarrays to work on
#define MAX_SRCS 16
#define MAX_DSTS 16
struct SubExecParam
{
  size_t    N;                                  // Number of floats this subExecutor works on
  int       numSrcs;                            // Number of source arrays
  int       numDsts;                            // Number of destination arrays
  float*    src[MAX_SRCS];                      // Source array pointers
  float*    dst[MAX_DSTS];                      // Destination array pointers
  long long startCycle;                         // Start timestamp for in-kernel timing (GPU-GFX executor)
  long long stopCycle;                          // Stop  timestamp for in-kernel timing (GPU-GFX executor)
};

void CpuReduceKernel(SubExecParam const& p)
{
  int const& numSrcs = p.numSrcs;
  int const& numDsts = p.numDsts;

  if (numSrcs == 0)
  {
    for (int i = 0; i < numDsts; ++i)
      memset(p.dst[i], MEMSET_CHAR, p.N * sizeof(float));
  }
  else if (numSrcs == 1)
  {
    float const* __restrict__ src = p.src[0];
    for (int i = 0; i < numDsts; ++i)
    {
      memcpy(p.dst[i], src, p.N * sizeof(float));
    }
  }
  else
  {
    for (int j = 0; j < p.N; j++)
    {
      float sum = p.src[0][j];
      for (int i = 1; i < numSrcs; i++) sum += p.src[i][j];
      for (int i = 0; i < numDsts; i++) p.dst[i][j] = sum;
    }
  }
}

std::string PrepSrcValueString()
{
  return "Element i = ((i * 517) modulo 383 + 31) * (srcBufferIdx + 1)";
}

__host__ __device__ float PrepSrcValue(int srcBufferIdx, size_t idx)
{
  return (((idx % 383) * 517) % 383 + 31) * (srcBufferIdx + 1);
}

// GPU kernel to prepare src buffer data
__global__ void
PrepSrcDataKernel(float* ptr, size_t N, int srcBufferIdx)
{
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < N;
       idx += blockDim.x * gridDim.x)
  {
    ptr[idx] = PrepSrcValue(srcBufferIdx, idx);
  }
}

// Helper function for memset
template <typename T> __device__ __forceinline__ T      MemsetVal();
template <>           __device__ __forceinline__ float  MemsetVal(){ return MEMSET_VAL; };
template <>           __device__ __forceinline__ float4 MemsetVal(){ return make_float4(MEMSET_VAL, MEMSET_VAL, MEMSET_VAL, MEMSET_VAL); }

// GPU copy kernel 0: 3 loops: unroll float 4, float4s, floats
template <int LOOP1_UNROLL>
__global__ void __launch_bounds__(BLOCKSIZE)
GpuReduceKernel(SubExecParam* params)
{
  int64_t startCycle = wall_clock64();

  // Operate on wavefront granularity
  SubExecParam& p    = params[blockIdx.x];
  int const numSrcs  = p.numSrcs;
  int const numDsts  = p.numDsts;
  int const waveId   = threadIdx.x / WARP_SIZE; // Wavefront number
  int const threadId = threadIdx.x % WARP_SIZE; // Thread index within wavefront

  // 1st loop - each wavefront operates on LOOP1_UNROLL x FLOATS_PER_PACK per thread per iteration
  // Determine the number of packed floats processed by the first loop
  size_t       Nrem        = p.N;
  size_t const loop1Npack  = (Nrem / (FLOATS_PER_PACK * LOOP1_UNROLL * WARP_SIZE)) * (LOOP1_UNROLL * WARP_SIZE);
  size_t const loop1Nelem  = loop1Npack * FLOATS_PER_PACK;
  size_t const loop1Inc    = BLOCKSIZE * LOOP1_UNROLL;
  size_t       loop1Offset = waveId * LOOP1_UNROLL * WARP_SIZE + threadId;

  while (loop1Offset < loop1Npack)
  {
    PackedFloat_t vals[LOOP1_UNROLL] = {};

    if (numSrcs == 0)
    {
      #pragma unroll
      for (int u = 0; u < LOOP1_UNROLL; ++u) vals[u] = MemsetVal<float4>();
    }
    else
    {
      for (int i = 0; i < numSrcs; ++i)
      {
        PackedFloat_t const* __restrict__ packedSrc = (PackedFloat_t const*)(p.src[i]) + loop1Offset;
        #pragma unroll
        for (int u = 0; u < LOOP1_UNROLL; ++u)
          vals[u] += *(packedSrc + u * WARP_SIZE);
      }
    }

    for (int i = 0; i < numDsts; ++i)
    {
      PackedFloat_t* __restrict__ packedDst = (PackedFloat_t*)(p.dst[i]) + loop1Offset;
      #pragma unroll
      for (int u = 0; u < LOOP1_UNROLL; ++u) *(packedDst + u * WARP_SIZE) = vals[u];
    }
    loop1Offset += loop1Inc;
  }
  Nrem -= loop1Nelem;

  if (Nrem > 0)
  {
    // 2nd loop - Each thread operates on FLOATS_PER_PACK per iteration
    // NOTE: Using int32_t due to smaller size requirements
    int32_t const loop2Npack  = Nrem / FLOATS_PER_PACK;
    int32_t const loop2Nelem  = loop2Npack * FLOATS_PER_PACK;
    int32_t const loop2Inc    = BLOCKSIZE;
    int32_t       loop2Offset = threadIdx.x;

    while (loop2Offset < loop2Npack)
    {
      PackedFloat_t val;
      if (numSrcs == 0)
      {
        val = MemsetVal<float4>();
      }
      else
      {
        val = {};
        for (int i = 0; i < numSrcs; ++i)
        {
          PackedFloat_t const* __restrict__ packedSrc = (PackedFloat_t const*)(p.src[i] + loop1Nelem) + loop2Offset;
          val += *packedSrc;
        }
      }

      for (int i = 0; i < numDsts; ++i)
      {
        PackedFloat_t* __restrict__ packedDst = (PackedFloat_t*)(p.dst[i] + loop1Nelem) + loop2Offset;
        *packedDst = val;
      }
      loop2Offset += loop2Inc;
    }
    Nrem -= loop2Nelem;

    // Deal with leftovers less than FLOATS_PER_PACK)
    if (threadIdx.x < Nrem)
    {
      int offset = loop1Nelem + loop2Nelem + threadIdx.x;
      float val = 0;
      if (numSrcs == 0)
      {
        val = MEMSET_VAL;
      }
      else
      {
        for (int i = 0; i < numSrcs; ++i)
          val += p.src[i][offset];
      }

      for (int i = 0; i < numDsts; ++i)
        p.dst[i][offset] = val;
    }
  }

  __syncthreads();
  if (threadIdx.x == 0)
  {
    p.startCycle = startCycle;
    p.stopCycle  = wall_clock64();
  }
}

template <typename FLOAT_TYPE, int UNROLL_FACTOR>
__device__ size_t GpuReduceFuncImpl2(SubExecParam const &p, size_t const offset, size_t const N)
{
  int    constexpr numFloatsPerPack = sizeof(FLOAT_TYPE) / sizeof(float); // Number of floats handled at a time per thread
  size_t constexpr loopPackInc      = BLOCKSIZE * UNROLL_FACTOR;
  size_t constexpr numPacksPerWave  = WARP_SIZE * UNROLL_FACTOR;
  int    const     waveId           = threadIdx.x / WARP_SIZE;            // Wavefront number
  int    const     threadId         = threadIdx.x % WARP_SIZE;            // Thread index within wavefront
  int    const     numSrcs          = p.numSrcs;
  int    const     numDsts          = p.numDsts;
  size_t const     numPacksDone     = (numFloatsPerPack == 1 && UNROLL_FACTOR == 1) ? N : (N / (FLOATS_PER_PACK * numPacksPerWave)) * numPacksPerWave;
  size_t const     numFloatsLeft    = N - numPacksDone * numFloatsPerPack;
  size_t           loopPackOffset   = waveId * numPacksPerWave + threadId;

  while (loopPackOffset < numPacksDone)
  {
    FLOAT_TYPE vals[UNROLL_FACTOR];

    if (numSrcs == 0)
    {
      #pragma unroll UNROLL_FACTOR
      for (int u = 0; u < UNROLL_FACTOR; ++u) vals[u] = MemsetVal<FLOAT_TYPE>();
    }
    else
    {
      FLOAT_TYPE const* __restrict__ src0Ptr = ((FLOAT_TYPE const*)(p.src[0] + offset)) + loopPackOffset;
      #pragma unroll UNROLL_FACTOR
      for (int u = 0; u < UNROLL_FACTOR; ++u)
        vals[u] = *(src0Ptr + u * WARP_SIZE);

      for (int i = 1; i < numSrcs; ++i)
      {
        FLOAT_TYPE const* __restrict__ srcPtr = ((FLOAT_TYPE const*)(p.src[i] + offset)) + loopPackOffset;

        #pragma unroll UNROLL_FACTOR
        for (int u = 0; u < UNROLL_FACTOR; ++u)
          vals[u] += *(srcPtr + u * WARP_SIZE);
      }
    }

    for (int i = 0; i < numDsts; ++i)
    {
      FLOAT_TYPE* __restrict__ dstPtr = (FLOAT_TYPE*)(p.dst[i + offset]) + loopPackOffset;
      #pragma unroll UNROLL_FACTOR
      for (int u = 0; u < UNROLL_FACTOR; ++u)
        *(dstPtr + u * WARP_SIZE) = vals[u];
    }
    loopPackOffset += loopPackInc;
  }

  return numFloatsLeft;
}

template <typename FLOAT_TYPE, int UNROLL_FACTOR>
__device__ size_t GpuReduceFuncImpl(SubExecParam const &p, size_t const offset, size_t const N)
{
  // Each thread in the block works on UNROLL_FACTOR FLOAT_TYPEs during each iteration of the loop
  int    constexpr numFloatsPerRead      = sizeof(FLOAT_TYPE) / sizeof(float);
  size_t constexpr numFloatsPerInnerLoop = BLOCKSIZE * numFloatsPerRead;
  size_t constexpr numFloatsPerOuterLoop = numFloatsPerInnerLoop * UNROLL_FACTOR;
  size_t const     numFloatsLeft         = (numFloatsPerRead == 1 && UNROLL_FACTOR == 1) ? 0 : N % numFloatsPerOuterLoop;
  size_t const     numFloatsDone         = N - numFloatsLeft;
  int    const     numSrcs               = p.numSrcs;
  int    const     numDsts               = p.numDsts;

  for (size_t idx = threadIdx.x * numFloatsPerRead; idx < numFloatsDone; idx += numFloatsPerOuterLoop)
  {
    FLOAT_TYPE tmp[UNROLL_FACTOR];

    if (numSrcs == 0)
    {
        #pragma unroll UNROLL_FACTOR
        for (int u = 0; u < UNROLL_FACTOR; ++u)
          tmp[u] = MemsetVal<FLOAT_TYPE>();
    }
    else
    {
      #pragma unroll UNROLL_FACTOR
      for (int u = 0; u < UNROLL_FACTOR; ++u)
        tmp[u] = *((FLOAT_TYPE*)(&p.src[0][offset + idx + u * numFloatsPerInnerLoop]));

      for (int i = 1; i < numSrcs; ++i)
      {
        #pragma unroll UNROLL_FACTOR
        for (int u = 0; u < UNROLL_FACTOR; ++u)
          tmp[u] += *((FLOAT_TYPE*)(&p.src[i][offset + idx + u * numFloatsPerInnerLoop]));
      }
    }

    for (int i = 0; i < numDsts; ++i)
    {
      for (int u = 0; u < UNROLL_FACTOR; ++u)
      {
        *((FLOAT_TYPE*)(&p.dst[i][offset + idx + u * numFloatsPerInnerLoop])) = tmp[u];
      }
    }
  }
  return numFloatsLeft;
}

template <typename FLOAT_TYPE>
__device__ size_t GpuReduceFunc(SubExecParam const &p, size_t const offset, size_t const N, int const unroll)
{
  switch (unroll)
  {
  case  1: return GpuReduceFuncImpl<FLOAT_TYPE,  1>(p, offset, N);
  case  2: return GpuReduceFuncImpl<FLOAT_TYPE,  2>(p, offset, N);
  case  3: return GpuReduceFuncImpl<FLOAT_TYPE,  3>(p, offset, N);
  case  4: return GpuReduceFuncImpl<FLOAT_TYPE,  4>(p, offset, N);
  case  5: return GpuReduceFuncImpl<FLOAT_TYPE,  5>(p, offset, N);
  case  6: return GpuReduceFuncImpl<FLOAT_TYPE,  6>(p, offset, N);
  case  7: return GpuReduceFuncImpl<FLOAT_TYPE,  7>(p, offset, N);
  case  8: return GpuReduceFuncImpl<FLOAT_TYPE,  8>(p, offset, N);
  case  9: return GpuReduceFuncImpl<FLOAT_TYPE,  9>(p, offset, N);
  case 10: return GpuReduceFuncImpl<FLOAT_TYPE, 10>(p, offset, N);
  case 11: return GpuReduceFuncImpl<FLOAT_TYPE, 11>(p, offset, N);
  case 12: return GpuReduceFuncImpl<FLOAT_TYPE, 12>(p, offset, N);
  case 13: return GpuReduceFuncImpl<FLOAT_TYPE, 13>(p, offset, N);
  case 14: return GpuReduceFuncImpl<FLOAT_TYPE, 14>(p, offset, N);
  case 15: return GpuReduceFuncImpl<FLOAT_TYPE, 15>(p, offset, N);
  case 16: return GpuReduceFuncImpl<FLOAT_TYPE, 16>(p, offset, N);
  default: return GpuReduceFuncImpl<FLOAT_TYPE,  1>(p, offset, N);
  }
}

// GPU copy kernel
__global__ void __launch_bounds__(BLOCKSIZE)
GpuReduceKernel2(SubExecParam* params)
{
  int64_t startCycle = wall_clock64();
  SubExecParam& p = params[blockIdx.x];

  size_t numFloatsLeft = GpuReduceFunc<float4>(p, 0, p.N, 8);
  if (numFloatsLeft)
    numFloatsLeft = GpuReduceFunc<float4>(p, p.N - numFloatsLeft, numFloatsLeft, 1);

  if (numFloatsLeft)
  GpuReduceFunc<float>(p, p.N - numFloatsLeft, numFloatsLeft, 1);

  __threadfence_system();
  if (threadIdx.x == 0)
  {
    p.startCycle = startCycle;
    p.stopCycle  = wall_clock64();
  }
}

#define NUM_GPU_KERNELS 18
typedef void (*GpuKernelFuncPtr)(SubExecParam*);

GpuKernelFuncPtr GpuKernelTable[NUM_GPU_KERNELS] =
{
  GpuReduceKernel<8>,
  GpuReduceKernel<1>,
  GpuReduceKernel<2>,
  GpuReduceKernel<3>,
  GpuReduceKernel<4>,
  GpuReduceKernel<5>,
  GpuReduceKernel<6>,
  GpuReduceKernel<7>,
  GpuReduceKernel<8>,
  GpuReduceKernel<9>,
  GpuReduceKernel<10>,
  GpuReduceKernel<11>,
  GpuReduceKernel<12>,
  GpuReduceKernel<13>,
  GpuReduceKernel<14>,
  GpuReduceKernel<15>,
  GpuReduceKernel<16>,
  GpuReduceKernel2
};

std::string GpuKernelNames[NUM_GPU_KERNELS] =
{
  "Default - 8xUnroll",
  "Unroll x1",
  "Unroll x2",
  "Unroll x3",
  "Unroll x4",
  "Unroll x5",
  "Unroll x6",
  "Unroll x7",
  "Unroll x8",
  "Unroll x9",
  "Unroll x10",
  "Unroll x11",
  "Unroll x12",
  "Unroll x13",
  "Unroll x14",
  "Unroll x15",
  "Unroll x16",
  "8xUnrollB",
};
