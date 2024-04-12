
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <sstream>

#include "Stream.h"
#include "hip/hip_runtime.h"
#ifndef __HIP_PLATFORM_NVCC__
#include "hip/hip_ext.h"
#endif

#define IMPLEMENTATION_STRING "HIP"

template <class T>
class HIPStream : public Stream<T>
{
#ifdef __HIP_PLATFORM_NVCC__
  #ifndef DWORDS_PER_LANE
  #define DWORDS_PER_LANE 1
  #endif
  #ifndef CHUNKS_PER_BLOCK
  #define CHUNKS_PER_BLOCK 8
  #endif
#else
  #ifndef DWORDS_PER_LANE
  #define DWORDS_PER_LANE 4
  #endif
  #ifndef CHUNKS_PER_BLOCK
  #define CHUNKS_PER_BLOCK 1
  #endif
#endif
  // make sure that either:
  //    DWORDS_PER_LANE is less than sizeof(T), in which case we default to 1 element
  //    or
  //    DWORDS_PER_LANE is divisible by sizeof(T)
  static_assert((DWORDS_PER_LANE * sizeof(unsigned int) < sizeof(T)) ||
                (DWORDS_PER_LANE * sizeof(unsigned int) % sizeof(T) == 0),
                "DWORDS_PER_LANE not divisible by sizeof(element_type)");

  static constexpr unsigned int chunks_per_block{CHUNKS_PER_BLOCK};
  // take into account the datatype size
  // that is, if we specify 4 DWORDS_PER_LANE, this is 2 FP64 elements
  // and 4 FP32 elements
  static constexpr unsigned int elements_per_lane{
    (DWORDS_PER_LANE * sizeof(unsigned int)) < sizeof(T) ? 1 : (
     DWORDS_PER_LANE * sizeof(unsigned int) / sizeof(T))};
  protected:
    // Size of arrays
    const unsigned int array_size;
    const unsigned int block_cnt;
    const bool evt_timing;
    hipEvent_t start_ev;
    hipEvent_t stop_ev;
    hipEvent_t coherent_ev;

    // Host array for partial sums for dot kernel
    T *sums;

    // Device side pointers to arrays
    T *d_a;
    T *d_b;
    T *d_c;

  public:
    HIPStream(const unsigned int, const bool, const int);
    ~HIPStream();

    virtual float read() override;
    virtual float write() override;
    virtual float copy() override;
    virtual float add() override;
    virtual float mul() override;
    virtual float triad() override;
    virtual T dot() override;

    virtual void init_arrays(T initA, T initB, T initC) override;
    virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;
};
