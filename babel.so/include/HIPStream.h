
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

  protected:

    unsigned int dwords_per_lane;
    unsigned int chunks_per_block;
    unsigned int elements_per_lane;
    unsigned int tb_size;

    // Size of arrays
    const unsigned int array_size;
    unsigned int block_cnt;
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
    HIPStream(const unsigned int, const bool, const int,
        const unsigned int, const unsigned int, const unsigned int);
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

