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
#include "include/probes/driver_overhead_probe.h"
#include <hip/hip_runtime.h>
#include <chrono>
#include <vector>
#include <algorithm>

__global__ 
void empty_kernel(){}


namespace gpim {

DriverOverheadProbe::DriverOverheadProbe() : initialized_(false) {
  iterations_ = 100;
  warmup_iterations_ = 10;
}

DriverOverheadProbe::~DriverOverheadProbe() {
  cleanup();
}

std::string DriverOverheadProbe::name() const {
  return "driver_overhead";
}

std::string DriverOverheadProbe::description() const {
  return "GPU driver and runtime overhead measurement";
}

std::string DriverOverheadProbe::unit() const {
  return "us";
}

int DriverOverheadProbe::initialize(uint16_t gpu_id) {
  gpu_id_ = gpu_id;

  hipError_t err = hipSetDevice(gpu_id);
  if (err != hipSuccess) {
    return -1;
  }

  initialized_ = true;
  return 0;
}

ProbeResult DriverOverheadProbe::execute() {
  ProbeResult result;
  result.probe_name = name();
  result.gpu_id = gpu_id_;
  result.unit = unit();
  result.direction = direction();
  result.timestamp_us = get_timestamp_us();
  result.passed = false;

  if (!initialized_) {
    return result;
  }

  hipSetDevice(gpu_id_);

  // Measure stream creation overhead
  double stream_creation_us = measure_stream_creation();

  // Measure sync overhead
  double sync_overhead_us = measure_sync_overhead();

  // Measure kernel launch overhead
  double kernel_launch_us = measure_kernel_launch_overhead();

  // Primary metric: stream creation (most indicative of driver overhead)
  result.value = stream_creation_us;

  // Secondary metrics
  result.secondary_metrics["sync_overhead_us"] = sync_overhead_us;
  result.secondary_metrics["kernel_launch_overhead_us"] = kernel_launch_us;

  result.passed = true;
  return result;
}

void DriverOverheadProbe::cleanup() {
  initialized_ = false;
}

double DriverOverheadProbe::measure_stream_creation() {
  std::vector<double> times;
  times.reserve(iterations_);

  // Warmup
  for (uint32_t i = 0; i < warmup_iterations_; i++) {
    hipStream_t stream;
    hipStreamCreate(&stream);
    hipStreamDestroy(stream);
  }

  // Measure
  for (uint32_t i = 0; i < iterations_; i++) {
    auto start = std::chrono::high_resolution_clock::now();

    hipStream_t stream;
    hipStreamCreate(&stream);
    hipStreamDestroy(stream);

    auto end = std::chrono::high_resolution_clock::now();
    double time_us = std::chrono::duration<double, std::micro>(end - start).count();
    times.push_back(time_us);
  }

  return compute_median(times);
}

double DriverOverheadProbe::measure_sync_overhead() {
  std::vector<double> times;
  times.reserve(iterations_);

  hipStream_t stream;
  hipStreamCreate(&stream);

  // Warmup
  for (uint32_t i = 0; i < warmup_iterations_; i++) {
    hipStreamSynchronize(stream);
  }

  // Measure
  for (uint32_t i = 0; i < iterations_; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    hipStreamSynchronize(stream);
    auto end = std::chrono::high_resolution_clock::now();

    double time_us = std::chrono::duration<double, std::micro>(end - start).count();
    times.push_back(time_us);
  }

  hipStreamDestroy(stream);

  return compute_median(times);
}


double DriverOverheadProbe::measure_kernel_launch_overhead() {
  std::vector<double> times;
  times.reserve(iterations_);

  // We'll measure the overhead by launching empty kernels
  // This is similar to kernel_launch_probe but focuses on driver overhead

  hipStream_t stream;
  hipStreamCreate(&stream);

  // Simple empty kernel
  //auto empty_kernel = [] __device__ () {};

  // Warmup
  for (uint32_t i = 0; i < warmup_iterations_; i++) {
    hipLaunchKernelGGL(empty_kernel, dim3(1), dim3(1), 0, stream);
    hipStreamSynchronize(stream);
  }

  // Measure
  for (uint32_t i = 0; i < iterations_; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(empty_kernel, dim3(1), dim3(1), 0, stream);
    hipStreamSynchronize(stream);
    auto end = std::chrono::high_resolution_clock::now();

    double time_us = std::chrono::duration<double, std::micro>(end - start).count();
    times.push_back(time_us);
  }

  hipStreamDestroy(stream);

  return compute_median(times);
}

} // namespace gpim
