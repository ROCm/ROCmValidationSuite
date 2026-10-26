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
#include "include/probes/membw_probe.h"
#include "include/gpu_util.h"
#include <hip/hip_runtime.h>
#include <chrono>
#include <vector>

__global__ void triad_kernel(float* a, const float* b, const float* c,
                             float scalar, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    a[i] = b[i] + scalar * c[i];
  }
}

namespace gpim {

MemBandwidthProbe::MemBandwidthProbe() : array_size_mb_(512), initialized_(false) {
  d_a_ = nullptr;
  d_b_ = nullptr;
  d_c_ = nullptr;
  stream_ = nullptr;
  iterations_ = 50;  // Inherited from ProbeBase
  warmup_iterations_ = 5;
}

MemBandwidthProbe::~MemBandwidthProbe() {
  cleanup();
}

std::string MemBandwidthProbe::name() const {
  return "memory_bandwidth";
}

std::string MemBandwidthProbe::description() const {
  return "Measures GPU memory bandwidth (STREAM-like triad)";
}

std::string MemBandwidthProbe::unit() const {
  return "GB/s";
}

int MemBandwidthProbe::initialize(uint16_t gpu_id) {
  gpu_id_ = gpu_id;

  hipError_t err = hipSetDevice(gpu_id);
  if (err != hipSuccess) {
    return -1;
  }

  err = hipStreamCreate(&stream_);
  if (err != hipSuccess) {
    return -1;
  }

  allocate_arrays();

  // Get AMD SMI handle for telemetry
  if (gpu_hip_to_smi_hdl(gpu_id, &device_handle_) != 0) {
    device_handle_ = nullptr;  // Telemetry will be unavailable
  }

  initialized_ = true;
  return 0;
}

void MemBandwidthProbe::allocate_arrays() {
  size_t n = (array_size_mb_ * 1024 * 1024) / sizeof(float);
  size_t bytes = n * sizeof(float);

  hipMalloc(&d_a_, bytes);
  hipMalloc(&d_b_, bytes);
  hipMalloc(&d_c_, bytes);

  // Initialize arrays
  std::vector<float> h_b(n, 1.0f);
  std::vector<float> h_c(n, 2.0f);

  hipMemcpy(d_b_, h_b.data(), bytes, hipMemcpyHostToDevice);
  hipMemcpy(d_c_, h_c.data(), bytes, hipMemcpyHostToDevice);
}

void MemBandwidthProbe::free_arrays() {
  if (d_a_) hipFree(d_a_);
  if (d_b_) hipFree(d_b_);
  if (d_c_) hipFree(d_c_);
  d_a_ = d_b_ = d_c_ = nullptr;
}

ProbeResult MemBandwidthProbe::execute() {
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

  size_t n = (array_size_mb_ * 1024 * 1024) / sizeof(float);
  size_t bytes = n * sizeof(float) * 3;  // Read 2 arrays, write 1

  dim3 block_size(256);
  dim3 grid_size((n + block_size.x - 1) / block_size.x);

  std::vector<double> bandwidths;
  bandwidths.reserve(iterations_);

  // Warmup
  for (uint32_t i = 0; i < warmup_iterations_; i++) {
    hipLaunchKernelGGL(triad_kernel, grid_size, block_size, 0, stream_,
                      d_a_, d_b_, d_c_, 3.0f, n);
  }
  hipStreamSynchronize(stream_);

  // Measurement
  for (uint32_t i = 0; i < iterations_; i++) {
    auto start = std::chrono::high_resolution_clock::now();

    hipLaunchKernelGGL(triad_kernel, grid_size, block_size, 0, stream_,
                      d_a_, d_b_, d_c_, 3.0f, n);
    hipStreamSynchronize(stream_);

    auto end = std::chrono::high_resolution_clock::now();

    double seconds = std::chrono::duration<double>(end - start).count();
    double gb_s = (static_cast<double>(bytes) / seconds) / 1e9;
    bandwidths.push_back(gb_s);
  }

  std::vector<double> bw_copy = bandwidths;
  double median_bw = compute_median(bw_copy);

  double min_bw = *std::min_element(bandwidths.begin(), bandwidths.end());
  double max_bw = *std::max_element(bandwidths.begin(), bandwidths.end());

  double sum = 0.0;
  for (double bw : bandwidths) sum += bw;
  double mean_bw = sum / bandwidths.size();

  result.value = median_bw;
  result.secondary_metrics["mean"] = mean_bw;
  result.secondary_metrics["min"] = min_bw;
  result.secondary_metrics["max"] = max_bw;
  result.secondary_metrics["array_size_mb"] = static_cast<double>(array_size_mb_);
  result.passed = true;

  capture_telemetry(result);

  return result;
}

void MemBandwidthProbe::cleanup() {
  if (initialized_) {
    free_arrays();
    if (stream_) hipStreamDestroy(stream_);
    stream_ = nullptr;
    initialized_ = false;
  }
}

}  // namespace gpim
