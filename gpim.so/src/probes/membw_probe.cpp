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

#include <vector>

__global__ void triad_kernel(float* a, const float* b, const float* c,
                             float scalar, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    a[i] = b[i] + scalar * c[i];
  }
}

namespace gpim {

MemBandwidthProbe::MemBandwidthProbe() : array_size_mb_(0), initialized_(false) {
  d_a_ = nullptr;
  d_b_ = nullptr;
  d_c_ = nullptr;
  stream_ = nullptr;
  iterations_ = 20;
  warmup_iterations_ = 5;
}

MemBandwidthProbe::~MemBandwidthProbe() {
  cleanup();
}

std::string MemBandwidthProbe::name() const { return "memory_bandwidth"; }

std::string MemBandwidthProbe::description() const {
  return "STREAM triad GPU memory bandwidth, timed with hipEvents";
}

std::string MemBandwidthProbe::unit() const { return "GB/s"; }

int MemBandwidthProbe::initialize(uint16_t gpu_id) {
  gpu_id_ = gpu_id;
  if (hipSetDevice(gpu_id) != hipSuccess) return -1;
  if (hipStreamCreate(&stream_) != hipSuccess) return -1;

  auto it = config_.find("array_size_mb");
  if (it != config_.end()) array_size_mb_ = static_cast<size_t>(std::stoul(it->second));

  if (array_size_mb_ == 0) {
    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, gpu_id) != hipSuccess) return -1;
    size_t l2 = prop.l2CacheSize;
    size_t target = 512ull * 1024 * 1024;
    if (l2 > 0 && 8ull * l2 > target) target = 8ull * l2;
    size_t free_mem = 0, total_mem = 0;
    if (hipMemGetInfo(&free_mem, &total_mem) != hipSuccess) return -1;
    if (target > free_mem / 8) target = free_mem / 8;
    array_size_mb_ = target / (1024 * 1024);
    if (array_size_mb_ < 64) array_size_mb_ = 64;
  }

  allocate_arrays();
  if (d_a_ == nullptr || d_b_ == nullptr || d_c_ == nullptr) {
    free_arrays();
    hipStreamDestroy(stream_);
    stream_ = nullptr;
    return -1;
  }

  if (gpu_hip_to_smi_hdl(gpu_id, &device_handle_) != 0) {
    device_handle_ = nullptr;
  }
  initialized_ = true;
  return 0;
}

void MemBandwidthProbe::allocate_arrays() {
  size_t n = (array_size_mb_ * 1024 * 1024) / sizeof(float);
  size_t bytes = n * sizeof(float);
  if (hipMalloc(&d_a_, bytes) != hipSuccess) d_a_ = nullptr;
  if (hipMalloc(&d_b_, bytes) != hipSuccess) d_b_ = nullptr;
  if (hipMalloc(&d_c_, bytes) != hipSuccess) d_c_ = nullptr;
  if (!d_a_ || !d_b_ || !d_c_) return;

  std::vector<float> h_b(n, 1.0f);
  std::vector<float> h_c(n, 2.0f);
  if (hipMemcpy(d_b_, h_b.data(), bytes, hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(d_c_, h_c.data(), bytes, hipMemcpyHostToDevice) != hipSuccess) {
    free_arrays();
  }
}

void MemBandwidthProbe::free_arrays() {
  if (d_a_) hipFree(d_a_);
  if (d_b_) hipFree(d_b_);
  if (d_c_) hipFree(d_c_);
  d_a_ = d_b_ = d_c_ = nullptr;
}

ProbeResult MemBandwidthProbe::execute() {
  ProbeResult result;
  result.gpu_id = gpu_id_;
  result.timestamp_us = get_timestamp_us();
  result.passed = false;
  if (!initialized_) return result;
  if (hipSetDevice(gpu_id_) != hipSuccess) return result;

  size_t n = (array_size_mb_ * 1024 * 1024) / sizeof(float);
  size_t bytes = n * sizeof(float) * 3;
  dim3 block_size(256);
  dim3 grid_size((n + block_size.x - 1) / block_size.x);

  hipEvent_t start, stop;
  if (hipEventCreate(&start) != hipSuccess || hipEventCreate(&stop) != hipSuccess) {
    return result;
  }

  for (uint32_t i = 0; i < warmup_iterations_; i++) {
    hipLaunchKernelGGL(triad_kernel, grid_size, block_size, 0, stream_,
                      d_a_, d_b_, d_c_, 3.0f, n);
  }
  if (hipStreamSynchronize(stream_) != hipSuccess) {
    hipEventDestroy(start);
    hipEventDestroy(stop);
    return result;
  }

  std::vector<double> bandwidths;
  bandwidths.reserve(iterations_);
  for (uint32_t i = 0; i < iterations_; i++) {
    if (hipEventRecord(start, stream_) != hipSuccess) break;
    hipLaunchKernelGGL(triad_kernel, grid_size, block_size, 0, stream_,
                      d_a_, d_b_, d_c_, 3.0f, n);
    if (hipEventRecord(stop, stream_) != hipSuccess ||
        hipEventSynchronize(stop) != hipSuccess) break;
    float ms = 0.0f;
    if (hipEventElapsedTime(&ms, start, stop) != hipSuccess || ms <= 0.0f) break;
    double seconds = static_cast<double>(ms) / 1000.0;
    bandwidths.push_back((static_cast<double>(bytes) / seconds) / 1e9);
  }
  hipEventDestroy(start);
  hipEventDestroy(stop);

  if (bandwidths.size() != iterations_) return result;

  std::vector<double> copy = bandwidths;
  double median_bw = compute_median(copy);
  add_metric(result, "memory_bandwidth", median_bw, "GB/s",
             MetricDirection::HIGHER_IS_BETTER);
  result.secondary_metrics["array_size_mb"] = static_cast<double>(array_size_mb_);
  result.passed = true;
  capture_telemetry(result);
  return result;
}

void MemBandwidthProbe::cleanup() {
  if (!initialized_) return;
  free_arrays();
  if (stream_) hipStreamDestroy(stream_);
  stream_ = nullptr;
  initialized_ = false;
}

}  // namespace gpim
