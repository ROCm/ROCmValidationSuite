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
#include "include/probes/kernel_launch_probe.h"
#include "include/gpu_util.h"
#include <hip/hip_runtime.h>
#include <chrono>
#include <vector>

__global__ void gpim_null_kernel() {
  // Empty kernel - measures pure dispatch overhead
}

namespace gpim {

KernelLaunchProbe::KernelLaunchProbe() : initialized_(false) {
  stream_ = nullptr;
  start_event_ = nullptr;
  stop_event_ = nullptr;
}

KernelLaunchProbe::~KernelLaunchProbe() {
  cleanup();
}

std::string KernelLaunchProbe::name() const {
  return "kernel_launch_latency";
}

std::string KernelLaunchProbe::description() const {
  return "Measures GPU kernel dispatch overhead";
}

std::string KernelLaunchProbe::unit() const {
  return "us";
}

int KernelLaunchProbe::initialize(uint16_t gpu_id) {
  gpu_id_ = gpu_id;

  hipError_t err = hipSetDevice(gpu_id);
  if (err != hipSuccess) {
    return -1;
  }

  err = hipStreamCreate(&stream_);
  if (err != hipSuccess) {
    return -1;
  }

  err = hipEventCreate(&start_event_);
  if (err != hipSuccess) {
    hipStreamDestroy(stream_);
    return -1;
  }

  err = hipEventCreate(&stop_event_);
  if (err != hipSuccess) {
    hipEventDestroy(start_event_);
    hipStreamDestroy(stream_);
    return -1;
  }

  // Get AMD SMI handle for telemetry
  if (gpu_hip_to_smi_hdl(gpu_id, &device_handle_) != 0) {
    device_handle_ = nullptr;  // Telemetry will be unavailable
  }

  warmup();
  initialized_ = true;
  return 0;
}

void KernelLaunchProbe::warmup() {
  for (uint32_t i = 0; i < warmup_iterations_; i++) {
    hipLaunchKernelGGL(gpim_null_kernel, dim3(1), dim3(1), 0, stream_);
  }
  hipStreamSynchronize(stream_);
}

ProbeResult KernelLaunchProbe::execute() {
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

  std::vector<double> latencies;
  latencies.reserve(iterations_);

  for (uint32_t i = 0; i < iterations_; i++) {
    auto start = std::chrono::high_resolution_clock::now();

    hipLaunchKernelGGL(gpim_null_kernel, dim3(1), dim3(1), 0, stream_);
    hipStreamSynchronize(stream_);

    auto end = std::chrono::high_resolution_clock::now();

    double us = std::chrono::duration<double, std::micro>(end - start).count();
    latencies.push_back(us);
  }

  // Calculate statistics
  std::vector<double> latencies_copy = latencies;
  double median = compute_median(latencies_copy);

  latencies_copy = latencies;
  double p95 = compute_percentile(latencies_copy, 0.95);

  double min_lat = *std::min_element(latencies.begin(), latencies.end());
  double max_lat = *std::max_element(latencies.begin(), latencies.end());

  double sum = 0.0;
  for (double lat : latencies) sum += lat;
  double mean = sum / latencies.size();
  double stddev = compute_stddev(latencies, mean);

  result.value = median;
  result.secondary_metrics["mean"] = mean;
  result.secondary_metrics["p95"] = p95;
  result.secondary_metrics["min"] = min_lat;
  result.secondary_metrics["max"] = max_lat;
  result.secondary_metrics["stddev"] = stddev;
  result.passed = true;

  capture_telemetry(result);

  return result;
}

void KernelLaunchProbe::cleanup() {
  if (initialized_) {
    if (stop_event_) hipEventDestroy(stop_event_);
    if (start_event_) hipEventDestroy(start_event_);
    if (stream_) hipStreamDestroy(stream_);
    initialized_ = false;
  }
}

}  // namespace gpim
