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
}

namespace gpim {

namespace {

bool hip_ok(hipError_t err, ProbeResult& result) {
  if (err == hipSuccess) return true;
  result.passed = false;
  result.secondary_metrics["hip_error"] = static_cast<double>(err);
  return false;
}

}  // namespace

KernelLaunchProbe::KernelLaunchProbe() : initialized_(false) {
  stream_ = nullptr;
  start_event_ = nullptr;
  stop_event_ = nullptr;
}

KernelLaunchProbe::~KernelLaunchProbe() {
  cleanup();
}

std::string KernelLaunchProbe::name() const {
  return "kernel_launch";
}

std::string KernelLaunchProbe::description() const {
  return "Kernel launch round-trip latency and back-to-back launch rate";
}

std::string KernelLaunchProbe::unit() const {
  return "us";
}

int KernelLaunchProbe::initialize(uint16_t gpu_id) {
  gpu_id_ = gpu_id;

  hipError_t err = hipSetDevice(gpu_id);
  if (err != hipSuccess) return -1;

  err = hipStreamCreate(&stream_);
  if (err != hipSuccess) return -1;

  err = hipEventCreate(&start_event_);
  if (err != hipSuccess) {
    hipStreamDestroy(stream_);
    stream_ = nullptr;
    return -1;
  }

  err = hipEventCreate(&stop_event_);
  if (err != hipSuccess) {
    hipEventDestroy(start_event_);
    hipStreamDestroy(stream_);
    start_event_ = nullptr;
    stream_ = nullptr;
    return -1;
  }

  if (gpu_hip_to_smi_hdl(gpu_id, &device_handle_) != 0) {
    device_handle_ = nullptr;
  }

  for (uint32_t i = 0; i < warmup_iterations_; i++) {
    hipLaunchKernelGGL(gpim_null_kernel, dim3(1), dim3(1), 0, stream_);
  }
  hipStreamSynchronize(stream_);
  hipGetLastError();

  initialized_ = true;
  return 0;
}

ProbeResult KernelLaunchProbe::execute() {
  ProbeResult result;
  result.gpu_id = gpu_id_;
  result.timestamp_us = get_timestamp_us();
  result.passed = false;

  if (!initialized_) return result;

  hipError_t err = hipSetDevice(gpu_id_);
  if (!hip_ok(err, result)) return result;

  std::vector<double> latencies;
  latencies.reserve(iterations_);

  for (uint32_t i = 0; i < iterations_; i++) {
    auto start = std::chrono::steady_clock::now();
    hipLaunchKernelGGL(gpim_null_kernel, dim3(1), dim3(1), 0, stream_);
    err = hipGetLastError();
    if (!hip_ok(err, result)) return result;
    err = hipStreamSynchronize(stream_);
    if (!hip_ok(err, result)) return result;
    auto end = std::chrono::steady_clock::now();
    latencies.push_back(
        std::chrono::duration<double, std::micro>(end - start).count());
  }

  std::vector<double> copy = latencies;
  double median_us = compute_median(copy);
  double sum = 0.0;
  for (double v : latencies) sum += v;
  double mean = sum / latencies.size();

  // Back-to-back: queue every launch, then one sync. The host interval is
  // the launch rate. hipEvents around the same burst record device time,
  // which for an empty kernel is not the launch rate.
  err = hipEventRecord(start_event_, stream_);
  if (!hip_ok(err, result)) return result;
  auto burst_start = std::chrono::steady_clock::now();
  for (uint32_t i = 0; i < iterations_; i++) {
    hipLaunchKernelGGL(gpim_null_kernel, dim3(1), dim3(1), 0, stream_);
  }
  err = hipGetLastError();
  if (!hip_ok(err, result)) return result;
  err = hipEventRecord(stop_event_, stream_);
  if (!hip_ok(err, result)) return result;
  err = hipEventSynchronize(stop_event_);
  if (!hip_ok(err, result)) return result;
  auto burst_end = std::chrono::steady_clock::now();

  float device_ms = 0.0f;
  err = hipEventElapsedTime(&device_ms, start_event_, stop_event_);
  if (!hip_ok(err, result)) return result;

  double host_s = std::chrono::duration<double>(burst_end - burst_start).count();
  if (host_s <= 0.0) {
    result.passed = false;
    return result;
  }
  double launches_per_s = static_cast<double>(iterations_) / host_s;

  add_metric(result, "kernel_launch_latency", median_us, "us",
             MetricDirection::LOWER_IS_BETTER);
  add_metric(result, "kernel_launch_throughput", launches_per_s, "launches/s",
             MetricDirection::HIGHER_IS_BETTER);
  result.secondary_metrics["latency_mean_us"] = mean;
  result.secondary_metrics["latency_stddev_us"] = compute_stddev(latencies, mean);
  result.secondary_metrics["burst_device_ms"] = device_ms;
  result.passed = true;
  capture_telemetry(result);
  return result;
}

void KernelLaunchProbe::cleanup() {
  if (!initialized_) return;
  if (stop_event_) hipEventDestroy(stop_event_);
  if (start_event_) hipEventDestroy(start_event_);
  if (stream_) hipStreamDestroy(stream_);
  stop_event_ = nullptr;
  start_event_ = nullptr;
  stream_ = nullptr;
  initialized_ = false;
}

}  // namespace gpim
