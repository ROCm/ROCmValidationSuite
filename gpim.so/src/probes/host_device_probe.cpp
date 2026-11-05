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
#include "include/probes/host_device_probe.h"
#include "include/gpu_util.h"

#include <hip/hip_runtime.h>

#include <cstdlib>
#include <cstring>
#include <vector>

namespace gpim {

HostDeviceProbe::HostDeviceProbe()
    : initialized_(false), bytes_(256ull * 1024 * 1024), host_(nullptr),
      device_(nullptr), stream_(nullptr) {
  iterations_ = 20;
  warmup_iterations_ = 5;
}

HostDeviceProbe::~HostDeviceProbe() { cleanup(); }

std::string HostDeviceProbe::name() const { return "host_device_bandwidth"; }

std::string HostDeviceProbe::description() const {
  return "Pinned host-to-device and device-to-host copy bandwidth";
}

std::string HostDeviceProbe::unit() const { return "GB/s"; }

int HostDeviceProbe::initialize(uint16_t gpu_id) {
  gpu_id_ = gpu_id;
  auto it = config_.find("bytes");
  if (it != config_.end()) bytes_ = static_cast<size_t>(std::stoull(it->second));
  if (bytes_ < 1024 * 1024) bytes_ = 1024 * 1024;

  if (hipSetDevice(gpu_id) != hipSuccess) return -1;
  if (hipStreamCreate(&stream_) != hipSuccess) return -1;

  if (posix_memalign(&host_, 4096, bytes_) != 0) {
    hipStreamDestroy(stream_);
    stream_ = nullptr;
    return -1;
  }
  std::memset(host_, 1, bytes_);
  if (hipHostRegister(host_, bytes_, hipHostRegisterPortable) != hipSuccess) {
    std::free(host_);
    host_ = nullptr;
    hipStreamDestroy(stream_);
    stream_ = nullptr;
    return -1;
  }
  if (hipMalloc(&device_, bytes_) != hipSuccess) {
    cleanup();
    return -1;
  }

  int numa = -1;
  hipDeviceGetAttribute(&numa, hipDeviceAttributeHostNumaId, gpu_id);
  // Stored later on each result; keep it in config as a side channel.
  config_["gpu_numa"] = std::to_string(numa);

  if (gpu_hip_to_smi_hdl(gpu_id, &device_handle_) != 0) device_handle_ = nullptr;
  initialized_ = true;
  return 0;
}

ProbeResult HostDeviceProbe::execute() {
  ProbeResult result;
  result.gpu_id = gpu_id_;
  result.timestamp_us = get_timestamp_us();
  result.passed = false;
  if (!initialized_) return result;
  if (hipSetDevice(gpu_id_) != hipSuccess) return result;

  hipEvent_t start, stop;
  if (hipEventCreate(&start) != hipSuccess || hipEventCreate(&stop) != hipSuccess) {
    return result;
  }

  auto time_copy = [&](hipMemcpyKind kind, std::vector<double>* out) -> bool {
    for (uint32_t i = 0; i < warmup_iterations_; i++) {
      if (hipMemcpyAsync(kind == hipMemcpyHostToDevice ? device_ : host_,
                         kind == hipMemcpyHostToDevice ? host_ : device_,
                         bytes_, kind, stream_) != hipSuccess) return false;
    }
    if (hipStreamSynchronize(stream_) != hipSuccess) return false;
    out->reserve(iterations_);
    for (uint32_t i = 0; i < iterations_; i++) {
      if (hipEventRecord(start, stream_) != hipSuccess) return false;
      void* dst = kind == hipMemcpyHostToDevice ? device_ : host_;
      void* src = kind == hipMemcpyHostToDevice ? host_ : device_;
      if (hipMemcpyAsync(dst, src, bytes_, kind, stream_) != hipSuccess) return false;
      if (hipEventRecord(stop, stream_) != hipSuccess) return false;
      if (hipEventSynchronize(stop) != hipSuccess) return false;
      float ms = 0.0f;
      if (hipEventElapsedTime(&ms, start, stop) != hipSuccess || ms <= 0.0f) return false;
      double seconds = static_cast<double>(ms) / 1000.0;
      out->push_back((static_cast<double>(bytes_) / seconds) / 1e9);
    }
    return out->size() == iterations_;
  };

  std::vector<double> h2d, d2h;
  bool ok = time_copy(hipMemcpyHostToDevice, &h2d) &&
            time_copy(hipMemcpyDeviceToHost, &d2h);
  hipEventDestroy(start);
  hipEventDestroy(stop);
  if (!ok) return result;

  std::vector<double> h2d_copy = h2d;
  std::vector<double> d2h_copy = d2h;
  add_metric(result, "host_to_device_bandwidth", compute_median(h2d_copy), "GB/s",
             MetricDirection::HIGHER_IS_BETTER);
  add_metric(result, "device_to_host_bandwidth", compute_median(d2h_copy), "GB/s",
             MetricDirection::HIGHER_IS_BETTER);
  auto it = config_.find("gpu_numa");
  if (it != config_.end()) result.secondary_metrics["gpu_numa"] = std::stod(it->second);
  result.secondary_metrics["bytes"] = static_cast<double>(bytes_);
  result.passed = true;
  capture_telemetry(result);
  return result;
}

void HostDeviceProbe::cleanup() {
  if (device_) hipFree(device_);
  if (host_) {
    hipHostUnregister(host_);
    std::free(host_);
  }
  if (stream_) hipStreamDestroy(stream_);
  device_ = nullptr;
  host_ = nullptr;
  stream_ = nullptr;
  initialized_ = false;
}

}  // namespace gpim
