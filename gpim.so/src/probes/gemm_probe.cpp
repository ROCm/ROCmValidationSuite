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
#include "include/probes/gemm_probe.h"
#include "include/gpu_util.h"
#include <chrono>
#include <vector>
#include <cmath>

namespace gpim {

GEMMProbe::GEMMProbe() : matrix_size_(4096), initialized_(false) {
  handle_ = nullptr;
  d_A_ = nullptr;
  d_B_ = nullptr;
  d_C_ = nullptr;
  stream_ = nullptr;
  iterations_ = 10;  // Fewer iterations for compute-heavy probe
}

GEMMProbe::~GEMMProbe() {
  cleanup();
}

std::string GEMMProbe::name() const {
  return "gemm_throughput";
}

std::string GEMMProbe::description() const {
  return "Measures GEMM compute throughput (TFLOPS)";
}

std::string GEMMProbe::unit() const {
  return "TFLOPS";
}

int GEMMProbe::initialize(uint16_t gpu_id) {
  gpu_id_ = gpu_id;

  hipError_t err = hipSetDevice(gpu_id);
  if (err != hipSuccess) {
    return -1;
  }

  err = hipStreamCreate(&stream_);
  if (err != hipSuccess) {
    return -1;
  }

  rocblas_status status = rocblas_create_handle(&handle_);
  if (status != rocblas_status_success) {
    hipStreamDestroy(stream_);
    return -1;
  }

  rocblas_set_stream(handle_, stream_);

  allocate_matrices();

  // Get AMD SMI handle for telemetry
  if (gpu_hip_to_smi_hdl(gpu_id, &device_handle_) != 0) {
    device_handle_ = nullptr;  // Telemetry will be unavailable
  }

  initialized_ = true;
  return 0;
}

void GEMMProbe::allocate_matrices() {
  size_t size = matrix_size_ * matrix_size_ * sizeof(float);
  hipMalloc(&d_A_, size);
  hipMalloc(&d_B_, size);
  hipMalloc(&d_C_, size);

  // Initialize with random data
  std::vector<float> h_A(matrix_size_ * matrix_size_, 1.0f);
  std::vector<float> h_B(matrix_size_ * matrix_size_, 1.0f);

  hipMemcpy(d_A_, h_A.data(), size, hipMemcpyHostToDevice);
  hipMemcpy(d_B_, h_B.data(), size, hipMemcpyHostToDevice);
}

void GEMMProbe::free_matrices() {
  if (d_A_) hipFree(d_A_);
  if (d_B_) hipFree(d_B_);
  if (d_C_) hipFree(d_C_);
  d_A_ = d_B_ = d_C_ = nullptr;
}

ProbeResult GEMMProbe::execute() {
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

  const float alpha = 1.0f;
  const float beta = 0.0f;
  int N = matrix_size_;

  // FLOPs = 2 * N^3 (multiply-add for each element)
  double flops = 2.0 * static_cast<double>(N) * N * N;

  std::vector<double> times;
  times.reserve(iterations_);

  // Warmup
  for (uint32_t i = 0; i < warmup_iterations_; i++) {
    rocblas_sgemm(handle_,
                  rocblas_operation_none,
                  rocblas_operation_none,
                  N, N, N,
                  &alpha,
                  d_A_, N,
                  d_B_, N,
                  &beta,
                  d_C_, N);
  }
  hipStreamSynchronize(stream_);

  // Measurement
  for (uint32_t i = 0; i < iterations_; i++) {
    auto start = std::chrono::high_resolution_clock::now();

    rocblas_sgemm(handle_,
                  rocblas_operation_none,
                  rocblas_operation_none,
                  N, N, N,
                  &alpha,
                  d_A_, N,
                  d_B_, N,
                  &beta,
                  d_C_, N);
    hipStreamSynchronize(stream_);

    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    times.push_back(ms);
  }

  // Calculate TFLOPS
  std::vector<double> times_copy = times;
  double median_ms = compute_median(times_copy);
  double tflops = (flops / median_ms) / 1e9;  // TFLOPS

  double min_ms = *std::min_element(times.begin(), times.end());
  double max_ms = *std::max_element(times.begin(), times.end());

  double sum = 0.0;
  for (double t : times) sum += t;
  double mean_ms = sum / times.size();

  result.value = tflops;
  result.secondary_metrics["time_ms"] = median_ms;
  result.secondary_metrics["time_min_ms"] = min_ms;
  result.secondary_metrics["time_max_ms"] = max_ms;
  result.secondary_metrics["time_mean_ms"] = mean_ms;
  result.secondary_metrics["matrix_size"] = matrix_size_;
  result.passed = true;

  capture_telemetry(result);

  return result;
}

void GEMMProbe::cleanup() {
  if (initialized_) {
    free_matrices();
    if (handle_) rocblas_destroy_handle(handle_);
    if (stream_) hipStreamDestroy(stream_);
    handle_ = nullptr;
    stream_ = nullptr;
    initialized_ = false;
  }
}

}  // namespace gpim
