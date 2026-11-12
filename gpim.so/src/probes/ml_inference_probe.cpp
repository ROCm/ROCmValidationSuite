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

#include "include/probes/ml_inference_probe.h"
#include <vector>
#include <algorithm>
#include <chrono>

namespace gpim {

// Simple GEMM kernel for ML inference simulation
__global__ void ml_gemm_kernel(const float* A, const float* B, float* C,
                               int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
      sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}

// Batch normalization kernel
__global__ void batchnorm_kernel(float* data, const float* mean,
                                 const float* variance, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    data[idx] = (data[idx] - mean[0]) / sqrtf(variance[0] + 1e-5f);
  }
}

MLInferenceProbe::MLInferenceProbe()
    : stream_(nullptr), d_input_(nullptr), d_output_(nullptr),
      d_weights_(nullptr), input_size_(0), output_size_(0), weight_size_(0),
      batch_size_(1), channels_(3), height_(224), width_(224),
      precision_("fp32") {
}

MLInferenceProbe::~MLInferenceProbe() {
  cleanup();
}

int MLInferenceProbe::initialize(uint16_t gpu_id) {
  gpu_id_ = gpu_id;

  // Set GPU device
  hipError_t err = hipSetDevice(gpu_id);
  if (err != hipSuccess) {
    return -1;
  }

  // Parse configuration
  auto it = config_.find("batch_size");
  if (it != config_.end()) {
    batch_size_ = std::stoi(it->second);
  }

  it = config_.find("precision");
  if (it != config_.end()) {
    precision_ = it->second;
  }

  // Allocate memory (simulate ResNet-50 first conv layer)
  // Input: batch_size x 3 x 224 x 224
  input_size_ = batch_size_ * channels_ * height_ * width_ * sizeof(float);

  // Output: batch_size x 64 x 112 x 112 (after conv + pool)
  output_size_ = batch_size_ * 64 * 112 * 112 * sizeof(float);

  // Weights: 64 x 3 x 7 x 7 (first conv layer)
  weight_size_ = 64 * 3 * 7 * 7 * sizeof(float);

  err = hipMalloc(&d_input_, input_size_);
  if (err != hipSuccess) return -1;

  err = hipMalloc(&d_output_, output_size_);
  if (err != hipSuccess) return -1;

  err = hipMalloc(&d_weights_, weight_size_);
  if (err != hipSuccess) return -1;

  // Create stream
  err = hipStreamCreate(&stream_);
  if (err != hipSuccess) return -1;

  // Initialize data with random values
  std::vector<float> h_input(input_size_ / sizeof(float), 1.0f);
  std::vector<float> h_weights(weight_size_ / sizeof(float), 0.1f);

  hipMemcpy(d_input_, h_input.data(), input_size_, hipMemcpyHostToDevice);
  hipMemcpy(d_weights_, h_weights.data(), weight_size_, hipMemcpyHostToDevice);

  return 0;
}

ProbeResult MLInferenceProbe::execute() {
  ProbeResult result;
  result.probe_name = name();
  result.gpu_id = gpu_id_;
  result.unit = unit();
  result.direction = direction();
  result.timestamp_us = get_timestamp_us();

  std::vector<double> latencies;

  // Warmup
  for (uint32_t i = 0; i < warmup_iterations_; i++) {
    measure_convolution();
  }

  // Measure convolution latency
  for (uint32_t i = 0; i < iterations_; i++) {
    double latency = measure_convolution();
    latencies.push_back(latency);
  }

  // Calculate statistics
  result.value = compute_median(latencies);

  std::vector<double> latencies_copy = latencies;
  result.secondary_metrics["p95"] = compute_percentile(latencies_copy, 0.95);
  result.secondary_metrics["min"] = *std::min_element(latencies.begin(), latencies.end());
  result.secondary_metrics["max"] = *std::max_element(latencies.begin(), latencies.end());

  double sum = 0.0;
  for (double v : latencies) sum += v;
  double mean = sum / latencies.size();
  result.secondary_metrics["mean"] = mean;
  result.secondary_metrics["std_dev"] = compute_stddev(latencies, mean);

  // Measure other operations
  result.secondary_metrics["matmul_ms"] = measure_matmul();
  result.secondary_metrics["batchnorm_ms"] = measure_batchnorm();

  // Capture telemetry
  capture_telemetry(result);

  result.passed = true;
  return result;
}

void MLInferenceProbe::cleanup() {
  if (d_input_) {
    hipFree(d_input_);
    d_input_ = nullptr;
  }
  if (d_output_) {
    hipFree(d_output_);
    d_output_ = nullptr;
  }
  if (d_weights_) {
    hipFree(d_weights_);
    d_weights_ = nullptr;
  }
  if (stream_) {
    hipStreamDestroy(stream_);
    stream_ = nullptr;
  }
}

double MLInferenceProbe::measure_convolution() {
  // Simulate 2D convolution using GEMM
  // For simplicity, we'll do a matrix multiplication that approximates conv operation

  int M = batch_size_ * 112 * 112;  // Output spatial dimensions
  int N = 64;  // Output channels
  int K = 3 * 7 * 7;  // Input channels * kernel size

  dim3 blockSize(16, 16);
  dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                (M + blockSize.y - 1) / blockSize.y);

  // Timing
  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);

  hipEventRecord(start, stream_);

  ml_gemm_kernel<<<gridSize, blockSize, 0, stream_>>>(
      (float*)d_input_, (float*)d_weights_, (float*)d_output_, M, N, K);

  hipEventRecord(stop, stream_);
  hipEventSynchronize(stop);

  float milliseconds = 0;
  hipEventElapsedTime(&milliseconds, start, stop);

  hipEventDestroy(start);
  hipEventDestroy(stop);

  return static_cast<double>(milliseconds);
}

double MLInferenceProbe::measure_matmul() {
  // Simulate attention mechanism: Q * K^T
  int seq_len = 512;
  int hidden_dim = 768;

  dim3 blockSize(16, 16);
  dim3 gridSize((hidden_dim + blockSize.x - 1) / blockSize.x,
                (seq_len + blockSize.y - 1) / blockSize.y);

  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);

  hipEventRecord(start, stream_);

  ml_gemm_kernel<<<gridSize, blockSize, 0, stream_>>>(
      (float*)d_input_, (float*)d_weights_, (float*)d_output_,
      seq_len, hidden_dim, hidden_dim);

  hipEventRecord(stop, stream_);
  hipEventSynchronize(stop);

  float milliseconds = 0;
  hipEventElapsedTime(&milliseconds, start, stop);

  hipEventDestroy(start);
  hipEventDestroy(stop);

  return static_cast<double>(milliseconds);
}

double MLInferenceProbe::measure_batchnorm() {
  int N = batch_size_ * 64 * 112 * 112;

  float mean = 0.5f;
  float variance = 0.25f;
  float *d_mean, *d_variance;

  hipMalloc(&d_mean, sizeof(float));
  hipMalloc(&d_variance, sizeof(float));
  hipMemcpy(d_mean, &mean, sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_variance, &variance, sizeof(float), hipMemcpyHostToDevice);

  int blockSize = 256;
  int gridSize = (N + blockSize - 1) / blockSize;

  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);

  hipEventRecord(start, stream_);

  batchnorm_kernel<<<gridSize, blockSize, 0, stream_>>>(
      (float*)d_output_, d_mean, d_variance, N);

  hipEventRecord(stop, stream_);
  hipEventSynchronize(stop);

  float milliseconds = 0;
  hipEventElapsedTime(&milliseconds, start, stop);

  hipEventDestroy(start);
  hipEventDestroy(stop);

  hipFree(d_mean);
  hipFree(d_variance);

  return static_cast<double>(milliseconds);
}

}  // namespace gpim
