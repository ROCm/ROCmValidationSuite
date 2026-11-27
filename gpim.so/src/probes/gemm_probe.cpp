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
#include "include/matrix_optimizer.h"

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace gpim {

namespace {

uint16_t float_to_bf16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7fffu + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

uint16_t float_to_fp16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  uint32_t sign = (bits >> 16) & 0x8000u;
  int32_t exp = static_cast<int32_t>((bits >> 23) & 0xff) - 127 + 15;
  uint32_t mant = bits & 0x7fffffu;
  if (exp <= 0) {
    if (exp < -10) return static_cast<uint16_t>(sign);
    mant |= 0x800000u;
    uint32_t shift = static_cast<uint32_t>(1 - exp);
    uint32_t half = (mant + (1u << (shift - 1)) + ((mant >> shift) & 1u)) >> shift;
    return static_cast<uint16_t>(sign | half);
  }
  if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) |
                               (mant >> 13));
}

}  // namespace

GEMMProbe::GEMMProbe()
    : initialized_(false), dtype_("bf16"), matrix_size_(0), seconds_(2.0),
      elem_bytes_(2), data_type_(HIP_R_16BF), stream_(nullptr), handle_(nullptr),
      layout_a_(nullptr), layout_b_(nullptr), layout_c_(nullptr),
      layout_d_(nullptr), matmul_(nullptr), workspace_(nullptr),
      d_A_(nullptr), d_B_(nullptr), d_C_(nullptr), d_D_(nullptr) {
  std::memset(&heuristic_, 0, sizeof(heuristic_));
}

GEMMProbe::~GEMMProbe() {
  cleanup();
}

std::string GEMMProbe::name() const { return "gemm"; }

std::string GEMMProbe::description() const {
  return "Sustained hipBLASLt GEMM throughput";
}

std::string GEMMProbe::unit() const { return "TFLOPS"; }

int GEMMProbe::choose_size(uint16_t gpu_id) {
  int n = matrix_size_;
  if (n <= 0) {
    MatrixOptimizer::GPUArchInfo arch = MatrixOptimizer::detect_architecture(gpu_id);
    MatrixOptimizer::OptimalMatrixSize opt =
        MatrixOptimizer::calculate_optimal_gemm_size(arch, dtype_);
    n = static_cast<int>(opt.M);
  }
  if (n < 256) n = 4096;

  size_t free_mem = 0, total_mem = 0;
  if (hipMemGetInfo(&free_mem, &total_mem) != hipSuccess || free_mem == 0) {
    return -1;
  }
  while (n > 256) {
    uint64_t bytes = 4ull * static_cast<uint64_t>(n) * n * elem_bytes_;
    if (bytes <= (free_mem * 7) / 10) break;
    n = (n * 9) / 10;
  }
  // hipBLASLt tile-friendly multiple.
  n = (n / 64) * 64;
  if (n < 256) return -1;
  return n;
}

bool GEMMProbe::setup_hipblaslt(int n) {
  if (hipblasLtCreate(&handle_) != HIPBLAS_STATUS_SUCCESS) return false;

  const int64_t dim = n;
  if (hipblasLtMatrixLayoutCreate(&layout_a_, data_type_, dim, dim, dim) !=
      HIPBLAS_STATUS_SUCCESS) return false;
  if (hipblasLtMatrixLayoutCreate(&layout_b_, data_type_, dim, dim, dim) !=
      HIPBLAS_STATUS_SUCCESS) return false;
  if (hipblasLtMatrixLayoutCreate(&layout_c_, data_type_, dim, dim, dim) !=
      HIPBLAS_STATUS_SUCCESS) return false;
  if (hipblasLtMatrixLayoutCreate(&layout_d_, data_type_, dim, dim, dim) !=
      HIPBLAS_STATUS_SUCCESS) return false;

  if (hipblasLtMatmulDescCreate(&matmul_, HIPBLAS_COMPUTE_32F, HIP_R_32F) !=
      HIPBLAS_STATUS_SUCCESS) return false;

  hipblasOperation_t op = HIPBLAS_OP_N;
  if (hipblasLtMatmulDescSetAttribute(matmul_, HIPBLASLT_MATMUL_DESC_TRANSA,
          &op, sizeof(op)) != HIPBLAS_STATUS_SUCCESS) return false;
  if (hipblasLtMatmulDescSetAttribute(matmul_, HIPBLASLT_MATMUL_DESC_TRANSB,
          &op, sizeof(op)) != HIPBLAS_STATUS_SUCCESS) return false;
  if (hipblasLtMatmulDescSetAttribute(matmul_,
          HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT,
          &data_type_, sizeof(data_type_)) != HIPBLAS_STATUS_SUCCESS) return false;
  if (hipblasLtMatmulDescSetAttribute(matmul_,
          HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT,
          &data_type_, sizeof(data_type_)) != HIPBLAS_STATUS_SUCCESS) return false;

  hipblasLtMatmulPreference_t pref;
  if (hipblasLtMatmulPreferenceCreate(&pref) != HIPBLAS_STATUS_SUCCESS) return false;
  size_t workspace_limit = 64ull * 1024 * 1024;
  hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                        &workspace_limit, sizeof(workspace_limit));

  int returned = 0;
  hipblasStatus_t st = hipblasLtMatmulAlgoGetHeuristic(
      handle_, matmul_, layout_a_, layout_b_, layout_c_, layout_d_,
      pref, 1, &heuristic_, &returned);
  hipblasLtMatmulPreferenceDestroy(pref);
  if (st != HIPBLAS_STATUS_SUCCESS || returned < 1) return false;

  if (heuristic_.workspaceSize > 0) {
    if (hipMalloc(&workspace_, heuristic_.workspaceSize) != hipSuccess) return false;
  }
  return true;
}

int GEMMProbe::initialize(uint16_t gpu_id) {
  gpu_id_ = gpu_id;
  auto it = config_.find("dtype");
  if (it != config_.end() && !it->second.empty()) dtype_ = it->second;
  it = config_.find("seconds");
  if (it != config_.end()) seconds_ = std::stod(it->second);
  if (seconds_ < 0.5) seconds_ = 0.5;
  if (seconds_ > 10.0) seconds_ = 10.0;
  it = config_.find("matrix_size");
  if (it != config_.end()) matrix_size_ = std::stoi(it->second);

  if (dtype_ == "fp32") {
    data_type_ = HIP_R_32F;
    elem_bytes_ = 4;
  } else if (dtype_ == "fp16") {
    data_type_ = HIP_R_16F;
    elem_bytes_ = 2;
  } else {
    dtype_ = "bf16";
    data_type_ = HIP_R_16BF;
    elem_bytes_ = 2;
  }

  if (hipSetDevice(gpu_id) != hipSuccess) return -1;
  if (hipStreamCreate(&stream_) != hipSuccess) return -1;

  int n = choose_size(gpu_id);
  if (n < 0) {
    release();
    return -1;
  }
  matrix_size_ = n;

  size_t count = static_cast<size_t>(n) * n;
  size_t bytes = count * elem_bytes_;
  if (hipMalloc(&d_A_, bytes) != hipSuccess ||
      hipMalloc(&d_B_, bytes) != hipSuccess ||
      hipMalloc(&d_C_, bytes) != hipSuccess ||
      hipMalloc(&d_D_, bytes) != hipSuccess) {
    release();
    return -1;
  }

  std::vector<uint16_t> half(count);
  std::vector<float> full;
  if (elem_bytes_ == 4) full.resize(count);
  uint32_t rng = 0xA5A5u ^ static_cast<uint32_t>(gpu_id * 17u + n);
  for (size_t i = 0; i < count; i++) {
    rng = rng * 1664525u + 1013904223u;
    float value = (static_cast<int32_t>(rng) / 2147483648.0f);
    if (elem_bytes_ == 4) {
      full[i] = value;
    } else if (dtype_ == "fp16") {
      half[i] = float_to_fp16(value);
    } else {
      half[i] = float_to_bf16(value);
    }
  }
  const void* src_a = elem_bytes_ == 4 ? static_cast<void*>(full.data())
                                      : static_cast<void*>(half.data());
  // Second matrix uses a different phase of the same generator.
  std::vector<uint16_t> half_b(count);
  std::vector<float> full_b;
  if (elem_bytes_ == 4) full_b.resize(count);
  for (size_t i = 0; i < count; i++) {
    rng = rng * 1664525u + 1013904223u;
    float value = (static_cast<int32_t>(rng) / 2147483648.0f);
    if (elem_bytes_ == 4) full_b[i] = value;
    else if (dtype_ == "fp16") half_b[i] = float_to_fp16(value);
    else half_b[i] = float_to_bf16(value);
  }
  const void* src_b = elem_bytes_ == 4 ? static_cast<void*>(full_b.data())
                                      : static_cast<void*>(half_b.data());
  if (hipMemcpy(d_A_, src_a, bytes, hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(d_B_, src_b, bytes, hipMemcpyHostToDevice) != hipSuccess) {
    release();
    return -1;
  }

  if (!setup_hipblaslt(n)) {
    release();
    return -1;
  }

  if (gpu_hip_to_smi_hdl(gpu_id, &device_handle_) != 0) {
    device_handle_ = nullptr;
  }
  initialized_ = true;
  return 0;
}

ProbeResult GEMMProbe::execute() {
  ProbeResult result;
  result.gpu_id = gpu_id_;
  result.timestamp_us = get_timestamp_us();
  result.passed = false;
  if (!initialized_) return result;
  if (hipSetDevice(gpu_id_) != hipSuccess) return result;

  const float alpha = 1.0f;
  const float beta = 0.0f;
  const int batch = 8;

  auto once = [&]() -> bool {
    return hipblasLtMatmul(handle_, matmul_, &alpha, d_A_, layout_a_,
                           d_B_, layout_b_, &beta, d_C_, layout_c_,
                           d_D_, layout_d_, &heuristic_.algo, workspace_,
                           heuristic_.workspaceSize, stream_) == HIPBLAS_STATUS_SUCCESS;
  };

  for (int i = 0; i < 3; i++) {
    if (!once()) return result;
  }
  if (hipStreamSynchronize(stream_) != hipSuccess) return result;

  hipEvent_t start, stop;
  if (hipEventCreate(&start) != hipSuccess || hipEventCreate(&stop) != hipSuccess) {
    return result;
  }

  int gemm_count = 0;
  double device_s = 0.0;
  auto wall_begin = std::chrono::steady_clock::now();
  while (std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_begin).count()
         < seconds_) {
    if (hipEventRecord(start, stream_) != hipSuccess) {
      hipEventDestroy(start);
      hipEventDestroy(stop);
      return result;
    }
    for (int i = 0; i < batch; i++) {
      if (!once()) {
        hipEventDestroy(start);
        hipEventDestroy(stop);
        return result;
      }
    }
    if (hipEventRecord(stop, stream_) != hipSuccess ||
        hipEventSynchronize(stop) != hipSuccess) {
      hipEventDestroy(start);
      hipEventDestroy(stop);
      return result;
    }
    float ms = 0.0f;
    if (hipEventElapsedTime(&ms, start, stop) != hipSuccess) {
      hipEventDestroy(start);
      hipEventDestroy(stop);
      return result;
    }
    device_s += static_cast<double>(ms) / 1000.0;
    gemm_count += batch;
  }
  hipEventDestroy(start);
  hipEventDestroy(stop);

  if (device_s <= 0.0 || gemm_count == 0) return result;

  double n = static_cast<double>(matrix_size_);
  double flops = 2.0 * n * n * n * static_cast<double>(gemm_count);
  double tflops = flops / device_s / 1e12;

  add_metric(result, "gemm_throughput", tflops, "TFLOPS",
             MetricDirection::HIGHER_IS_BETTER);
  result.secondary_metrics["matrix_size"] = matrix_size_;
  result.secondary_metrics["gemm_count"] = gemm_count;
  result.secondary_metrics["device_seconds"] = device_s;
  result.secondary_metrics["dtype_bf16"] = (dtype_ == "bf16") ? 1.0 : 0.0;
  result.passed = true;
  capture_telemetry(result);
  return result;
}

void GEMMProbe::release() {
  if (workspace_) hipFree(workspace_);
  if (d_A_) hipFree(d_A_);
  if (d_B_) hipFree(d_B_);
  if (d_C_) hipFree(d_C_);
  if (d_D_) hipFree(d_D_);
  if (matmul_) hipblasLtMatmulDescDestroy(matmul_);
  if (layout_a_) hipblasLtMatrixLayoutDestroy(layout_a_);
  if (layout_b_) hipblasLtMatrixLayoutDestroy(layout_b_);
  if (layout_c_) hipblasLtMatrixLayoutDestroy(layout_c_);
  if (layout_d_) hipblasLtMatrixLayoutDestroy(layout_d_);
  if (handle_) hipblasLtDestroy(handle_);
  if (stream_) hipStreamDestroy(stream_);
  workspace_ = d_A_ = d_B_ = d_C_ = d_D_ = nullptr;
  matmul_ = nullptr;
  layout_a_ = layout_b_ = layout_c_ = layout_d_ = nullptr;
  handle_ = nullptr;
  stream_ = nullptr;
}

void GEMMProbe::cleanup() {
  if (!initialized_ && handle_ == nullptr && stream_ == nullptr && d_A_ == nullptr) {
    return;
  }
  release();
  initialized_ = false;
}

}  // namespace gpim
