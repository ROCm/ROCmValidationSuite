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
#ifndef GPIM_SO_INCLUDE_PROBES_GEMM_PROBE_H_
#define GPIM_SO_INCLUDE_PROBES_GEMM_PROBE_H_

#include "include/probes/probe_base.h"

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <string>

namespace gpim {

/**
 * @class GEMMProbe
 * @brief Sustained hipBLASLt GEMM throughput
 *
 * Default data type is bf16 with fp32 accumulation. Input matrices are random,
 * not constant, so the run draws power. One sample is a timed window
 * (config "seconds", default 2) of back-to-back gemms, timed with hipEvents.
 * Matrix size comes from config "matrix_size", or from MatrixOptimizer when
 * that value is 0.
 */
class GEMMProbe : public ProbeBase {
 public:
  GEMMProbe();
  virtual ~GEMMProbe();

  std::string name() const override;
  std::string description() const override;
  std::string unit() const override;
  MetricDirection direction() const override {
    return MetricDirection::HIGHER_IS_BETTER;
  }

  int initialize(uint16_t gpu_id) override;
  ProbeResult execute() override;
  void cleanup() override;

 private:
  bool initialized_;
  std::string dtype_;
  int matrix_size_;
  double seconds_;
  size_t elem_bytes_;
  hipDataType data_type_;
  hipStream_t stream_;
  hipblasLtHandle_t handle_;
  hipblasLtMatrixLayout_t layout_a_;
  hipblasLtMatrixLayout_t layout_b_;
  hipblasLtMatrixLayout_t layout_c_;
  hipblasLtMatrixLayout_t layout_d_;
  hipblasLtMatmulDesc_t matmul_;
  hipblasLtMatmulHeuristicResult_t heuristic_;
  void* workspace_;
  void* d_A_;
  void* d_B_;
  void* d_C_;
  void* d_D_;

  int choose_size(uint16_t gpu_id);
  bool setup_hipblaslt(int n);
  void release();
};

}  // namespace gpim

#endif  // GPIM_SO_INCLUDE_PROBES_GEMM_PROBE_H_
