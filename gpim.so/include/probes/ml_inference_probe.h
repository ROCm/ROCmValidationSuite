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
#ifndef GPIM_SO_INCLUDE_PROBES_ML_INFERENCE_PROBE_H_
#define GPIM_SO_INCLUDE_PROBES_ML_INFERENCE_PROBE_H_

#include "include/probes/probe_base.h"
#include "hip/hip_runtime.h"

namespace gpim {

/**
 * @class MLInferenceProbe
 * @brief Measures ML inference latency using common operations
 *
 * Tests:
 * - 2D Convolution (typical CNN layer)
 * - Matrix multiplication (attention mechanism)
 * - Batch normalization
 * - ReLU activation
 *
 * Supports FP32, FP16, BF16, INT8 precision
 */
class MLInferenceProbe : public ProbeBase {
 public:
  MLInferenceProbe();
  ~MLInferenceProbe() override;

  std::string name() const override { return "ml_inference"; }
  std::string description() const override {
    return "ML inference latency for common operations (conv, matmul, batchnorm)";
  }
  std::string unit() const override { return "ms"; }
  MetricDirection direction() const override {
    return MetricDirection::LOWER_IS_BETTER;
  }

  int initialize(uint16_t gpu_id) override;
  ProbeResult execute() override;
  void cleanup() override;

 private:
  hipStream_t stream_;
  void* d_input_;
  void* d_output_;
  void* d_weights_;
  size_t input_size_;
  size_t output_size_;
  size_t weight_size_;

  int batch_size_;
  int channels_;
  int height_;
  int width_;
  std::string precision_;  // "fp32", "fp16", "bf16", "int8"

  /**
   * @brief Run convolution operation
   */
  double measure_convolution();

  /**
   * @brief Run matrix multiplication (simulating attention)
   */
  double measure_matmul();

  /**
   * @brief Run batch normalization
   */
  double measure_batchnorm();
};

}  // namespace gpim

#endif  // GPIM_SO_INCLUDE_PROBES_ML_INFERENCE_PROBE_H_
