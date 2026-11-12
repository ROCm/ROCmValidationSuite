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
#ifndef GPIM_SO_INCLUDE_PROBES_ECC_ERROR_RATE_PROBE_H_
#define GPIM_SO_INCLUDE_PROBES_ECC_ERROR_RATE_PROBE_H_

#include "include/probes/probe_base.h"

namespace gpim {

/**
 * @class ECCErrorRateProbe
 * @brief Monitors ECC error rate
 *
 * Tests:
 * - Correctable ECC errors
 * - Uncorrectable ECC errors
 * - Error rate trends
 */
class ECCErrorRateProbe : public ProbeBase {
 public:
  ECCErrorRateProbe();
  ~ECCErrorRateProbe() override;

  std::string name() const override { return "ecc_error_rate"; }
  std::string description() const override {
    return "ECC error monitoring (correctable and uncorrectable errors)";
  }
  std::string unit() const override { return "errors/hour"; }
  MetricDirection direction() const override {
    return MetricDirection::LOWER_IS_BETTER;
  }

  int initialize(uint16_t gpu_id) override;
  ProbeResult execute() override;
  void cleanup() override;

 private:
  uint64_t initial_correctable_errors_;
  uint64_t initial_uncorrectable_errors_;
  uint64_t start_time_us_;

  /**
   * @brief Get current ECC error counts
   */
  int get_ecc_error_counts(uint64_t* correctable, uint64_t* uncorrectable);

  /**
   * @brief Calculate error rate (errors per hour)
   */
  double calculate_error_rate(uint64_t errors, uint64_t duration_us);
};

}  // namespace gpim

#endif  // GPIM_SO_INCLUDE_PROBES_ECC_ERROR_RATE_PROBE_H_
