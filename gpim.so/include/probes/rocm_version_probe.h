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
#ifndef GPIM_SO_INCLUDE_PROBES_ROCM_VERSION_PROBE_H_
#define GPIM_SO_INCLUDE_PROBES_ROCM_VERSION_PROBE_H_

#include "include/probes/probe_base.h"
#include <rocblas/rocblas.h>

namespace gpim {

/**
 * @class ROCmVersionProbe
 * @brief Check ROCm driver/runtime versions and detect mismatches
 *
 * This probe validates the ROCm software stack versions to detect:
 * - Known buggy versions
 * - Version mismatches between components
 * - Unexpected version regressions after updates
 */
class ROCmVersionProbe : public ProbeBase {
 public:
  ROCmVersionProbe();
  ~ROCmVersionProbe();

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
  /**
   * @brief Get ROCm version from installation
   */
  std::string get_rocm_version();
  /**
   * @brief Parse version string to comparable number
   * Example: "6.0.2" -> 60002
   */
  uint32_t parse_version_string(const std::string& version);

  /**
   * @brief Check if version is in known buggy versions list
   */
  bool is_known_buggy_version(const std::string& component,
                              uint32_t version);

  /**
   * @brief Get ROCm driver version from SMI
   */
  std::string get_driver_version();

  /**
   * @brief Get HIP runtime version
   */
  uint32_t get_hip_version();

  /**
   * @brief Get rocBLAS version
   */
  std::string get_rocblas_version();
};

} // namespace gpim

#endif  // GPIM_SO_INCLUDE_PROBES_ROCM_VERSION_PROBE_H_
