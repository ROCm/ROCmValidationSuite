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
#ifndef GPIM_SO_INCLUDE_PROBES_KERNEL_LAUNCH_PROBE_H_
#define GPIM_SO_INCLUDE_PROBES_KERNEL_LAUNCH_PROBE_H_

#include "include/probes/probe_base.h"
#include <hip/hip_runtime.h>

namespace gpim {

/**
 * @class KernelLaunchProbe
 * @brief Measures GPU kernel dispatch overhead (launch latency)
 *
 * This probe launches a minimal null kernel repeatedly to measure
 * the pure dispatch overhead without any actual compute work.
 * Increased launch latency can indicate:
 * - ROCm/HIP runtime issues
 * - System resource contention
 * - Driver problems
 * - CPU scheduling issues
 *
 * Typical baseline: 5-15 microseconds
 * Warning threshold: >50% increase from baseline
 * Critical threshold: >100% increase from baseline
 */
class KernelLaunchProbe : public ProbeBase {
 public:
  KernelLaunchProbe();
  virtual ~KernelLaunchProbe();

  // ProbeBase interface implementation
  std::string name() const override;
  std::string description() const override;
  std::string unit() const override;
  MetricDirection direction() const override {
    return MetricDirection::LOWER_IS_BETTER;
  }

  int initialize(uint16_t gpu_id) override;
  ProbeResult execute() override;
  void cleanup() override;

 private:
  hipStream_t stream_;        //!< HIP stream for kernel execution
  hipEvent_t start_event_;    //!< Event for timing start
  hipEvent_t stop_event_;     //!< Event for timing stop
  bool initialized_;          //!< Initialization state

  /**
   * @brief Warm up the GPU and runtime
   */
  void warmup();
};

}  // namespace gpim

#endif  // GPIM_SO_INCLUDE_PROBES_KERNEL_LAUNCH_PROBE_H_
