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
#ifndef GPIM_SO_INCLUDE_PROBES_PLATFORM_CHECKS_PROBE_H_
#define GPIM_SO_INCLUDE_PROBES_PLATFORM_CHECKS_PROBE_H_

#include "include/probes/probe_base.h"

namespace gpim {

/**
 * @brief Pass/fail platform snapshot
 *
 * Not a statistical baseline. Fails when the negotiated PCIe width is below
 * the device maximum or the GPU reports uncorrectable ECC errors. Power cap
 * and partition count are recorded for the baseline fingerprint work in a
 * later phase.
 */
class PlatformChecksProbe : public ProbeBase {
 public:
  PlatformChecksProbe();
  ~PlatformChecksProbe() override;

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
};

}  // namespace gpim

#endif  // GPIM_SO_INCLUDE_PROBES_PLATFORM_CHECKS_PROBE_H_
