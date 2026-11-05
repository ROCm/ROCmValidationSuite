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
#ifndef GPIM_SO_INCLUDE_PROBES_POWER_STABILITY_PROBE_H_
#define GPIM_SO_INCLUDE_PROBES_POWER_STABILITY_PROBE_H_

#include "include/probes/probe_base.h"

namespace gpim {

/**
 * @class PowerStabilityProbe
 * @brief Monitors power consumption stability under load
 *
 * Tests:
 * - Power draw variance during sustained load
 * - Power throttling detection
 * - Power supply stability (ripple detection)
 */
class PowerStabilityProbe : public ProbeBase {
 public:
  PowerStabilityProbe();
  ~PowerStabilityProbe() override;

  std::string name() const override { return "power_stability"; }
  std::string description() const override {
    return "Power consumption stability and throttling detection";
  }
  std::string unit() const override { return "W"; }
  MetricDirection direction() const override {
    return MetricDirection::LOWER_IS_BETTER;
  }

  int initialize(uint16_t gpu_id) override;
  ProbeResult execute() override;
  void cleanup() override;

 private:
  /**
   * @brief Sample power consumption over time
   */
  std::vector<double> sample_power_consumption(int duration_ms);

  /**
   * @brief Detect power throttling events
   */
  bool detect_power_throttling();

  /**
   * @brief Calculate power variance (stability metric)
   */
  double calculate_power_variance(const std::vector<double>& samples);
};

}  // namespace gpim

#endif  // GPIM_SO_INCLUDE_PROBES_POWER_STABILITY_PROBE_H_
