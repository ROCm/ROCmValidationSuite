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
#ifndef GPIM_SO_INCLUDE_PROBES_PCIE_BANDWIDTH_PROBE_H_
#define GPIM_SO_INCLUDE_PROBES_PCIE_BANDWIDTH_PROBE_H_

#include "include/probes/probe_base.h"
#include "hip/hip_runtime.h"

namespace gpim {

/**
 * @class PCIeBandwidthProbe
 * @brief Measures PCIe bandwidth in both directions
 *
 * Tests:
 * - Host-to-Device (H2D) bandwidth
 * - Device-to-Host (D2H) bandwidth
 * - Bidirectional bandwidth
 * - P2P bandwidth (if multiple GPUs available)
 */
class PCIeBandwidthProbe : public ProbeBase {
 public:
  PCIeBandwidthProbe();
  ~PCIeBandwidthProbe() override;

  std::string name() const override { return "pcie_bandwidth"; }
  std::string description() const override {
    return "PCIe bandwidth (H2D, D2H, bidirectional, P2P)";
  }
  std::string unit() const override { return "GB/s"; }
  MetricDirection direction() const override {
    return MetricDirection::HIGHER_IS_BETTER;
  }

  int initialize(uint16_t gpu_id) override;
  ProbeResult execute() override;
  void cleanup() override;

 private:
  void* h_buffer_;
  void* d_buffer_;
  size_t transfer_size_;
  hipStream_t stream_;

  /**
   * @brief Measure H2D bandwidth
   */
  double measure_h2d_bandwidth();

  /**
   * @brief Measure D2H bandwidth
   */
  double measure_d2h_bandwidth();

  /**
   * @brief Measure bidirectional bandwidth
   */
  double measure_bidirectional_bandwidth();
};

}  // namespace gpim

#endif  // GPIM_SO_INCLUDE_PROBES_PCIE_BANDWIDTH_PROBE_H_
