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
#ifndef GPIM_SO_INCLUDE_PROBES_HOST_DEVICE_PROBE_H_
#define GPIM_SO_INCLUDE_PROBES_HOST_DEVICE_PROBE_H_

#include "include/probes/probe_base.h"

#include <hip/hip_runtime.h>

namespace gpim {

/**
 * @brief Pinned host-to-device and device-to-host copy bandwidth
 *
 * The host buffer is allocated with malloc and registered, so it follows the
 * CPU the process is running on. A process bound to the wrong NUMA node shows
 * up here as lower bandwidth while HBM bandwidth stays put.
 */
class HostDeviceProbe : public ProbeBase {
 public:
  HostDeviceProbe();
  ~HostDeviceProbe() override;

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
  size_t bytes_;
  void* host_;
  void* device_;
  hipStream_t stream_;
};

}  // namespace gpim

#endif  // GPIM_SO_INCLUDE_PROBES_HOST_DEVICE_PROBE_H_
