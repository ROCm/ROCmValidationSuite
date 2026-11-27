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
#ifndef GPIM_SO_INCLUDE_TELEMETRY_SAMPLER_H_
#define GPIM_SO_INCLUDE_TELEMETRY_SAMPLER_H_

#include <atomic>
#include <thread>

#include "amd_smi/amdsmi.h"
#include "include/probes/probe_base.h"

namespace gpim {

/**
 * @brief Samples GPU metrics on a background thread while a probe runs
 *
 * Interval is 20 ms. stop() joins the thread and returns min/avg/max of
 * hotspot temperature, gfx clock, and socket power, plus the change in the
 * power and thermal throttle residency counters across the window.
 */
class TelemetrySampler {
 public:
  TelemetrySampler();
  ~TelemetrySampler();

  void start(amdsmi_processor_handle handle);
  TelemetrySnapshot stop();

 private:
  void loop();

  amdsmi_processor_handle handle_;
  std::atomic<bool> stop_;
  std::thread thread_;
  TelemetrySnapshot summary_;
};

}  // namespace gpim

#endif  // GPIM_SO_INCLUDE_TELEMETRY_SAMPLER_H_
