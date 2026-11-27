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
#include "include/telemetry_sampler.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <vector>

namespace gpim {

namespace {

bool usable_u16(uint16_t v) {
  return v != 0 && v != UINT16_MAX;
}

}  // namespace

TelemetrySampler::TelemetrySampler()
    : handle_(nullptr), stop_(false) {}

TelemetrySampler::~TelemetrySampler() {
  if (thread_.joinable()) {
    stop_ = true;
    thread_.join();
  }
}

void TelemetrySampler::start(amdsmi_processor_handle handle) {
  if (thread_.joinable()) {
    stop();
  }
  handle_ = handle;
  summary_ = TelemetrySnapshot();
  stop_ = false;
  if (handle_ == nullptr) {
    return;
  }
  thread_ = std::thread(&TelemetrySampler::loop, this);
}

TelemetrySnapshot TelemetrySampler::stop() {
  stop_ = true;
  if (thread_.joinable()) {
    thread_.join();
  }
  return summary_;
}

void TelemetrySampler::loop() {
  struct Sample {
    double temp_c;
    double clock_mhz;
    double power_w;
    uint32_t util;
    uint64_t ppt;
    uint64_t thermal;
    bool have_temp;
    bool have_clock;
    bool have_power;
  };
  std::vector<Sample> samples;

  while (!stop_) {
    amdsmi_gpu_metrics_t metrics;
    memset(&metrics, 0, sizeof(metrics));
    if (amdsmi_get_gpu_metrics_info(handle_, &metrics) == AMDSMI_STATUS_SUCCESS) {
      Sample s;
      s.have_temp = usable_u16(metrics.temperature_hotspot);
      s.temp_c = s.have_temp ? metrics.temperature_hotspot : 0;
      uint16_t clk = metrics.current_gfxclks[0];
      if (!usable_u16(clk)) clk = metrics.current_gfxclk;
      s.have_clock = usable_u16(clk);
      s.clock_mhz = s.have_clock ? clk : 0;
      uint16_t pwr = metrics.current_socket_power;
      if (!usable_u16(pwr)) pwr = metrics.average_socket_power;
      s.have_power = usable_u16(pwr);
      s.power_w = s.have_power ? pwr : 0;
      s.util = metrics.average_gfx_activity;
      s.ppt = metrics.ppt_residency_acc;
      s.thermal = metrics.socket_thm_residency_acc;
      samples.push_back(s);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }

  if (samples.empty()) {
    return;
  }

  auto accumulate = [](bool present, double value, double* mn, double* mx,
                       double* sum, int* n) {
    if (!present) return;
    if (*n == 0 || value < *mn) *mn = value;
    if (*n == 0 || value > *mx) *mx = value;
    *sum += value;
    (*n)++;
  };

  double tsum = 0, csum = 0, psum = 0;
  int nt = 0, nc = 0, np = 0;
  summary_.temp_c_min = summary_.clock_mhz_min = summary_.power_w_min = 0;
  summary_.temp_c_max = summary_.clock_mhz_max = summary_.power_w_max = 0;
  for (const auto& s : samples) {
    accumulate(s.have_temp, s.temp_c, &summary_.temp_c_min, &summary_.temp_c_max,
               &tsum, &nt);
    accumulate(s.have_clock, s.clock_mhz, &summary_.clock_mhz_min,
               &summary_.clock_mhz_max, &csum, &nc);
    accumulate(s.have_power, s.power_w, &summary_.power_w_min,
               &summary_.power_w_max, &psum, &np);
  }
  if (nt) summary_.temp_c_avg = tsum / nt;
  if (nc) summary_.clock_mhz_avg = csum / nc;
  if (np) summary_.power_w_avg = psum / np;

  summary_.temp_junction_mc = static_cast<int64_t>(summary_.temp_c_avg * 1000.0);
  summary_.gpu_clock_mhz = static_cast<uint64_t>(summary_.clock_mhz_avg);
  summary_.power_usage_uw = static_cast<uint64_t>(summary_.power_w_avg * 1000000.0);
  summary_.gpu_utilization_pct = samples.back().util;
  if (samples.size() >= 2) {
    summary_.ppt_residency_delta = samples.back().ppt - samples.front().ppt;
    summary_.thermal_residency_delta =
        samples.back().thermal - samples.front().thermal;
  }
  summary_.sample_count = static_cast<int>(samples.size());
  summary_.valid = (nt + nc + np) > 0;
}

}  // namespace gpim
