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
#include "include/probes/probe_base.h"
#include <chrono>
#include <algorithm>
#include <cmath>
#include <sys/time.h>
#include "amd_smi/amdsmi.h"

namespace gpim {

uint64_t ProbeBase::get_timestamp_us() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
}

void ProbeBase::capture_telemetry(ProbeResult& result) {
  result.telemetry.valid = false;

  if (device_handle_ == nullptr) {
    return;
  }

  amdsmi_status_t status;

  // Temperature (junction)
  int64_t temp_mc = 0;
  status = amdsmi_get_temp_metric(device_handle_,
                                   AMDSMI_TEMPERATURE_TYPE_JUNCTION,
                                   AMDSMI_TEMP_CURRENT,
                                   &temp_mc);
  if (status == AMDSMI_STATUS_SUCCESS) {
    result.telemetry.temp_junction_mc = temp_mc;
  }

  // GPU clock
  amdsmi_frequencies_t frequencies;
  status = amdsmi_get_clk_freq(device_handle_, AMDSMI_CLK_TYPE_SYS, &frequencies);
  if (status == AMDSMI_STATUS_SUCCESS && frequencies.current < frequencies.num_supported) {
    result.telemetry.gpu_clock_mhz = frequencies.frequency[frequencies.current] / 1000000;
  }

  // Power usage
  amdsmi_power_info_t power_info;
  status = amdsmi_get_power_info(device_handle_, &power_info);
  if (status == AMDSMI_STATUS_SUCCESS) {
    result.telemetry.power_usage_uw = power_info.current_socket_power;
  }

  // GPU utilization
  amdsmi_engine_usage_t usage;
  status = amdsmi_get_gpu_activity(device_handle_, &usage);
  if (status == AMDSMI_STATUS_SUCCESS) {
    result.telemetry.gpu_utilization_pct = usage.gfx_activity;
  }

  result.telemetry.valid = true;
}

double ProbeBase::compute_median(std::vector<double>& values) {
  if (values.empty()) return 0.0;

  size_t n = values.size();
  std::nth_element(values.begin(), values.begin() + n/2, values.end());

  if (n % 2 == 0) {
    double median1 = values[n/2];
    std::nth_element(values.begin(), values.begin() + n/2 - 1, values.end());
    double median2 = values[n/2 - 1];
    return (median1 + median2) / 2.0;
  } else {
    return values[n/2];
  }
}

double ProbeBase::compute_percentile(std::vector<double>& values, double percentile) {
  if (values.empty()) return 0.0;
  if (percentile < 0.0) percentile = 0.0;
  if (percentile > 1.0) percentile = 1.0;

  size_t n = values.size();
  size_t idx = static_cast<size_t>(percentile * (n - 1));

  std::nth_element(values.begin(), values.begin() + idx, values.end());
  return values[idx];
}

double ProbeBase::compute_stddev(const std::vector<double>& values, double mean) {
  if (values.size() < 2) return 0.0;

  double sum_sq_diff = 0.0;
  for (double v : values) {
    double diff = v - mean;
    sum_sq_diff += diff * diff;
  }

  return std::sqrt(sum_sq_diff / (values.size() - 1));
}

}  // namespace gpim
