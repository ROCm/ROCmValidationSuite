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
#include "include/probes/platform_checks_probe.h"
#include "include/gpu_util.h"

#include "amd_smi/amdsmi.h"

#include <cstring>

namespace gpim {

PlatformChecksProbe::PlatformChecksProbe() : initialized_(false) {}

PlatformChecksProbe::~PlatformChecksProbe() { cleanup(); }

std::string PlatformChecksProbe::name() const { return "platform_checks"; }

std::string PlatformChecksProbe::description() const {
  return "PCIe link, ECC, power cap, and partition snapshot";
}

std::string PlatformChecksProbe::unit() const { return "pass"; }

int PlatformChecksProbe::initialize(uint16_t gpu_id) {
  gpu_id_ = gpu_id;
  if (gpu_hip_to_smi_hdl(gpu_id, &device_handle_) != 0) {
    device_handle_ = nullptr;
    return -1;
  }
  initialized_ = true;
  return 0;
}

ProbeResult PlatformChecksProbe::execute() {
  ProbeResult result;
  result.probe_name = name();
  result.gpu_id = gpu_id_;
  result.unit = unit();
  result.timestamp_us = get_timestamp_us();
  result.passed = false;
  if (!initialized_ || device_handle_ == nullptr) return result;

  bool pcie_ok = false;
  amdsmi_pcie_info_t pcie;
  std::memset(&pcie, 0, sizeof(pcie));
  if (amdsmi_get_pcie_info(device_handle_, &pcie) == AMDSMI_STATUS_SUCCESS) {
    result.secondary_metrics["pcie_width"] = pcie.pcie_metric.pcie_width;
    result.secondary_metrics["pcie_width_max"] = pcie.pcie_static.max_pcie_width;
    result.secondary_metrics["pcie_speed"] = pcie.pcie_metric.pcie_speed;
    result.secondary_metrics["pcie_speed_max"] = pcie.pcie_static.max_pcie_speed;
    pcie_ok = pcie.pcie_static.max_pcie_width == 0 ||
              pcie.pcie_metric.pcie_width >= pcie.pcie_static.max_pcie_width;
  }

  bool ecc_ok = true;
  amdsmi_error_count_t ecc;
  std::memset(&ecc, 0, sizeof(ecc));
  if (amdsmi_get_gpu_ecc_count(device_handle_, AMDSMI_GPU_BLOCK_UMC, &ecc) ==
      AMDSMI_STATUS_SUCCESS) {
    result.secondary_metrics["ecc_correctable"] =
        static_cast<double>(ecc.correctable_count);
    result.secondary_metrics["ecc_uncorrectable"] =
        static_cast<double>(ecc.uncorrectable_count);
    ecc_ok = ecc.uncorrectable_count == 0;
  }

  amdsmi_power_cap_info_t cap;
  std::memset(&cap, 0, sizeof(cap));
  if (amdsmi_get_power_cap_info(device_handle_, 0, &cap) == AMDSMI_STATUS_SUCCESS) {
    result.secondary_metrics["power_cap_uw"] = static_cast<double>(cap.power_cap);
  }

  amdsmi_gpu_metrics_t metrics;
  std::memset(&metrics, 0, sizeof(metrics));
  if (amdsmi_get_gpu_metrics_info(device_handle_, &metrics) == AMDSMI_STATUS_SUCCESS) {
    result.secondary_metrics["num_partition"] = metrics.num_partition;
  }

  result.passed = pcie_ok && ecc_ok;
  add_metric(result, "platform_ok", result.passed ? 1.0 : 0.0, "pass",
             MetricDirection::HIGHER_IS_BETTER, false);
  capture_telemetry(result);
  return result;
}

void PlatformChecksProbe::cleanup() { initialized_ = false; }

}  // namespace gpim
