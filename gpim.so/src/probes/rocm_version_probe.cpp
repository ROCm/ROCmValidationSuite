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
#include "include/probes/rocm_version_probe.h"
#include "include/gpu_util.h"
#include <hip/hip_runtime.h>
#include <fstream>
#include <cstdlib>
#include <sstream>
#include <algorithm>

namespace gpim {

ROCmVersionProbe::ROCmVersionProbe() : initialized_(false) {
  iterations_ = 1;  // Version check only needs to run once
}

ROCmVersionProbe::~ROCmVersionProbe() {
  cleanup();
}

std::string ROCmVersionProbe::name() const {
  return "rocm_version_check";
}

std::string ROCmVersionProbe::description() const {
  return "ROCm software stack version validation";
}

std::string ROCmVersionProbe::unit() const {
  return "version";
}

int ROCmVersionProbe::initialize(uint16_t gpu_id) {
  gpu_id_ = gpu_id;

  // Get AMD SMI handle for driver version
  if (gpu_hip_to_smi_hdl(gpu_id, &device_handle_) != 0) {
    device_handle_ = nullptr;
  }

  initialized_ = true;
  return 0;
}

ProbeResult ROCmVersionProbe::execute() {
  ProbeResult result;
  result.probe_name = name();
  result.gpu_id = gpu_id_;
  result.unit = unit();
  result.direction = direction();
  result.timestamp_us = get_timestamp_us();
  result.passed = true;

  if (!initialized_) {
    result.passed = false;
    return result;
  }
  // Get ROCm version (from installation)
  std::string rocm_version = get_rocm_version();
  uint32_t rocm_ver_num = parse_version_string(rocm_version);

  // Get driver version
  std::string driver_version = get_driver_version();
  uint32_t driver_ver_num = parse_version_string(driver_version);

  // Get HIP runtime version
  uint32_t hip_version = get_hip_version();

  // Get rocBLAS version
  std::string rocblas_version = get_rocblas_version();
  uint32_t rocblas_ver_num = parse_version_string(rocblas_version);

  // Store versions as secondary metrics
  result.secondary_metrics["rocm_version"] = rocm_ver_num;  
  result.secondary_metrics["driver_version"] = driver_ver_num;
  result.secondary_metrics["hip_version"] = hip_version;
  result.secondary_metrics["rocblas_version"] = rocblas_ver_num;

  // Check for known issues
  bool rocm_ok = !is_known_buggy_version("rocm", rocm_ver_num);
  bool driver_ok = !is_known_buggy_version("driver", driver_ver_num);
  bool rocblas_ok = !is_known_buggy_version("rocblas", rocblas_ver_num);

  result.passed = (rocm_ok && driver_ok && rocblas_ok);
  result.value = result.passed ? 1.0 : 0.0;

  // Capture telemetry
  capture_telemetry(result);

  return result;
}

void ROCmVersionProbe::cleanup() {
  initialized_ = false;
}

uint32_t ROCmVersionProbe::parse_version_string(const std::string& version) {
  // Parse version string like "6.0.2" to 60002
  std::istringstream iss(version);
  uint32_t major = 0, minor = 0, patch = 0;
  char dot;

  iss >> major >> dot >> minor >> dot >> patch;

  return major * 10000 + minor * 100 + patch;
}

bool ROCmVersionProbe::is_known_buggy_version(const std::string& component,
                                              uint32_t version) {
  // Site-specific deny list supplied through configuration, e.g.
  // set_config("known_bad_rocblas", "4.1.0,4.2.0"). Empty by default.
  auto it = config_.find("known_bad_" + component);
  if (it == config_.end()) {
    return false;
  }

  std::istringstream list(it->second);
  std::string entry;
  while (std::getline(list, entry, ',')) {
    entry.erase(0, entry.find_first_not_of(" \t"));
    entry.erase(entry.find_last_not_of(" \t") + 1);
    if (!entry.empty() && parse_version_string(entry) == version) {
      return true;
    }
  }

  return false;
}

std::string ROCmVersionProbe::get_rocm_version() {
  // Try reading from /opt/rocm/.info/version file
  std::ifstream version_file("/opt/rocm/.info/version");
  if (version_file.is_open()) {
    std::string version_line;
    std::getline(version_file, version_line);
    version_file.close();

    // Trim whitespace
    version_line.erase(0, version_line.find_first_not_of(" \t\n\r"));
    version_line.erase(version_line.find_last_not_of(" \t\n\r") + 1);

    if (!version_line.empty()) {
      return version_line;
    }
  }

  // Fallback: try environment variable
  const char* rocm_ver_env = std::getenv("ROCM_VERSION");
  if (rocm_ver_env) {
    return std::string(rocm_ver_env);
  }

  // Last resort: parse from ROCM_PATH
  const char* rocm_path = std::getenv("ROCM_PATH");
  if (rocm_path) {
    std::string path(rocm_path);
    // Check if path ends with version number like /opt/rocm-7.2.0
    size_t dash_pos = path.find_last_of("-");
    if (dash_pos != std::string::npos) {
      std::string ver_candidate = path.substr(dash_pos + 1);
      // Check if it looks like a version (contains dots and digits)
      if (ver_candidate.find('.') != std::string::npos) {
        return ver_candidate;
      }
    }
  }

  return "0.0.0";
}

std::string ROCmVersionProbe::get_driver_version() {
  if (!device_handle_) {
    return "0.0.0";
  }
  amdsmi_driver_info_t driver_info;
  amdsmi_status_t status = amdsmi_get_gpu_driver_info(device_handle_, &driver_info);

  if (status != AMDSMI_STATUS_SUCCESS) {
    return "0.0.0";
  }

  return std::string(driver_info.driver_version);
}

uint32_t ROCmVersionProbe::get_hip_version() {
  int hip_version = 0;
  hipError_t err = hipRuntimeGetVersion(&hip_version);

  if (err != hipSuccess) {
    return 0;
  }

  // HIP version is encoded as: major*10000000 + minor*100000 + patch
  // Convert to our format: major*10000 + minor*100 + patch
  uint32_t major = hip_version / 10000000;
  uint32_t minor = (hip_version / 100000) % 100;
  uint32_t patch = (hip_version / 1000) % 100;

  return major * 10000 + minor * 100 + patch;
}

std::string ROCmVersionProbe::get_rocblas_version() {
  char version_str[256] = {0};
  rocblas_status status = rocblas_get_version_string(version_str, 256);

  if (status != rocblas_status_success) {
    return "0.0.0";
  }

  return std::string(version_str);
}

} // namespace gpim
