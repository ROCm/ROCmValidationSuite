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
#include "include/baseline_repository.h"
#include "amd_smi/amdsmi.h"
#include "include/rsmi_util.h"
#include <hip/hip_runtime.h>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <cstring>

#ifdef _WIN32
#include <direct.h>
#define mkdir(path, mode) _mkdir(path)
#endif

namespace gpim {

BaselineRepository::BaselineRepository() {
  local_cache_dir_ = "/var/lib/rvs/baseline_cache";
  shipped_dir_ = "/usr/share/rvs/reference_baselines";
}

BaselineRepository::~BaselineRepository() {
}

BaselineRepository::BaselineStatus BaselineRepository::check_availability(
    const BaselineQuery& query) {

  std::string path_out;

  // Priority 1: Check local cache
  if (exists_in_cache(query, path_out)) {
    return BaselineStatus::LOCAL_AVAILABLE;
  }

  // Priority 2: Check shipped baselines
  if (exists_in_shipped(query, path_out)) {
    return BaselineStatus::LOCAL_AVAILABLE;
  }

  // Priority 3: Check remote repository
  std::string url_out;
  if (exists_in_remote(query, url_out)) {
    return BaselineStatus::REMOTE_AVAILABLE;
  }

  return BaselineStatus::NOT_AVAILABLE;
}

int BaselineRepository::fetch_baseline(const BaselineQuery& query,
                                       const std::string& output_path) {
  std::string source_path;

  // Priority 1: Local cache
  if (exists_in_cache(query, source_path)) {
    std::ifstream src(source_path, std::ios::binary);
    std::ofstream dst(output_path, std::ios::binary);
    dst << src.rdbuf();
    return 0;
  }

  // Priority 2: Shipped with RVS
  if (exists_in_shipped(query, source_path)) {
    std::ifstream src(source_path, std::ios::binary);
    std::ofstream dst(output_path, std::ios::binary);
    dst << src.rdbuf();

    // Copy to cache for future use
    std::string cache_filename = query.gpu_model + "_" + query.rocm_version + ".json";
    std::string cache_path = local_cache_dir_ + "/" + cache_filename;

    src.clear();
    src.seekg(0);
    std::ofstream cache(cache_path, std::ios::binary);
    cache << src.rdbuf();

    return 0;
  }

  // Priority 3: Download from remote
  std::string url;
  if (exists_in_remote(query, url)) {
    int rc = download_from_remote(url, output_path);
    if (rc == 0) {
      // Save to cache
      std::string cache_filename = query.gpu_model + "_" + query.rocm_version + ".json";
      std::string cache_path = local_cache_dir_ + "/" + cache_filename;

      std::ifstream src(output_path, std::ios::binary);
      std::ofstream dst(cache_path, std::ios::binary);
      dst << src.rdbuf();
    }
    return rc;
  }

  return -1;  // Not found
}

bool BaselineRepository::exists_in_cache(const BaselineQuery& query,
                                        std::string& path_out) {
  std::string filename = query.gpu_model + "_" + query.rocm_version + ".json";
  path_out = local_cache_dir_ + "/" + filename;

  struct stat buffer;
  return (stat(path_out.c_str(), &buffer) == 0);
}

bool BaselineRepository::exists_in_shipped(const BaselineQuery& query,
                                          std::string& path_out) {
  std::string filename = query.gpu_model + "_" + query.rocm_version + ".json";
  path_out = shipped_dir_ + "/" + filename;

  struct stat buffer;
  return (stat(path_out.c_str(), &buffer) == 0);
}

bool BaselineRepository::exists_in_remote(const BaselineQuery& query,
                                         std::string& url_out) {
  if (remote_repo_url_.empty()) {
    return false;
  }
  url_out = remote_repo_url_ + "/" + query.rocm_version + "/" +
            query.gpu_model + ".json";

  // Remote lookup is not implemented.
  return false;
}

int BaselineRepository::download_from_remote(const std::string& url,
                                            const std::string& output_path) {
  // Remote download is not implemented.
  return -1;
}

bool BaselineRepository::validate_baseline_file(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) return false;

  // Simple validation: check for required JSON fields
  std::string content((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());

  return (content.find("gpu_model") != std::string::npos &&
          content.find("baselines") != std::string::npos);
}

std::string BaselineRepository::detect_gpu_model(uint16_t gpu_id) {
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, gpu_id) != hipSuccess) {
    return "UNKNOWN";
  }

  std::string name(prop.name);

  // Parse GPU model from name
  if (name.find("MI355X") != std::string::npos) {
    return "MI355X";
  } else if (name.find("MI355") != std::string::npos) {
    return "MI355";
  } else if (name.find("MI350X") != std::string::npos) {
    return "MI350X";
  } else if (name.find("MI350") != std::string::npos) {
    return "MI350";
  } else if (name.find("MI300X") != std::string::npos) {
    return "MI300X";
  } else if (name.find("MI300A") != std::string::npos) {
    return "MI300A";
  } else if (name.find("MI250X") != std::string::npos) {
    return "MI250X";
  } else if (name.find("MI250") != std::string::npos) {
    return "MI250";
  } else if (name.find("MI210") != std::string::npos) {
    return "MI210";
  } else if (name.find("MI100") != std::string::npos) {
    return "MI100";
  }
  return "UNKNOWN";
}

std::string BaselineRepository::get_rocm_version() {
  std::map<uint64_t, amdsmi_processor_handle> smi_map;
  smi_map = rvs::get_smi_pci_map();
  if (smi_map.empty())
          return "0.0.0";
  amdsmi_driver_info_t driver_info;
  amdsmi_status_t status = amdsmi_get_gpu_driver_info(smi_map.begin()->second, &driver_info);

  if (status != AMDSMI_STATUS_SUCCESS) {
    return "0.0.0";
  }

  // Parse version from driver string
  // Example: "6.0.60000-1" -> "6.0.2"
  std::string ver_str(driver_info.driver_version);

  // Simple parsing - just take first two components
  size_t first_dot = ver_str.find('.');
  if (first_dot == std::string::npos) return "0.0.0";

  size_t second_dot = ver_str.find('.', first_dot + 1);
  if (second_dot == std::string::npos) return "0.0.0";

  std::string major = ver_str.substr(0, first_dot);
  std::string minor = ver_str.substr(first_dot + 1, second_dot - first_dot - 1);

  // Extract patch (simplified)
  return major + "." + minor + ".0";
}

int BaselineRepository::contribute_baseline(
    const std::string& baseline_path,
    const std::map<std::string, std::string>& metadata) {

  // Validate baseline before contributing
  if (!validate_baseline_file(baseline_path)) {
    return -1;
  }

  // Contribution upload is not implemented.
  return -1;
}

std::string BaselineRepository::create_contribution_package(
    const std::string& baseline_path,
    const std::map<std::string, std::string>& metadata) {

  // Contribution packaging is not implemented.
  return "";
}

std::vector<std::string> BaselineRepository::get_baseline_search_paths(
    const BaselineQuery& query) {

  std::vector<std::string> paths;

  std::string filename = query.gpu_model + "_" + query.rocm_version + ".json";

  // Priority order
  paths.push_back(local_cache_dir_ + "/" + filename);
  paths.push_back(shipped_dir_ + "/" + filename);

  return paths;
}

} // namespace gpim
