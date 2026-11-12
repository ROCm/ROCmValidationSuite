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

#include "include/baseline_version_manager.h"
#include <sstream>
#include <fstream>
#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <sys/utsname.h>

namespace gpim {

std::string BaselineVersion::to_string() const {
  std::ostringstream oss;
  oss << major << "." << minor << "." << patch;
  return oss.str();
}

BaselineVersion BaselineVersion::from_string(const std::string& version_str) {
  BaselineVersion v;

  size_t pos1 = version_str.find('.');
  if (pos1 != std::string::npos) {
    v.major = std::stoi(version_str.substr(0, pos1));

    size_t pos2 = version_str.find('.', pos1 + 1);
    if (pos2 != std::string::npos) {
      v.minor = std::stoi(version_str.substr(pos1 + 1, pos2 - pos1 - 1));
      v.patch = std::stoi(version_str.substr(pos2 + 1));
    } else {
      v.minor = std::stoi(version_str.substr(pos1 + 1));
    }
  } else {
    v.major = std::stoi(version_str);
  }

  return v;
}

BaselineVersionManager::BaselineVersionManager() {
  // Current baseline format version
  current_version_.major = 1;
  current_version_.minor = 0;
  current_version_.patch = 0;
}

BaselineVersionManager::~BaselineVersionManager() {
}

std::string BaselineVersionManager::generate_baseline_filename(
    const std::string& base_path,
    const std::string& gpu_model,
    const std::string& rocm_version,
    const std::string& driver_version) const {

  std::ostringstream oss;
  oss << base_path;

  // Add path separator if not present
  if (!base_path.empty() && base_path[base_path.length() - 1] != '/') {
    oss << "/";
  }

  // Format: {GPU_MODEL}_{ROCM_VERSION}_{DRIVER_VERSION}_{DATE}.json
  oss << sanitize_for_filename(gpu_model) << "_"
      << sanitize_for_filename(rocm_version) << "_"
      << sanitize_for_filename(driver_version) << "_"
      << get_current_date() << ".json";

  return oss.str();
}

int BaselineVersionManager::parse_filename(const std::string& filename,
                                          BaselineVersion* version) const {
  if (!version) return -1;

  // Extract just the filename (remove path)
  size_t last_slash = filename.find_last_of("/\\");
  std::string base_filename = (last_slash != std::string::npos) ?
                              filename.substr(last_slash + 1) : filename;

  // Remove .json extension
  size_t dot_pos = base_filename.find_last_of('.');
  if (dot_pos != std::string::npos) {
    base_filename = base_filename.substr(0, dot_pos);
  }

  // Parse format: {GPU_MODEL}_{ROCM_VERSION}_{DRIVER_VERSION}_{DATE}
  size_t pos = 0;
  size_t next_pos;

  // GPU model
  next_pos = base_filename.find('_', pos);
  if (next_pos == std::string::npos) return -1;
  version->gpu_model = base_filename.substr(pos, next_pos - pos);
  pos = next_pos + 1;

  // ROCm version
  next_pos = base_filename.find('_', pos);
  if (next_pos == std::string::npos) return -1;
  version->rocm_version = base_filename.substr(pos, next_pos - pos);
  pos = next_pos + 1;

  // Driver version
  next_pos = base_filename.find('_', pos);
  if (next_pos == std::string::npos) return -1;
  version->driver_version = base_filename.substr(pos, next_pos - pos);
  pos = next_pos + 1;

  // Date
  version->date = base_filename.substr(pos);

  // Set default format version
  version->major = 1;
  version->minor = 0;
  version->patch = 0;

  return 0;
}

bool BaselineVersionManager::is_compatible(const BaselineVersion& v1,
                                          const BaselineVersion& v2) const {
  // Compatible if major version matches
  // Minor/patch differences are backward compatible
  if (v1.major != v2.major) {
    return false;
  }

  // Also check GPU model matches
  if (!v1.gpu_model.empty() && !v2.gpu_model.empty()) {
    if (v1.gpu_model != v2.gpu_model) {
      return false;
    }
  }

  return true;
}

BaselineVersion BaselineVersionManager::get_current_version() const {
  return current_version_;
}

void BaselineVersionManager::set_version(int major, int minor, int patch) {
  current_version_.major = major;
  current_version_.minor = minor;
  current_version_.patch = patch;
}

bool BaselineVersionManager::needs_migration(const BaselineVersion& file_version) const {
  // Need migration if file version is older than current
  if (file_version.major < current_version_.major) {
    return true;
  }

  if (file_version.major == current_version_.major &&
      file_version.minor < current_version_.minor) {
    return true;
  }

  return false;
}

std::string BaselineVersionManager::detect_gpu_model(uint16_t gpu_id) const {
  // Try to read from device name file (AMD GPU specific)
  std::ostringstream path;
  path << "/sys/class/drm/card" << gpu_id << "/device/product_name";

  std::ifstream file(path.str());
  if (file.is_open()) {
    std::string model;
    std::getline(file, model);
    file.close();

    // Trim whitespace
    model.erase(0, model.find_first_not_of(" \t\n\r"));
    model.erase(model.find_last_not_of(" \t\n\r") + 1);

    if (!model.empty()) {
      // Extract model name (e.g., "AMD Instinct MI300X" -> "MI300X")
      size_t mi_pos = model.find("MI");
      if (mi_pos != std::string::npos) {
        size_t space_after = model.find(' ', mi_pos);
        if (space_after != std::string::npos) {
          return model.substr(mi_pos, space_after - mi_pos);
        } else {
          return model.substr(mi_pos);
        }
      }
      return model;
    }
  }

  // Fallback: try reading from device ID
  path.str("");
  path << "/sys/class/drm/card" << gpu_id << "/device/device";
  file.open(path.str());
  if (file.is_open()) {
    std::string device_id;
    std::getline(file, device_id);
    file.close();

    // Map device IDs to model names (partial list)
    if (device_id.find("0x74a0") != std::string::npos) return "MI300X";
    if (device_id.find("0x74a1") != std::string::npos) return "MI300A";
    if (device_id.find("0x7408") != std::string::npos) return "MI250X";
    if (device_id.find("0x740c") != std::string::npos) return "MI250";
    if (device_id.find("0x738c") != std::string::npos) return "MI210";
    if (device_id.find("0x740f") != std::string::npos) return "MI210";
  }

  return "unknown";
}

std::string BaselineVersionManager::detect_driver_version() const {
  // Try reading from modinfo
  FILE* pipe = popen("modinfo amdgpu 2>/dev/null | grep '^version:' | awk '{print $2}'", "r");
  if (pipe) {
    char buffer[256];
    if (fgets(buffer, sizeof(buffer), pipe)) {
      pclose(pipe);
      std::string version(buffer);
      // Trim whitespace
      version.erase(0, version.find_first_not_of(" \t\n\r"));
      version.erase(version.find_last_not_of(" \t\n\r") + 1);
      if (!version.empty()) {
        return version;
      }
    }
    pclose(pipe);
  }

  // Fallback: try kernel version
  struct utsname uts;
  if (uname(&uts) == 0) {
    return std::string(uts.release);
  }

  return "unknown";
}

std::string BaselineVersionManager::detect_rocm_version() const {
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

  // Try hipconfig
  FILE* pipe = popen("hipconfig --version 2>/dev/null", "r");
  if (pipe) {
    char buffer[256];
    if (fgets(buffer, sizeof(buffer), pipe)) {
      pclose(pipe);
      std::string version(buffer);
      // Extract version number
      size_t pos = version.find_first_of("0123456789");
      if (pos != std::string::npos) {
        version = version.substr(pos);
        version.erase(version.find_last_not_of(" \t\n\r") + 1);
        return version;
      }
    }
    pclose(pipe);
  }

  return "unknown";
}

std::string BaselineVersionManager::get_current_date() const {
  time_t now = time(nullptr);
  struct tm* timeinfo = gmtime(&now);

  char buffer[9];  // YYYYMMDD + null terminator
  strftime(buffer, sizeof(buffer), "%Y%m%d", timeinfo);

  return std::string(buffer);
}

std::string BaselineVersionManager::sanitize_for_filename(const std::string& str) const {
  std::string result = str;

  // Replace invalid filename characters with underscore
  const std::string invalid_chars = " /\\:*?\"<>|";
  for (char& c : result) {
    if (invalid_chars.find(c) != std::string::npos) {
      c = '_';
    }
  }

  return result;
}

}  // namespace gpim
