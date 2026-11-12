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
 
#include "include/baseline_manager.h"
#include <fstream>
#include <sstream>
#include <ctime>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <unistd.h>
#include <cstdlib>

namespace gpim {

BaselineManager::BaselineManager(const std::string& baseline_file)
    : baseline_file_(baseline_file) {
}

BaselineManager::~BaselineManager() {
}

int BaselineManager::load() {
  std::ifstream file(baseline_file_);
  if (!file.is_open()) {
    return 1;  // File doesn't exist (not an error for first run)
  }

  // Simple JSON parsing (very basic, assuming our own format)
  std::string line;
  bool in_baselines_array = false;
  Baseline current_baseline;
  bool parsing_baseline = false;

  while (std::getline(file, line)) {
    // Skip empty and comment lines
    if (line.empty() || line.find("//") == 0) continue;

    // Check for baselines array
    if (line.find("\"baselines\"") != std::string::npos) {
      in_baselines_array = true;
      continue;
    }

    if (!in_baselines_array) continue;

    // Parse baseline object
    if (line.find("{") != std::string::npos && parsing_baseline == false) {
      parsing_baseline = true;
      current_baseline = Baseline();
      continue;
    }

    if (parsing_baseline) {
      // Extract key-value pairs
      if (line.find("\"probe_name\"") != std::string::npos) {
        size_t start = line.find(":") + 1;
        size_t first_quote = line.find("\"", start);
        size_t second_quote = line.find("\"", first_quote + 1);
        current_baseline.probe_name = line.substr(first_quote + 1,
                                                   second_quote - first_quote - 1);
      }
      else if (line.find("\"gpu_id\"") != std::string::npos) {
        size_t start = line.find(":") + 1;
        current_baseline.gpu_id = std::stoi(line.substr(start));
      }
      else if (line.find("\"value\"") != std::string::npos) {
        size_t start = line.find(":") + 1;
        current_baseline.value = std::stod(line.substr(start));
      }
      else if (line.find("\"std_dev\"") != std::string::npos) {
        size_t start = line.find(":") + 1;
        current_baseline.std_dev = std::stod(line.substr(start));
      }
      else if (line.find("\"timestamp\"") != std::string::npos) {
        size_t start = line.find(":") + 1;
        current_baseline.timestamp = std::stoull(line.substr(start));
      }
      else if (line.find("\"sample_count\"") != std::string::npos) {
        size_t start = line.find(":") + 1;
        current_baseline.sample_count = std::stoi(line.substr(start));
      }

      if (line.find("}") != std::string::npos) {
        // Store baseline
        std::string key = make_key(current_baseline.probe_name,
                                   current_baseline.gpu_id);
        baselines_[key] = current_baseline;
        parsing_baseline = false;
      }
    }
  }

  file.close();
  return 0;
}

int BaselineManager::save() {
  std::filesystem::path parent = std::filesystem::path(baseline_file_).parent_path();
  if (!parent.empty()) {
    std::error_code ec;
    std::filesystem::create_directories(parent, ec);
  }

  std::ofstream file(baseline_file_);
  if (!file.is_open()) {
    return -1;
  }

  // Write JSON manually (simple format)
  file << "{\n";
  file << "  \"version\": 1,\n";

  time_t now = time(nullptr);
  char time_str[100];
  strftime(time_str, sizeof(time_str), "%Y-%m-%dT%H:%M:%SZ", gmtime(&now));
  file << "  \"created\": \"" << time_str << "\",\n";

  file << "  \"node_info\": " << collect_metadata() << ",\n";
  file << "  \"baselines\": [\n";

  size_t count = 0;
  for (const auto& pair : baselines_) {
    const Baseline& b = pair.second;

    file << "    {\n";
    file << "      \"probe_name\": \"" << b.probe_name << "\",\n";
    file << "      \"gpu_id\": " << b.gpu_id << ",\n";
    file << "      \"value\": " << b.value << ",\n";
    file << "      \"std_dev\": " << b.std_dev << ",\n";
    file << "      \"sample_count\": " << b.sample_count << ",\n";
    file << "      \"timestamp\": " << b.timestamp;

    // Write metadata (secondary_metrics) if present
    if (!b.metadata.empty()) {
      file << ",\n";
      file << "      \"metadata\": " << b.metadata << "\n";
    } else {
      file << "\n";
    }

    count++;
    if (count < baselines_.size()) {
      file << "    },\n";
    } else {
      file << "    }\n";
    }
  }

  file << "  ]\n";
  file << "}\n";

  file.close();
  return 0;
}

int BaselineManager::create_baseline(const std::string& probe_name,
                                     uint16_t gpu_id,
                                     const std::vector<ProbeResult>& samples) {
  if (samples.empty()) {
    return -1;
  }

  std::vector<double> values;
  values.reserve(samples.size());
  for (const auto& s : samples) {
    values.push_back(s.value);
  }

  // Calculate statistics
  std::vector<double> values_copy = values;
  std::sort(values_copy.begin(), values_copy.end());
  double median = values_copy[values_copy.size() / 2];

  double sum = 0.0;
  for (double v : values) sum += v;
  double mean = sum / values.size();

  double sum_sq_diff = 0.0;
  for (double v : values) {
    double diff = v - mean;
    sum_sq_diff += diff * diff;
  }
  double std_dev = std::sqrt(sum_sq_diff / (values.size() - 1));

  Baseline baseline;
  baseline.probe_name = probe_name;
  baseline.gpu_id = gpu_id;
  baseline.value = median;
  baseline.std_dev = std_dev;
  baseline.timestamp = ProbeBase::get_timestamp_us();
  baseline.sample_count = samples.size();

  // Capture secondary_metrics from first sample (they should be identical across samples)
  if (!samples.empty() && !samples[0].secondary_metrics.empty()) {
    std::ostringstream meta_json;
    meta_json << "{";
    size_t count = 0;
    for (const auto& metric : samples[0].secondary_metrics) {
      if (count > 0) meta_json << ",";
      meta_json << "\"" << metric.first << "\":" << metric.second;
      count++;
    }
    meta_json << "}";
    baseline.metadata = meta_json.str();
  }

  std::string key = make_key(probe_name, gpu_id);
  baselines_[key] = baseline;

  return 0;
}

int BaselineManager::get_baseline(const std::string& probe_name,
                                 uint16_t gpu_id,
                                 Baseline* baseline) const {
  std::string key = make_key(probe_name, gpu_id);
  auto it = baselines_.find(key);
  if (it == baselines_.end()) {
    return 1;  // Not found
  }

  *baseline = it->second;
  return 0;
}

bool BaselineManager::has_baseline(const std::string& probe_name,
                                   uint16_t gpu_id) const {
  std::string key = make_key(probe_name, gpu_id);
  return baselines_.find(key) != baselines_.end();
}

std::vector<Baseline> BaselineManager::list_baselines(int gpu_id) const {
  std::vector<Baseline> result;

  for (const auto& pair : baselines_) {
    if (gpu_id == -1 || pair.second.gpu_id == static_cast<uint16_t>(gpu_id)) {
      result.push_back(pair.second);
    }
  }

  return result;
}

int BaselineManager::delete_baseline(const std::string& probe_name,
                                     uint16_t gpu_id) {
  std::string key = make_key(probe_name, gpu_id);
  auto it = baselines_.find(key);
  if (it == baselines_.end()) {
    return 1;  // Not found
  }

  baselines_.erase(it);
  return 0;
}

std::string BaselineManager::export_json() const {
  std::ostringstream oss;

  oss << "{\n";
  oss << "  \"version\": 1,\n";
  oss << "  \"baselines\": [\n";

  size_t count = 0;
  for (const auto& pair : baselines_) {
    const Baseline& b = pair.second;

    oss << "    {\n";
    oss << "      \"probe_name\": \"" << b.probe_name << "\",\n";
    oss << "      \"gpu_id\": " << b.gpu_id << ",\n";
    oss << "      \"value\": " << b.value << ",\n";
    oss << "      \"std_dev\": " << b.std_dev << ",\n";
    oss << "      \"sample_count\": " << b.sample_count << ",\n";
    oss << "      \"timestamp\": " << b.timestamp << "\n";

    count++;
    if (count < baselines_.size()) {
      oss << "    },\n";
    } else {
      oss << "    }\n";
    }
  }

  oss << "  ]\n";
  oss << "}\n";

  return oss.str();
}

int BaselineManager::import_json(const std::string& json, bool merge) {
  if (!merge) {
    baselines_.clear();
  }

  // Very basic parsing - in production, use a proper JSON library
  // For now, just extract baseline data
  // (This is simplified; real implementation would parse properly)

  return 0;
}

std::string BaselineManager::make_key(const std::string& probe_name,
                                     uint16_t gpu_id) {
  return probe_name + ":" + std::to_string(gpu_id);
}

std::string BaselineManager::collect_metadata() const {
  std::ostringstream oss;

  std::string rocm_version = get_rocm_version();

  oss << "{\n";
  if (record_hostname_) {
    char hostname[256] = {0};
    gethostname(hostname, sizeof(hostname) - 1);
    oss << "      \"hostname\": \"" << hostname << "\",\n";
  }
  oss << "      \"rocm_version\": \"" << rocm_version << "\"\n";
  oss << "    }";

  return oss.str();
}

std::string BaselineManager::get_rocm_version() const {
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
    size_t dash_pos = path.find_last_of("-");
    if (dash_pos != std::string::npos) {
      std::string ver_candidate = path.substr(dash_pos + 1);
      if (ver_candidate.find('.') != std::string::npos) {
        return ver_candidate;
      }
    }
  }

  return "unknown";
}

}  // namespace gpim
