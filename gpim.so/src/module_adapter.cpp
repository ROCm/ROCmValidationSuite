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
#include "include/module_adapter.h"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <unistd.h>

namespace gpim {

// ============================================================================
// GSTAdapter Implementation
// ============================================================================

GSTAdapter::GSTAdapter() : duration_ms_(10000), matrix_size_(8640) {
  gpu_id_ = 0;
}

GSTAdapter::~GSTAdapter() {
}

ProbeResult GSTAdapter::execute_and_capture() {
  ProbeResult result;
  result.probe_name = get_baseline_key();
  result.gpu_id = gpu_id_;
  result.unit = "GFLOPS";
  result.timestamp_us = ProbeBase::get_timestamp_us();
  result.passed = false;

  // Generate temporary config file
  std::string config_content = generate_config();
  std::string config_path = "/tmp/gpim_gst_adapter_" +
                           std::to_string(getpid()) + ".conf";

  std::ofstream config_file(config_path);
  if (!config_file.is_open()) {
    result.value = 0.0;
    return result;
  }
  config_file << config_content;
  config_file.close();

  // Execute RVS with GST module
  std::string cmd = "rvs -c " + config_path + " 2>&1";
  FILE* pipe = popen(cmd.c_str(), "r");
  if (!pipe) {
    unlink(config_path.c_str());
    result.value = 0.0;
    return result;
  }

  // Read output
  std::string output;
  char buffer[256];
  while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    output += buffer;
  }
  pclose(pipe);

  // Cleanup
  unlink(config_path.c_str());

  // Parse GFLOPS from output
  result.value = parse_output(output);
  result.passed = (result.value > 0.0);

  return result;
}

std::string GSTAdapter::get_module_name() const {
  return "gst";
}

std::string GSTAdapter::get_baseline_key() const {
  return "gst_gflops";
}

void GSTAdapter::set_gpu_id(uint16_t gpu_id) {
  gpu_id_ = gpu_id;
}

void GSTAdapter::set_duration(uint64_t duration_ms) {
  duration_ms_ = duration_ms;
}

void GSTAdapter::set_matrix_size(uint64_t size) {
  matrix_size_ = size;
}

std::string GSTAdapter::generate_config() {
  std::ostringstream oss;
  oss << "actions:\n"
      << "- name: gpim_gst_baseline\n"
      << "  device: " << gpu_id_ << "\n"
      << "  module: gst\n"
      << "  parallel: false\n"
      << "  duration: " << duration_ms_ << "\n"
      << "  ramp_interval: 3000\n"
      << "  log_interval: 1000\n"
      << "  max_violations: 1\n"
      << "  copy_matrix: false\n"
      << "  target_stress: 0\n"
      << "  tolerance: 0.1\n"
      << "  matrix_size_a: " << matrix_size_ << "\n"
      << "  matrix_size_b: " << matrix_size_ << "\n"
      << "  matrix_size_c: " << matrix_size_ << "\n"
      << "  ops_type: sgemm\n";

  return oss.str();
}

double GSTAdapter::parse_output(const std::string& output) {
  // Parse GST output for GFLOPS value
  // Look for pattern like "gflops: 47.3" or "gflops_1 47.3"

  size_t pos = output.find("gflops");
  if (pos == std::string::npos) {
    return 0.0;
  }

  // Find the number after "gflops"
  size_t num_start = output.find_first_of("0123456789.", pos);
  if (num_start == std::string::npos) {
    return 0.0;
  }

  size_t num_end = output.find_first_not_of("0123456789.", num_start);
  std::string gflops_str = output.substr(num_start, num_end - num_start);

  return std::stod(gflops_str);
}

// ============================================================================
// IETAdapter Implementation
// ============================================================================

IETAdapter::IETAdapter() : duration_ms_(10000) {
  gpu_id_ = 0;
}

IETAdapter::~IETAdapter() {
}

ProbeResult IETAdapter::execute_and_capture() {
  ProbeResult result;
  result.probe_name = get_baseline_key();
  result.gpu_id = gpu_id_;
  result.unit = "Watts";
  result.timestamp_us = ProbeBase::get_timestamp_us();
  result.passed = false;

  // Generate temporary config file
  std::string config_content = generate_config();
  std::string config_path = "/tmp/gpim_iet_adapter_" +
                           std::to_string(getpid()) + ".conf";

  std::ofstream config_file(config_path);
  if (!config_file.is_open()) {
    result.value = 0.0;
    return result;
  }
  config_file << config_content;
  config_file.close();

  // Execute RVS with IET module
  std::string cmd = "rvs -c " + config_path + " 2>&1";
  FILE* pipe = popen(cmd.c_str(), "r");
  if (!pipe) {
    unlink(config_path.c_str());
    result.value = 0.0;
    return result;
  }

  // Read output
  std::string output;
  char buffer[256];
  while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    output += buffer;
  }
  pclose(pipe);

  // Cleanup
  unlink(config_path.c_str());

  // Parse power from output
  result.value = parse_output(output);
  result.passed = (result.value > 0.0);

  return result;
}

std::string IETAdapter::get_module_name() const {
  return "iet";
}

std::string IETAdapter::get_baseline_key() const {
  return "iet_power";
}

void IETAdapter::set_gpu_id(uint16_t gpu_id) {
  gpu_id_ = gpu_id;
}

void IETAdapter::set_duration(uint64_t duration_ms) {
  duration_ms_ = duration_ms;
}

std::string IETAdapter::generate_config() {
  std::ostringstream oss;
  oss << "actions:\n"
      << "- name: gpim_iet_baseline\n"
      << "  device: " << gpu_id_ << "\n"
      << "  module: iet\n"
      << "  parallel: false\n"
      << "  duration: " << duration_ms_ << "\n"
      << "  ramp_interval: 3000\n"
      << "  sample_interval: 1000\n"
      << "  log_interval: 1000\n"
      << "  max_violations: 1\n"
      << "  target_power: 0\n"
      << "  tolerance: 0.1\n"
      << "  matrix_size: 8640\n"
      << "  ops_type: sgemm\n";

  return oss.str();
}

double IETAdapter::parse_output(const std::string& output) {
  // Parse IET output for power value
  // Look for pattern like "power: 540" or "avg_power 540"

  size_t pos = output.find("power");
  if (pos == std::string::npos) {
    return 0.0;
  }

  // Find the number after "power"
  size_t num_start = output.find_first_of("0123456789.", pos);
  if (num_start == std::string::npos) {
    return 0.0;
  }

  size_t num_end = output.find_first_not_of("0123456789.", num_start);
  std::string power_str = output.substr(num_start, num_end - num_start);

  return std::stod(power_str);
}

// ============================================================================
// BABELAdapter Implementation
// ============================================================================

BABELAdapter::BABELAdapter() {
  gpu_id_ = 0;
}

BABELAdapter::~BABELAdapter() {
}

ProbeResult BABELAdapter::execute_and_capture() {
  ProbeResult result;
  result.probe_name = get_baseline_key();
  result.gpu_id = gpu_id_;
  result.unit = "GB/s";
  result.timestamp_us = ProbeBase::get_timestamp_us();
  result.passed = false;

  // Generate temporary config file
  std::string config_content = generate_config();
  std::string config_path = "/tmp/gpim_babel_adapter_" +
                           std::to_string(getpid()) + ".conf";

  std::ofstream config_file(config_path);
  if (!config_file.is_open()) {
    result.value = 0.0;
    return result;
  }
  config_file << config_content;
  config_file.close();

  // Execute RVS with BABEL module
  std::string cmd = "rvs -c " + config_path + " 2>&1";
  FILE* pipe = popen(cmd.c_str(), "r");
  if (!pipe) {
    unlink(config_path.c_str());
    result.value = 0.0;
    return result;
  }

  // Read output
  std::string output;
  char buffer[256];
  while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    output += buffer;
  }
  pclose(pipe);

  // Cleanup
  unlink(config_path.c_str());

  // Parse bandwidth from output
  result.value = parse_output(output);
  result.passed = (result.value > 0.0);

  return result;
}

std::string BABELAdapter::get_module_name() const {
  return "babel";
}

std::string BABELAdapter::get_baseline_key() const {
  return "babel_bandwidth";
}

void BABELAdapter::set_gpu_id(uint16_t gpu_id) {
  gpu_id_ = gpu_id;
}

std::string BABELAdapter::generate_config() {
  std::ostringstream oss;
  oss << "actions:\n"
      << "- name: gpim_babel_baseline\n"
      << "  device: " << gpu_id_ << "\n"
      << "  module: babel\n"
      << "  test: copy\n"
      << "  num_iter: 100\n"
      << "  array_size: 268435456\n";

  return oss.str();
}

double BABELAdapter::parse_output(const std::string& output) {
  // Parse BABEL output for bandwidth value
  // Look for pattern like "bandwidth: 1420" or "copy 1420.5"

  size_t pos = output.find("bandwidth");
  if (pos == std::string::npos) {
    pos = output.find("copy");
  }

  if (pos == std::string::npos) {
    return 0.0;
  }

  // Find the number after keyword
  size_t num_start = output.find_first_of("0123456789.", pos);
  if (num_start == std::string::npos) {
    return 0.0;
  }

  size_t num_end = output.find_first_not_of("0123456789.", num_start);
  std::string bw_str = output.substr(num_start, num_end - num_start);

  return std::stod(bw_str);
}

} // namespace gpim
