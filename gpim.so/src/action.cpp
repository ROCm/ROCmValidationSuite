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
#include "include/action.h"
#include "include/probes/kernel_launch_probe.h"
#include "include/probes/gemm_probe.h"
#include "include/probes/membw_probe.h"
#include "include/probes/rocm_version_probe.h"
#include "include/probes/driver_overhead_probe.h"
#include "include/baseline_manager.h"
#include "include/regression_detector.h"
#include "include/diagnostic_engine.h"

#include "include/rvsloglp.h"
#include "include/gpu_util.h"
#include "include/rvs_util.h"

#include <hip/hip_runtime.h>
#include <memory>
#include <sstream>
#include <map>

using std::string;
using std::vector;
using std::map;

gpim_action::gpim_action() {
  module_name = "gpim";
  bjson = false;

  mode_ = "single";
  baseline_auto_create_ = false;
  baseline_record_hostname_ = false;
  baseline_path_ = "/var/lib/rvs/gpim_baselines.json";
  baseline_update_policy_ = "manual";
  regression_detection_enabled_ = true;
  consecutive_failures_required_ = 1;
  diagnostics_enabled_ = true;
  report_log_interval_ = 1000;
  fail_on_regression_ = false;
  total_probes_executed_ = 0;
  total_regressions_detected_ = 0;

  // Default thresholds
  regression_thresholds_["warning"] = 5.0;
  regression_thresholds_["minor"] = 10.0;
  regression_thresholds_["major"] = 20.0;
  regression_thresholds_["critical"] = 50.0;

  // Default probes enabled
  probes_enabled_["kernel_launch"] = true;
  probes_enabled_["gemm"] = true;
  probes_enabled_["memory_bandwidth"] = true;
  probes_enabled_["rocm_version"] = true;
  probes_enabled_["driver_overhead"] = true;
}

gpim_action::~gpim_action() {
  // Unique pointers auto-cleanup
}

int gpim_action::run() {
  rvs::lp::Log("[GPIM] Starting GPU Performance Integrity Monitor", rvs::loginfo);

  // Parse configuration
  if (property_get_all() != 0) {
    rvs::lp::Err("Configuration parsing failed", module_name, action_name);
    return -1;
  }

  // Get HIP device count
  int hip_num_gpu_devices = 0;
  hipGetDeviceCount(&hip_num_gpu_devices);
  if (hip_num_gpu_devices == 0) {
    rvs::lp::Err("No AMD GPUs found", module_name, action_name);
    return -1;
  }

  // Fetch GPU list using RVS utility function
  map<int, uint16_t> gpim_gpus_device_index;
  bool amd_gpus_found = fetch_gpu_list(hip_num_gpu_devices, gpim_gpus_device_index,
                                        property_device, property_device_id,
                                        property_device_all, property_device_index,
                                        property_device_index_all);

  if (!amd_gpus_found) {
    rvs::lp::Err("No compatible AMD GPUs found", module_name, action_name);
    return -1;
  }

  // Store GPU device index map
  gpim_gpus_device_index_ = gpim_gpus_device_index;

  // Log GPU device map
  std::ostringstream gpu_info;
  gpu_info << "[GPIM] Found " << gpim_gpus_device_index_.size() << " GPU(s) to monitor";
  rvs::lp::Log(gpu_info.str(), rvs::loginfo);

  // Initialize baseline manager
  baseline_manager_.reset(new gpim::BaselineManager(baseline_path_));
  baseline_manager_->set_record_hostname(baseline_record_hostname_);

  // Initialize regression detector
  regression_detector_.reset(new gpim::RegressionDetector(baseline_manager_.get()));
  regression_detector_->set_threshold(gpim::RegressionSeverity::WARNING,
                                      regression_thresholds_["warning"]);
  regression_detector_->set_threshold(gpim::RegressionSeverity::MINOR,
                                      regression_thresholds_["minor"]);
  regression_detector_->set_threshold(gpim::RegressionSeverity::MAJOR,
                                      regression_thresholds_["major"]);
  regression_detector_->set_threshold(gpim::RegressionSeverity::CRITICAL,
                                      regression_thresholds_["critical"]);
  regression_detector_->set_consecutive_failures_required(consecutive_failures_required_);

  // Initialize diagnostic engine
  diagnostic_engine_.reset(new gpim::DiagnosticEngine());

  // Dispatch based on mode
  int result = 0;
  if (mode_ == "baseline") {
    result = run_baseline_mode();
  } else if (mode_ == "single") {
    result = run_single_mode();
    // Log validation summary (only for validation modes)
    log_summary_report();
  } else {
    rvs::lp::Err("Unknown mode: " + mode_, module_name, action_name);
    return -1;
  }

  // Output JSON if requested
  if (bjson) {
    output_json_report();
  }

  return result;
}

int gpim_action::property_get_all() {
  // Mode
  if (property_get("mode", &mode_) != 0) {
    mode_ = "single";  // Default
  }

  // Baseline config
  std::string baseline_path_str;
  if (property_get("baseline_path", &baseline_path_str) == 0) {
    baseline_path_ = baseline_path_str;
  }

  property_get("baseline_auto_create", &baseline_auto_create_);
  property_get("baseline_record_hostname", &baseline_record_hostname_);

  // Regression config
  property_get("regression_detection_enabled", &regression_detection_enabled_);

  float threshold;
  if (property_get("regression_threshold_warning", &threshold) == 0) {
    regression_thresholds_["warning"] = threshold;
  }
  if (property_get("regression_threshold_minor", &threshold) == 0) {
    regression_thresholds_["minor"] = threshold;
  }
  if (property_get("regression_threshold_major", &threshold) == 0) {
    regression_thresholds_["major"] = threshold;
  }
  if (property_get("regression_threshold_critical", &threshold) == 0) {
    regression_thresholds_["critical"] = threshold;
  }

  int failures;
  if (property_get_int("regression_consecutive_failures", &failures) == 0) {
    consecutive_failures_required_ = failures;
  }

  // Diagnostics
  property_get("diagnostics_enabled", &diagnostics_enabled_);

  // Reporting
  property_get("fail_on_regression", &fail_on_regression_);

  // Probes
  property_get("probe_kernel_launch_enabled", &probes_enabled_["kernel_launch"]);
  property_get("probe_gemm_enabled", &probes_enabled_["gemm"]);
  property_get("probe_memory_bandwidth_enabled", &probes_enabled_["memory_bandwidth"]);
  property_get("probe_rocm_version_enabled", &probes_enabled_["rocm_version"]);
  property_get("probe_driver_overhead_enabled", &probes_enabled_["driver_overhead"]);

  return 0;
}

int gpim_action::run_baseline_mode() {
  rvs::lp::Log("[GPIM] Running in BASELINE mode", rvs::loginfo);

  int baseline_runs = 25;
  property_get_int("baseline_runs", &baseline_runs);

  std::ostringstream log_msg;
  log_msg << "[GPIM] Processing " << gpim_gpus_device_index_.size() << " GPU(s)";
  rvs::lp::Log(log_msg.str(), rvs::loginfo);

  int baselines_created = 0;

  // Iterate over GPU device index map
  for (auto it = gpim_gpus_device_index_.begin(); it != gpim_gpus_device_index_.end(); it++) {
    int hip_device_index = it->first;
    uint16_t gpu_id = it->second;

    std::ostringstream oss;
    oss << "[GPIM] Creating baselines for GPU " << gpu_id
        << " (HIP device " << hip_device_index << ", " << baseline_runs << " samples)";
    rvs::lp::Log(oss.str(), rvs::loginfo);

    // Kernel launch probe
    if (probes_enabled_["kernel_launch"]) {
      gpim::KernelLaunchProbe probe;
      if (probe.initialize(hip_device_index) == 0) {
        std::vector<gpim::ProbeResult> samples;
        for (int i = 0; i < baseline_runs; i++) {
          samples.push_back(probe.execute());
        }
        baseline_manager_->create_baseline(probe.name(), gpu_id, samples);
        rvs::lp::Log("[GPIM] Created baseline for kernel_launch_latency", rvs::loginfo);
        probe.cleanup();
        baselines_created++;
      } else {
        rvs::lp::Err("Failed to initialize kernel_launch probe for GPU " + std::to_string(gpu_id),
                     module_name, action_name);
      }
    }

    // GEMM probe
    if (probes_enabled_["gemm"]) {
      gpim::GEMMProbe probe;
      if (probe.initialize(hip_device_index) == 0) {
        std::vector<gpim::ProbeResult> samples;
        for (int i = 0; i < baseline_runs; i++) {
          samples.push_back(probe.execute());
        }
        baseline_manager_->create_baseline(probe.name(), gpu_id, samples);
        rvs::lp::Log("[GPIM] Created baseline for gemm_throughput", rvs::loginfo);
        probe.cleanup();
        baselines_created++;
      } else {
        rvs::lp::Err("Failed to initialize GEMM probe for GPU " + std::to_string(gpu_id),
                     module_name, action_name);
      }
    }

    // Memory bandwidth probe
    if (probes_enabled_["memory_bandwidth"]) {
      gpim::MemBandwidthProbe probe;
      if (probe.initialize(hip_device_index) == 0) {
        std::vector<gpim::ProbeResult> samples;
        for (int i = 0; i < baseline_runs; i++) {
          samples.push_back(probe.execute());
        }
        baseline_manager_->create_baseline(probe.name(), gpu_id, samples);
        rvs::lp::Log("[GPIM] Created baseline for memory_bandwidth", rvs::loginfo);
        probe.cleanup();
        baselines_created++;
      } else {
        rvs::lp::Err("Failed to initialize memory bandwidth probe for GPU " + std::to_string(gpu_id),
                     module_name, action_name);
      }
    }
    // ROCM version probe
    if (probes_enabled_["rocm_version"]) {
      gpim::ROCmVersionProbe probe;
      if (probe.initialize(hip_device_index) == 0) {
        std::vector<gpim::ProbeResult> samples;
        for (int i = 0; i < baseline_runs; i++) {
          samples.push_back(probe.execute());
        }
        baseline_manager_->create_baseline(probe.name(), gpu_id, samples);
        rvs::lp::Log("[GPIM] Created baseline for ROCM version", rvs::loginfo);
        probe.cleanup();
        baselines_created++;
      } else {
        rvs::lp::Err("Failed to initialize ROCM version probe for GPU " + std::to_string(gpu_id),
                     module_name, action_name);
      }
    }
    // Driver overhead probe
    if (probes_enabled_["driver_overhead"]) {
      gpim::DriverOverheadProbe probe;
      if (probe.initialize(hip_device_index) == 0) {
        std::vector<gpim::ProbeResult> samples;
        for (int i = 0; i < baseline_runs; i++) {
          samples.push_back(probe.execute());
        }
        baseline_manager_->create_baseline(probe.name(), gpu_id, samples);
        rvs::lp::Log("[GPIM] Created baseline for driver overhead", rvs::loginfo);
        probe.cleanup();
        baselines_created++;
      } else {
        rvs::lp::Err("Failed to initialize driver overhead probe for GPU " + std::to_string(gpu_id),
                     module_name, action_name);
      }
    }
  }

  // Save baselines
  if (baseline_manager_->save() != 0) {
    rvs::lp::Err("Failed to save baselines to " + baseline_path_, module_name, action_name);
    return -1;
  }

  rvs::lp::Log("[GPIM] Baselines saved to " + baseline_path_, rvs::logresults);

  // Log baseline mode summary
  std::ostringstream summary;
  summary << "\n[GPIM] Baseline Creation Summary:\n"
          << "  GPUs Processed: " << gpim_gpus_device_index_.size() << "\n"
          << "  Baselines Created: " << baselines_created << "\n"
          << "  Samples per Baseline: " << baseline_runs;
  rvs::lp::Log(summary.str(), rvs::logresults);

  return 0;
}

int gpim_action::run_single_mode() {
  rvs::lp::Log("[GPIM] Running in SINGLE validation mode", rvs::loginfo);

  // Load baselines
  int rc = baseline_manager_->load();
  if (rc != 0) {
    if (baseline_auto_create_) {
      rvs::lp::Log("[GPIM] No baselines found, creating new ones", rvs::loginfo);
      return run_baseline_mode();
    } else {
      rvs::lp::Err("No baselines found at " + baseline_path_, module_name, action_name);
      return -1;
    }
  }

  bool any_regression = false;

  // Iterate over GPU device index map
  for (auto it = gpim_gpus_device_index_.begin(); it != gpim_gpus_device_index_.end(); it++) {
    int hip_device_index = it->first;
    uint16_t gpu_id = it->second;

    std::ostringstream oss;
    oss << "[GPIM] Validating GPU " << gpu_id << " (HIP device " << hip_device_index << ")";
    rvs::lp::Log(oss.str(), rvs::loginfo);

    std::vector<gpim::ProbeResult> results;

    // Run kernel launch probe
    if (probes_enabled_["kernel_launch"]) {
      gpim::KernelLaunchProbe probe;
      if (probe.initialize(hip_device_index) == 0) {
        gpim::ProbeResult result = probe.execute();
        result.gpu_id = gpu_id;  // Store RVS GPU ID for baseline comparison
        results.push_back(result);
        total_probes_executed_++;

        std::ostringstream msg;
        msg << "  kernel_launch_latency: " << result.value << " " << result.unit;
        rvs::lp::Log(msg.str(), rvs::loginfo);

        probe.cleanup();
      }
    }

    // Run GEMM probe
    if (probes_enabled_["gemm"]) {
      gpim::GEMMProbe probe;
      if (probe.initialize(hip_device_index) == 0) {
        gpim::ProbeResult result = probe.execute();
        result.gpu_id = gpu_id;  // Store RVS GPU ID for baseline comparison
        results.push_back(result);
        total_probes_executed_++;

        std::ostringstream msg;
        msg << "  gemm_throughput: " << result.value << " " << result.unit;
        rvs::lp::Log(msg.str(), rvs::loginfo);

        probe.cleanup();
      }
    }

    // Run memory bandwidth probe
    if (probes_enabled_["memory_bandwidth"]) {
      gpim::MemBandwidthProbe probe;
      if (probe.initialize(hip_device_index) == 0) {
        gpim::ProbeResult result = probe.execute();
        result.gpu_id = gpu_id;  // Store RVS GPU ID for baseline comparison
        results.push_back(result);
        total_probes_executed_++;

        std::ostringstream msg;
        msg << "  memory_bandwidth: " << result.value << " " << result.unit;
        rvs::lp::Log(msg.str(), rvs::loginfo);

        probe.cleanup();
      }
    }
    // Run ROCM version probe if enabled
    if (probes_enabled_["rocm_version"]) {
      gpim::ROCmVersionProbe probe;
      if (probe.initialize(hip_device_index) == 0) {
        gpim::ProbeResult result = probe.execute();
        result.gpu_id = gpu_id;  // Store RVS GPU ID for baseline comparison
        results.push_back(result);
        total_probes_executed_++;

        std::ostringstream msg;
        msg << "  ROCM version: " << result.value;
        rvs::lp::Log(msg.str(), rvs::loginfo);

        probe.cleanup();  
      }
    }
    // Run driver overhead probe if enabled
    if (probes_enabled_["driver_overhead"]) {
      gpim::DriverOverheadProbe probe;
      if (probe.initialize(hip_device_index) == 0) {
        gpim::ProbeResult result = probe.execute();
        result.gpu_id = gpu_id;  // Store RVS GPU ID for baseline comparison
        results.push_back(result);
        total_probes_executed_++;

        std::ostringstream msg;
        msg << "  driver overhead: " << result.value << " " << result.unit;
        rvs::lp::Log(msg.str(), rvs::loginfo);

        probe.cleanup();
      }
    }
    // Detect regressions
    if (regression_detection_enabled_) {
      for (const auto& result : results) {
        gpim::RegressionReport regression = regression_detector_->detect(result);

        if (regression.severity != gpim::RegressionSeverity::NONE) {
          total_regressions_detected_++;
          any_regression = true;

          std::string severity_str = gpim::severity_to_string(regression.severity);
          severity_counts_[severity_str]++;

          std::ostringstream msg;
          msg << "[" << severity_str << "] GPU " << gpu_id << ": "
              << result.probe_name << " degraded by "
              << regression.degradation_pct << "% ("
              << regression.baseline_value << " -> "
              << regression.current_value << " " << result.unit << ")";
          rvs::lp::Log(msg.str(), rvs::logerror);

          // Run diagnostics
          if (diagnostics_enabled_) {
            auto findings = diagnostic_engine_->diagnose(result, regression);

            if (!findings.empty()) {
              rvs::lp::Log("  Diagnostic findings (ranked by confidence):", rvs::loginfo);

              for (size_t i = 0; i < findings.size() && i < 3; i++) {
                const auto& finding = findings[i];
                std::ostringstream diag_msg;
                diag_msg << "  [" << finding.confidence << "%] "
                        << finding.description << "\n"
                        << "    Evidence: " << finding.evidence << "\n"
                        << "    Recommendation:\n";

                // Split recommendation into lines
                std::string rec = finding.recommendation;
                size_t pos = 0;
                while ((pos = rec.find('\n')) != std::string::npos) {
                  diag_msg << "      " << rec.substr(0, pos) << "\n";
                  rec.erase(0, pos + 1);
                }
                if (!rec.empty()) {
                  diag_msg << "      " << rec;
                }

                rvs::lp::Log(diag_msg.str(), rvs::loginfo);
              }
            }
          }
        }
      }
    }
  }

  if (fail_on_regression_ && any_regression) {
    return 1;  // Non-zero exit code for regression detected
  }

  return 0;
}

void gpim_action::log_summary_report() {
  std::ostringstream oss;
  oss << "\n[GPIM] Summary Report:\n"
      << "  Total Probes Executed: " << total_probes_executed_ << "\n"
      << "  Regressions Detected: " << total_regressions_detected_;

  if (total_regressions_detected_ > 0) {
    oss << "\n  By Severity:\n";
    for (const auto& pair : severity_counts_) {
      oss << "    " << pair.first << ": " << pair.second << "\n";
    }
  }

  rvs::lp::Log(oss.str(), rvs::logresults);
}

void gpim_action::output_json_report() {
  // Use RVS JSON logging API
  void* json_root = rvs::lp::LogRecordCreate(module_name.c_str(),
                                              action_name.c_str(),
                                              rvs::loginfo, 0, 0);
  if (!json_root) return;

  rvs::lp::AddString(json_root, "mode", mode_.c_str());
  rvs::lp::AddInt(json_root, "total_probes_executed", total_probes_executed_);
  rvs::lp::AddInt(json_root, "regressions_detected", total_regressions_detected_);

  rvs::lp::LogRecordFlush(json_root);
}

