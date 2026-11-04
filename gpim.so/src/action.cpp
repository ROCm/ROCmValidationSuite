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
#include "include/baseline_manager.h"
#include "include/diagnostic_engine.h"
#include "include/probe_registry.h"
#include "include/regression_detector.h"
#include "include/telemetry_sampler.h"

#include "include/rvsloglp.h"
#include "include/gpu_util.h"
#include "include/rvs_util.h"

#include <hip/hip_runtime.h>

#include <map>
#include <memory>
#include <sstream>

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
  check_samples_ = 5;
  gemm_dtype_ = "bf16";
  gemm_seconds_ = 2.0;
  gemm_matrix_size_ = 0;
  membw_array_size_mb_ = 0;

  regression_thresholds_["warning"] = 5.0;
  regression_thresholds_["minor"] = 10.0;
  regression_thresholds_["major"] = 20.0;
  regression_thresholds_["critical"] = 50.0;

  for (const auto& name : gpim::ProbeRegistry::names()) {
    probes_enabled_[name] = true;
  }
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

  property_get_int("check_samples", &check_samples_);
  if (check_samples_ < 1) check_samples_ = 1;

  std::string gemm_dtype;
  if (property_get("gemm_dtype", &gemm_dtype) == 0 && !gemm_dtype.empty()) {
    gemm_dtype_ = gemm_dtype;
  }
  float gemm_seconds = 0.0f;
  if (property_get("gemm_seconds", &gemm_seconds) == 0 && gemm_seconds > 0.0f) {
    gemm_seconds_ = gemm_seconds;
  }
  property_get_int("gemm_matrix_size", &gemm_matrix_size_);
  property_get_int("membw_array_size_mb", &membw_array_size_mb_);

  for (const auto& name : gpim::ProbeRegistry::names()) {
    property_get("probe_" + name + "_enabled", &probes_enabled_[name]);
  }
  // Older confs used this name for runtime_api_latency.
  bool legacy_driver = true;
  if (property_get("probe_driver_overhead_enabled", &legacy_driver) == 0) {
    probes_enabled_["runtime_api_latency"] = legacy_driver;
  }

  return 0;
}

int gpim_action::run_baseline_mode() {
  rvs::lp::Log("[GPIM] Running in BASELINE mode", rvs::loginfo);

  int baseline_runs = 25;
  property_get_int("baseline_runs", &baseline_runs);
  if (baseline_runs < 2) baseline_runs = 2;

  std::ostringstream log_msg;
  log_msg << "[GPIM] Processing " << gpim_gpus_device_index_.size()
          << " GPU(s), " << baseline_runs << " samples each";
  rvs::lp::Log(log_msg.str(), rvs::loginfo);

  int baselines_created = 0;
  int probe_failures = 0;

  for (auto it = gpim_gpus_device_index_.begin();
       it != gpim_gpus_device_index_.end(); ++it) {
    int hip_device_index = it->first;
    uint16_t gpu_id = it->second;

    for (const auto& probe_name : gpim::ProbeRegistry::names()) {
      if (!probes_enabled_[probe_name]) continue;

      std::unique_ptr<gpim::ProbeBase> probe = gpim::ProbeRegistry::create(probe_name);
      if (!probe) continue;
      probe->set_config("dtype", gemm_dtype_);
      probe->set_config("seconds", std::to_string(gemm_seconds_));
      probe->set_config("matrix_size", std::to_string(gemm_matrix_size_));
      probe->set_config("array_size_mb", std::to_string(membw_array_size_mb_));

      if (probe->initialize(hip_device_index) != 0) {
        rvs::lp::Err("Failed to initialize " + probe_name + " for GPU " +
                     std::to_string(gpu_id), module_name, action_name);
        probe_failures++;
        continue;
      }

      gpim::TelemetrySampler sampler;
      sampler.start(probe->smi_handle());
      std::map<std::string, std::vector<gpim::ProbeResult>> by_metric;
      bool failed = false;
      for (int i = 0; i < baseline_runs; i++) {
        gpim::ProbeResult sample = probe->execute();
        sample.gpu_id = gpu_id;
        if (!sample.passed) {
          failed = true;
          break;
        }
        for (const auto& metric : sample.metrics) {
          if (!metric.monitor) continue;
          gpim::ProbeResult one = sample;
          one.metrics.clear();
          one.probe_name = metric.name;
          one.value = metric.value;
          one.unit = metric.unit;
          one.direction = metric.direction;
          by_metric[metric.name].push_back(one);
        }
      }
      gpim::TelemetrySnapshot snap = sampler.stop();
      probe->cleanup();

      if (failed || by_metric.empty()) {
        if (failed) {
          rvs::lp::Err(probe_name + " failed on GPU " + std::to_string(gpu_id),
                       module_name, action_name);
          probe_failures++;
        }
        continue;
      }

      for (auto& pair : by_metric) {
        for (auto& sample : pair.second) {
          if (!snap.valid) continue;
          sample.telemetry = snap;
          sample.secondary_metrics["telemetry_temp_c_avg"] = snap.temp_c_avg;
          sample.secondary_metrics["telemetry_clock_mhz_avg"] = snap.clock_mhz_avg;
          sample.secondary_metrics["telemetry_power_w_avg"] = snap.power_w_avg;
          sample.secondary_metrics["telemetry_ppt_delta"] =
              static_cast<double>(snap.ppt_residency_delta);
          sample.secondary_metrics["telemetry_thermal_delta"] =
              static_cast<double>(snap.thermal_residency_delta);
        }
        if (baseline_manager_->create_baseline(pair.first, gpu_id, pair.second) != 0) {
          probe_failures++;
          continue;
        }
        std::ostringstream msg;
        msg << "[GPIM] GPU " << gpu_id << " baseline " << pair.first
            << " = " << pair.second.front().value << " " << pair.second.front().unit
            << " (" << pair.second.size() << " samples)";
        rvs::lp::Log(msg.str(), rvs::loginfo);
        baselines_created++;
      }
    }
  }

  if (baseline_manager_->save() != 0) {
    rvs::lp::Err("Failed to save baselines to " + baseline_path_, module_name, action_name);
    return -1;
  }

  std::ostringstream summary;
  summary << "\n[GPIM] Baseline Creation Summary:\n"
          << "  GPUs Processed: " << gpim_gpus_device_index_.size() << "\n"
          << "  Baselines Created: " << baselines_created << "\n"
          << "  Probe Failures: " << probe_failures << "\n"
          << "  Samples per Baseline: " << baseline_runs;
  rvs::lp::Log(summary.str(), rvs::logresults);
  rvs::lp::Log("[GPIM] Baselines saved to " + baseline_path_, rvs::logresults);
  return probe_failures == 0 ? 0 : 1;
}

int gpim_action::run_single_mode() {
  rvs::lp::Log("[GPIM] Running in SINGLE validation mode", rvs::loginfo);

  int rc = baseline_manager_->load();
  if (rc != 0) {
    if (baseline_auto_create_) {
      rvs::lp::Log("[GPIM] No baselines found, creating new ones", rvs::loginfo);
      return run_baseline_mode();
    }
    rvs::lp::Err("No baselines found at " + baseline_path_, module_name, action_name);
    return -1;
  }

  bool any_regression = false;
  int probe_failures = 0;

  for (auto it = gpim_gpus_device_index_.begin();
       it != gpim_gpus_device_index_.end(); ++it) {
    int hip_device_index = it->first;
    uint16_t gpu_id = it->second;

    std::ostringstream oss;
    oss << "[GPIM] Validating GPU " << gpu_id << " (HIP device " << hip_device_index << ")";
    rvs::lp::Log(oss.str(), rvs::loginfo);

    for (const auto& probe_name : gpim::ProbeRegistry::names()) {
      if (!probes_enabled_[probe_name]) continue;

      std::unique_ptr<gpim::ProbeBase> probe = gpim::ProbeRegistry::create(probe_name);
      if (!probe) continue;
      probe->set_config("dtype", gemm_dtype_);
      probe->set_config("seconds", std::to_string(gemm_seconds_));
      probe->set_config("matrix_size", std::to_string(gemm_matrix_size_));
      probe->set_config("array_size_mb", std::to_string(membw_array_size_mb_));

      if (probe->initialize(hip_device_index) != 0) {
        rvs::lp::Err("Failed to initialize " + probe_name + " for GPU " +
                     std::to_string(gpu_id), module_name, action_name);
        probe_failures++;
        continue;
      }

      gpim::TelemetrySampler sampler;
      sampler.start(probe->smi_handle());
      std::vector<gpim::ProbeResult> samples;
      bool failed = false;
      for (int i = 0; i < check_samples_; i++) {
        gpim::ProbeResult sample = probe->execute();
        sample.gpu_id = gpu_id;
        if (!sample.passed) {
          failed = true;
          for (const auto& metric : sample.metrics) {
            if (metric.monitor) continue;
            std::ostringstream msg;
            msg << "  " << metric.name << ": " << metric.value << " " << metric.unit
                << " (failed)";
            rvs::lp::Log(msg.str(), rvs::logerror);
          }
          break;
        }
        samples.push_back(sample);
      }
      gpim::TelemetrySnapshot snap = sampler.stop();
      probe->cleanup();

      if (failed || samples.empty()) {
        rvs::lp::Err(probe_name + " failed on GPU " + std::to_string(gpu_id),
                     module_name, action_name);
        probe_failures++;
        continue;
      }

      if (!samples.front().metrics.empty()) {
        for (const auto& metric : samples.front().metrics) {
          if (metric.monitor) continue;
          std::ostringstream msg;
          msg << "  " << metric.name << ": " << metric.value << " " << metric.unit;
          rvs::lp::Log(msg.str(), metric.value == 0.0 ? rvs::logerror : rvs::loginfo);
        }
      }

      std::map<std::string, std::vector<double>> values;
      std::map<std::string, gpim::ProbeResult> exemplar;
      for (const auto& sample : samples) {
        for (const auto& metric : sample.metrics) {
          if (!metric.monitor) continue;
          values[metric.name].push_back(metric.value);
          if (!exemplar.count(metric.name)) {
            gpim::ProbeResult one = sample;
            one.metrics.clear();
            one.probe_name = metric.name;
            one.unit = metric.unit;
            one.direction = metric.direction;
            exemplar[metric.name] = one;
          }
        }
      }

      for (auto& pair : values) {
        gpim::ProbeResult result = exemplar[pair.first];
        std::vector<double> copy = pair.second;
        result.value = gpim::ProbeBase::compute_median(copy);
        result.gpu_id = gpu_id;
        if (snap.valid) {
          result.telemetry = snap;
          result.secondary_metrics["telemetry_temp_c_avg"] = snap.temp_c_avg;
          result.secondary_metrics["telemetry_clock_mhz_avg"] = snap.clock_mhz_avg;
          result.secondary_metrics["telemetry_power_w_avg"] = snap.power_w_avg;
        }
        total_probes_executed_++;

        std::ostringstream msg;
        msg << "  " << result.probe_name << ": " << result.value << " " << result.unit
            << " (" << pair.second.size() << " samples)";
        rvs::lp::Log(msg.str(), rvs::loginfo);

        if (!regression_detection_enabled_) continue;
        gpim::RegressionReport regression = regression_detector_->detect(result);
        if (regression.severity == gpim::RegressionSeverity::NONE) continue;

        total_regressions_detected_++;
        any_regression = true;
        std::string severity_str = gpim::severity_to_string(regression.severity);
        severity_counts_[severity_str]++;

        std::ostringstream reg;
        reg << "[" << severity_str << "] GPU " << gpu_id << ": "
            << result.probe_name << " degraded by "
            << regression.degradation_pct << "% ("
            << regression.baseline_value << " -> "
            << regression.current_value << " " << result.unit << ")";
        rvs::lp::Log(reg.str(), rvs::logerror);

        if (!diagnostics_enabled_) continue;
        auto findings = diagnostic_engine_->diagnose(result, regression);
        for (size_t i = 0; i < findings.size() && i < 3; i++) {
          std::ostringstream diag;
          diag << "  [" << findings[i].confidence << "%] "
               << findings[i].description << "\n    Evidence: "
               << findings[i].evidence << "\n    Recommendation: "
               << findings[i].recommendation;
          rvs::lp::Log(diag.str(), rvs::loginfo);
        }
      }
    }
  }

  if (probe_failures > 0) return 1;
  if (fail_on_regression_ && any_regression) return 1;
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

