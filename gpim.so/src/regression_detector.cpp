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
#include "include/regression_detector.h"
#include <cmath>

namespace gpim {

RegressionDetector::RegressionDetector(BaselineManager* baseline_mgr)
    : baseline_mgr_(baseline_mgr), consecutive_failures_required_(1) {
  thresholds_ = get_default_thresholds();
}

RegressionDetector::~RegressionDetector() {
}

void RegressionDetector::set_threshold(RegressionSeverity severity,
                                       double threshold_pct) {
  thresholds_[severity] = threshold_pct;
}

void RegressionDetector::set_consecutive_failures_required(int count) {
  consecutive_failures_required_ = count;
}

RegressionReport RegressionDetector::detect(const ProbeResult& result) {
  RegressionReport report;
  report.probe_name = result.probe_name;
  report.gpu_id = result.gpu_id;
  report.current_value = result.value;
  report.timestamp = result.timestamp_us;
  report.severity = RegressionSeverity::NONE;

  // Get baseline
  Baseline baseline;
  int rc = baseline_mgr_->get_baseline(result.probe_name, result.gpu_id,
                                       &baseline);
  if (rc != 0) {
    // No baseline available
    return report;
  }

  report.baseline_value = baseline.value;

  // Calculate degradation
  bool higher_is_better = (result.direction == MetricDirection::HIGHER_IS_BETTER);
  double degradation_pct = calculate_degradation(baseline.value,
                                                 result.value,
                                                 higher_is_better);

  report.degradation_pct = degradation_pct;

  // Classify severity
  report.severity = classify_severity(degradation_pct);

  // Check consecutive failures
  if (report.severity != RegressionSeverity::NONE) {
    std::string key = make_key(result.probe_name, result.gpu_id);
    failure_counts_[key]++;

    if (failure_counts_[key] < consecutive_failures_required_) {
      // Not enough consecutive failures yet
      report.severity = RegressionSeverity::NONE;
    }
  } else {
    // Reset failure counter
    std::string key = make_key(result.probe_name, result.gpu_id);
    failure_counts_[key] = 0;
  }

  return report;
}

std::vector<RegressionReport> RegressionDetector::detect_batch(
    const std::vector<ProbeResult>& results) {
  std::vector<RegressionReport> reports;

  for (const auto& result : results) {
    RegressionReport report = detect(result);
    if (report.severity != RegressionSeverity::NONE) {
      reports.push_back(report);
    }
  }

  return reports;
}

void RegressionDetector::reset_failure_counters() {
  failure_counts_.clear();
}

std::map<RegressionSeverity, double> RegressionDetector::get_default_thresholds() {
  std::map<RegressionSeverity, double> defaults;
  defaults[RegressionSeverity::WARNING] = 5.0;   // 5%
  defaults[RegressionSeverity::MINOR] = 10.0;    // 10%
  defaults[RegressionSeverity::MAJOR] = 20.0;    // 20%
  defaults[RegressionSeverity::CRITICAL] = 50.0; // 50%
  return defaults;
}

double RegressionDetector::calculate_degradation(double baseline, double current,
                                                bool higher_is_better) {
  if (baseline == 0.0) {
    return 0.0;
  }

  if (higher_is_better) {
    // For throughput metrics (higher is better)
    // Degradation is positive when performance drops
    return ((baseline - current) / baseline) * 100.0;
  } else {
    // For latency metrics (lower is better)
    // Degradation is positive when latency increases
    return ((current - baseline) / baseline) * 100.0;
  }
}

RegressionSeverity RegressionDetector::classify_severity(
    double degradation_pct) const {
  if (degradation_pct < 0.0) {
    // Performance improved, not a regression
    return RegressionSeverity::NONE;
  }

  // Check thresholds from highest to lowest
  if (degradation_pct >= thresholds_.at(RegressionSeverity::CRITICAL)) {
    return RegressionSeverity::CRITICAL;
  }
  if (degradation_pct >= thresholds_.at(RegressionSeverity::MAJOR)) {
    return RegressionSeverity::MAJOR;
  }
  if (degradation_pct >= thresholds_.at(RegressionSeverity::MINOR)) {
    return RegressionSeverity::MINOR;
  }
  if (degradation_pct >= thresholds_.at(RegressionSeverity::WARNING)) {
    return RegressionSeverity::WARNING;
  }

  return RegressionSeverity::NONE;
}

std::string RegressionDetector::make_key(const std::string& probe_name,
                                        uint16_t gpu_id) {
  return probe_name + ":" + std::to_string(gpu_id);
}

}  // namespace gpim
