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
#ifndef GPIM_SO_INCLUDE_REGRESSION_DETECTOR_H_
#define GPIM_SO_INCLUDE_REGRESSION_DETECTOR_H_

#include <string>
#include <vector>
#include <map>

#include "include/probes/probe_base.h"
#include "include/baseline_manager.h"

namespace gpim {

/**
 * @enum RegressionSeverity
 * @brief Severity levels for performance regressions
 */
enum class RegressionSeverity {
  NONE,         //!< No regression detected
  WARNING,      //!< Minor degradation (5-10%)
  MINOR,        //!< Moderate degradation (10-20%)
  MAJOR,        //!< Significant degradation (20-50%)
  CRITICAL      //!< Severe degradation (>50%)
};

/**
 * @brief Convert severity enum to string
 */
inline std::string severity_to_string(RegressionSeverity severity) {
  switch (severity) {
    case RegressionSeverity::NONE:     return "NONE";
    case RegressionSeverity::WARNING:  return "WARNING";
    case RegressionSeverity::MINOR:    return "MINOR";
    case RegressionSeverity::MAJOR:    return "MAJOR";
    case RegressionSeverity::CRITICAL: return "CRITICAL";
    default:                           return "UNKNOWN";
  }
}

/**
 * @struct RegressionReport
 * @brief Detailed report of a detected performance regression
 */
struct RegressionReport {
  std::string probe_name;           //!< Probe that detected regression
  uint16_t gpu_id;                  //!< Affected GPU
  double baseline_value;            //!< Expected baseline value
  double current_value;             //!< Measured current value
  double degradation_pct;           //!< Degradation percentage (positive = worse)
  RegressionSeverity severity;      //!< Severity classification
  uint64_t timestamp;               //!< When regression was detected
  std::vector<std::string> possible_causes;  //!< Diagnostic hints

  RegressionReport() : gpu_id(0), baseline_value(0.0), current_value(0.0),
                      degradation_pct(0.0), severity(RegressionSeverity::NONE),
                      timestamp(0) {}
};

/**
 * @class RegressionDetector
 * @brief Detects performance regressions by comparing against baselines
 *
 * Supports configurable thresholds for different severity levels and
 * consecutive failure tracking to avoid false positives from transient issues.
 */
class RegressionDetector {
 public:
  /**
   * @brief Constructor
   * @param baseline_mgr Baseline manager instance
   */
  explicit RegressionDetector(BaselineManager* baseline_mgr);
  ~RegressionDetector();

  /**
   * @brief Set threshold for a specific severity level
   * @param severity Severity level
   * @param threshold_pct Degradation percentage (e.g., 10.0 for 10%)
   */
  void set_threshold(RegressionSeverity severity, double threshold_pct);

  /**
   * @brief Set number of consecutive failures required before reporting
   * @param count Number of consecutive failures (default: 1)
   */
  void set_consecutive_failures_required(int count);

  /**
   * @brief Detect regression in a probe result
   * @param result Probe result to analyze
   * @return RegressionReport (severity will be NONE if no regression)
   */
  RegressionReport detect(const ProbeResult& result);

  /**
   * @brief Batch detection for multiple results
   * @param results Vector of probe results
   * @return Vector of regression reports (only regressions, not NONE)
   */
  std::vector<RegressionReport> detect_batch(
      const std::vector<ProbeResult>& results);

  /**
   * @brief Reset consecutive failure counters
   */
  void reset_failure_counters();

  /**
   * @brief Get default threshold configuration (for ML workloads)
   */
  static std::map<RegressionSeverity, double> get_default_thresholds();

 private:
  BaselineManager* baseline_mgr_;   //!< Reference to baseline manager

  // Threshold configuration (severity -> degradation %)
  std::map<RegressionSeverity, double> thresholds_;

  // Consecutive failure tracking
  int consecutive_failures_required_;
  std::map<std::string, int> failure_counts_;  // key: probe_name:gpu_id

  /**
   * @brief Calculate degradation percentage
   * @param baseline Expected baseline value
   * @param current Current measured value
   * @param higher_is_better If true, higher values are better (e.g., throughput)
   * @return Degradation percentage (positive = performance degraded)
   */
  static double calculate_degradation(double baseline, double current,
                                      bool higher_is_better);

  /**
   * @brief Classify degradation severity based on thresholds
   */
  RegressionSeverity classify_severity(double degradation_pct) const;

  /**
   * @brief Generate failure tracking key
   */
  static std::string make_key(const std::string& probe_name, uint16_t gpu_id);
};

}  // namespace gpim

#endif  // GPIM_SO_INCLUDE_REGRESSION_DETECTOR_H_
