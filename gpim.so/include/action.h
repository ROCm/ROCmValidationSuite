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
#ifndef GPIM_SO_INCLUDE_ACTION_H_
#define GPIM_SO_INCLUDE_ACTION_H_

#include <vector>
#include <string>
#include <map>
#include <memory>

#include "include/rvsactionbase.h"

// Forward declarations
namespace gpim {
class BaselineManager;
class RegressionDetector;
class DiagnosticEngine;
}

/**
 * @class gpim_action
 * @ingroup GPIM
 *
 * @brief GPU Performance Integrity Monitor action implementation
 *
 * Monitors GPU performance through lightweight microbenchmark probes,
 * compares against baselines, detects regressions, and provides diagnostic
 * insights for ML training and inference workloads.
 *
 * Operational modes:
 * - baseline: Create performance baselines
 * - single: One-time validation run against stored baselines
 */
class gpim_action : public rvs::actionbase {
 public:
  gpim_action();
  virtual ~gpim_action();

  /**
   * @brief Main action execution entry point
   * @return 0 if successful, non-zero otherwise
   */
  virtual int run(void);

 protected:
  /**
   * @brief Parse and validate configuration properties
   * @return 0 if successful, non-zero otherwise
   */
  int property_get_all();

  /**
   * @brief Execute baseline creation mode
   * @return 0 if successful, non-zero otherwise
   */
  int run_baseline_mode();

  /**
   * @brief Execute single validation run mode
   * @return 0 if successful, non-zero otherwise
   */
  int run_single_mode();

  /**
   * @brief Log summary report with actionable recommendations
   */
  void log_summary_report();

  /**
   * @brief Output results in JSON format
   */
  void output_json_report();

 protected:
  // Operational mode
  std::string mode_;  // "baseline", "single"

  // GPU selection
  std::vector<uint16_t> gpu_ids_;
  std::map<int, uint16_t> gpim_gpus_device_index_;  // HIP device index -> RVS GPU ID

  // Probe configuration
  std::map<std::string, bool> probes_enabled_;
  std::map<std::string, uint64_t> probe_intervals_;  // milliseconds
  std::map<std::string, std::map<std::string, std::string>> probe_configs_;

  // Baseline configuration
  bool baseline_auto_create_;
  bool baseline_record_hostname_;
  std::string baseline_path_;
  std::string baseline_update_policy_;  // "never", "manual", "auto"

  // Regression detection
  bool regression_detection_enabled_;
  std::map<std::string, double> regression_thresholds_;  // severity -> threshold %
  int consecutive_failures_required_;

  // Diagnostics
  bool diagnostics_enabled_;
  std::vector<std::string> diagnostic_rules_;

  // Reporting
  uint64_t report_log_interval_;  // milliseconds
  std::string report_path_;
  bool fail_on_regression_;

  // Alert actions
  std::string alert_script_;

  // Core components
  std::unique_ptr<gpim::BaselineManager> baseline_manager_;
  std::unique_ptr<gpim::RegressionDetector> regression_detector_;
  std::unique_ptr<gpim::DiagnosticEngine> diagnostic_engine_;

  // Results tracking
  int total_probes_executed_;
  int total_regressions_detected_;
  std::map<std::string, int> severity_counts_;  // severity -> count
};

#endif  // GPIM_SO_INCLUDE_ACTION_H_
