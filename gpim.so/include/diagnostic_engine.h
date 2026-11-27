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
#ifndef GPIM_SO_INCLUDE_DIAGNOSTIC_ENGINE_H_
#define GPIM_SO_INCLUDE_DIAGNOSTIC_ENGINE_H_

#include <string>
#include <vector>
#include <functional>

#include "include/probes/probe_base.h"
#include "include/regression_detector.h"

namespace gpim {

/**
 * @struct DiagnosticFinding
 * @brief A single diagnostic finding from root cause analysis
 */
struct DiagnosticFinding {
  std::string rule_name;            //!< Diagnostic rule that fired
  std::string description;          //!< What was detected
  std::string evidence;             //!< Supporting data
  std::string recommendation;       //!< Actionable fix suggestion
  int confidence;                   //!< Confidence level (0-100)

  DiagnosticFinding() : confidence(0) {}
};

/**
 * @struct DiagnosticRule
 * @brief Rule for diagnosing performance regressions
 */
struct DiagnosticRule {
  std::string name;
  std::string description;

  // Condition function: returns true if rule applies
  std::function<bool(const ProbeResult&, const RegressionReport&)> condition;

  // Generate detailed finding
  std::function<DiagnosticFinding(const ProbeResult&, const RegressionReport&)> generate_finding;
};

/**
 * @class DiagnosticEngine
 * @brief ML-focused root cause analysis for performance regressions
 *
 * Analyzes regressions and provides actionable recommendations.
 * Optimized for ML training and inference workload patterns.
 *
 * Key diagnostic categories:
 * 1. Thermal issues (throttling, cooling failures)
 * 2. Power limiting (insufficient power budget, capping)
 * 3. Memory issues (ECC errors, bandwidth degradation, fragmentation)
 * 4. PCIe/interconnect (NUMA misconfig, link degradation)
 * 5. Runtime overhead (driver issues, system contention)
 * 6. ML-specific (mixed precision issues, batch size sensitivity)
 */
class DiagnosticEngine {
 public:
  DiagnosticEngine();
  ~DiagnosticEngine();

  /**
   * @brief Diagnose a performance regression
   * @param result Probe result that triggered regression
   * @param regression Regression report details
   * @return Vector of diagnostic findings (ranked by confidence)
   */
  std::vector<DiagnosticFinding> diagnose(
      const ProbeResult& result,
      const RegressionReport& regression);

  /**
   * @brief Enable/disable specific diagnostic rules
   * @param rule_name Name of rule to toggle
   * @param enabled True to enable, false to disable
   */
  void set_rule_enabled(const std::string& rule_name, bool enabled);

  /**
   * @brief Get list of available diagnostic rules
   */
  std::vector<std::string> list_rules() const;

 private:
  std::vector<DiagnosticRule> rules_;           //!< Registered rules
  std::map<std::string, bool> rules_enabled_;   //!< Rule enable/disable state

  /**
   * @brief Initialize default ML-focused diagnostic rules
   */
  void initialize_default_rules();

  // Individual diagnostic rule implementations

  /**
   * @brief Detect thermal throttling (high temp + reduced clocks)
   */
  void add_thermal_throttling_rule();

  /**
   * @brief Detect power limiting (power near cap + perf drop)
   */
  void add_power_limit_rule();

  /**
   * @brief Detect memory errors (ECC errors)
   */
  void add_memory_errors_rule();

  /**
   * @brief Detect PCIe/NUMA issues (bandwidth degradation)
   */
  void add_pcie_degradation_rule();

  /**
   * @brief Detect kernel launch overhead (runtime issues)
   */
  void add_runtime_overhead_rule();

  /**
   * @brief Detect clock frequency issues
   */
  void add_clock_frequency_rule();

  /**
   * @brief Detect memory bandwidth bottleneck
   */
  void add_memory_bandwidth_rule();

  /**
   * @brief ML-specific: detect compute vs memory bound shift
   */
  void add_ml_workload_shift_rule();

  /**
   * @brief Sort findings by confidence (highest first)
   */
  static void rank_findings(std::vector<DiagnosticFinding>& findings);
};

}  // namespace gpim

#endif  // GPIM_SO_INCLUDE_DIAGNOSTIC_ENGINE_H_
