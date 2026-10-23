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
#ifndef GPIM_SO_INCLUDE_DIAGNOSTIC_RULE_REGISTRY_H_
#define GPIM_SO_INCLUDE_DIAGNOSTIC_RULE_REGISTRY_H_

#include <string>
#include <vector>
#include <map>
#include <functional>

#include "include/probes/probe_base.h"
#include "include/regression_detector.h"

namespace gpim {

/**
 * @enum DiagnosticCategory
 * @brief Category of diagnostic rule
 */
enum class DiagnosticCategory {
  HARDWARE,         // Hardware issues (thermal, power, memory)
  SOFTWARE,         // Software issues (driver, runtime, libraries)
  CONFIGURATION,    // Configuration issues (NUMA, PCIe, BIOS)
  WORKLOAD         // Workload-specific issues (batch size, precision)
};

/**
 * @struct DiagnosticFinding
 * @brief Enhanced diagnostic finding with remediation steps
 */
struct DiagnosticFinding {
  std::string rule_name;                        //!< Rule that generated this finding
  DiagnosticCategory category;                  //!< Category of the issue
  std::string description;                      //!< What was detected
  std::string evidence;                         //!< Supporting data
  std::vector<std::string> supporting_evidence; //!< Additional evidence from multiple sources
  std::string recommendation;                   //!< Suggested fix
  std::vector<std::string> remediation_steps;   //!< Step-by-step remediation
  int confidence;                               //!< Confidence score (0-100)
  std::string severity;                         //!< informational, low, medium, high, critical
  std::vector<std::string> related_findings;    //!< Related diagnostic findings

  DiagnosticFinding() : category(DiagnosticCategory::HARDWARE),
                       confidence(0) {}

  /**
   * @brief Get category as string
   */
  std::string category_to_string() const {
    switch (category) {
      case DiagnosticCategory::HARDWARE: return "Hardware";
      case DiagnosticCategory::SOFTWARE: return "Software";
      case DiagnosticCategory::CONFIGURATION: return "Configuration";
      case DiagnosticCategory::WORKLOAD: return "Workload";
    }
    return "Unknown";
  }
};

/**
 * @struct DiagnosticRule
 * @brief Definition of a diagnostic rule
 */
struct DiagnosticRule {
  std::string name;                             //!< Rule name
  DiagnosticCategory category;                  //!< Category
  int priority;                                 //!< Execution priority (1=highest)
  bool enabled;                                 //!< Is rule enabled?
  std::vector<std::string> required_probes;     //!< Probes needed for this rule
  std::vector<std::string> dependencies;        //!< Other rules that must run first

  // Condition function: returns true if rule should trigger
  std::function<bool(const ProbeResult&, const RegressionReport&)> condition;

  // Diagnostic function: generates finding if condition is true
  std::function<DiagnosticFinding(const ProbeResult&, const RegressionReport&)> diagnose;

  DiagnosticRule() : category(DiagnosticCategory::HARDWARE),
                    priority(5), enabled(true) {}
};

/**
 * @class DiagnosticRuleRegistry
 * @brief Registry of all diagnostic rules with 20+ comprehensive rules
 *
 * Categories:
 * - Hardware: 8 rules (thermal, power, memory, PCIe, NUMA, cache, ECC, clock)
 * - Software: 6 rules (driver, ROCm, runtime, libraries, firmware, CPU overhead)
 * - Configuration: 4 rules (NUMA, BIOS, power policy, scheduler)
 * - Workload: 4 rules (batch size, precision, kernel interference, memory pressure)
 */
class DiagnosticRuleRegistry {
 public:
  DiagnosticRuleRegistry();
  ~DiagnosticRuleRegistry();

  /**
   * @brief Register all default diagnostic rules (20+ rules)
   */
  void register_default_rules();

  /**
   * @brief Register a custom diagnostic rule
   */
  void register_rule(const DiagnosticRule& rule);

  /**
   * @brief Get all enabled rules, sorted by priority
   */
  std::vector<DiagnosticRule> get_enabled_rules() const;

  /**
   * @brief Enable/disable a specific rule
   */
  void set_rule_enabled(const std::string& rule_name, bool enabled);

  /**
   * @brief Get rule by name
   */
  DiagnosticRule* get_rule(const std::string& rule_name);

  /**
   * @brief List all rules
   */
  std::vector<std::string> list_rules() const;

  /**
   * @brief Get rules by category
   */
  std::vector<DiagnosticRule> get_rules_by_category(DiagnosticCategory category) const;

 private:
  std::map<std::string, DiagnosticRule> rules_;  //!< All registered rules

  // Hardware rules
  void register_thermal_throttling_rule();
  void register_power_limiting_rule();
  void register_memory_errors_rule();
  void register_pcie_degradation_rule();
  void register_numa_misconfiguration_rule();
  void register_cache_thrashing_rule();
  void register_ecc_errors_rule();
  void register_clock_instability_rule();

  // Software rules
  void register_driver_regression_rule();
  void register_rocm_version_mismatch_rule();
  void register_runtime_overhead_rule();
  void register_library_version_mismatch_rule();
  void register_firmware_bug_rule();
  void register_cpu_bottleneck_rule();

  // Configuration rules
  void register_numa_node_misconfiguration_rule();
  void register_bios_settings_rule();
  void register_power_policy_rule();
  void register_scheduler_inefficiency_rule();

  // Workload rules
  void register_batch_size_sensitivity_rule();
  void register_mixed_precision_issue_rule();
  void register_kernel_interference_rule();
  void register_memory_pressure_rule();
};

}  // namespace gpim

#endif  // GPIM_SO_INCLUDE_DIAGNOSTIC_RULE_REGISTRY_H_
