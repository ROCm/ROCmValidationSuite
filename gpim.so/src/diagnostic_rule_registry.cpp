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

#include "include/diagnostic_rule_registry.h"
#include <algorithm>
#include <sstream>

namespace gpim {

DiagnosticRuleRegistry::DiagnosticRuleRegistry() {
  register_default_rules();
}

DiagnosticRuleRegistry::~DiagnosticRuleRegistry() {
}

void DiagnosticRuleRegistry::register_default_rules() {
  // Hardware rules (8)
  register_thermal_throttling_rule();
  register_power_limiting_rule();
  register_memory_errors_rule();
  register_pcie_degradation_rule();
  register_numa_misconfiguration_rule();
  register_cache_thrashing_rule();
  register_ecc_errors_rule();
  register_clock_instability_rule();

  // Software rules (6)
  register_driver_regression_rule();
  register_rocm_version_mismatch_rule();
  register_runtime_overhead_rule();
  register_library_version_mismatch_rule();
  register_firmware_bug_rule();
  register_cpu_bottleneck_rule();

  // Configuration rules (4)
  register_numa_node_misconfiguration_rule();
  register_bios_settings_rule();
  register_power_policy_rule();
  register_scheduler_inefficiency_rule();

  // Workload rules (4)
  register_batch_size_sensitivity_rule();
  register_mixed_precision_issue_rule();
  register_kernel_interference_rule();
  register_memory_pressure_rule();
}

void DiagnosticRuleRegistry::register_rule(const DiagnosticRule& rule) {
  rules_[rule.name] = rule;
}

std::vector<DiagnosticRule> DiagnosticRuleRegistry::get_enabled_rules() const {
  std::vector<DiagnosticRule> enabled;

  for (const auto& pair : rules_) {
    if (pair.second.enabled) {
      enabled.push_back(pair.second);
    }
  }

  // Sort by priority (lower number = higher priority)
  std::sort(enabled.begin(), enabled.end(),
            [](const DiagnosticRule& a, const DiagnosticRule& b) {
              return a.priority < b.priority;
            });

  return enabled;
}

void DiagnosticRuleRegistry::set_rule_enabled(const std::string& rule_name,
                                             bool enabled) {
  auto it = rules_.find(rule_name);
  if (it != rules_.end()) {
    it->second.enabled = enabled;
  }
}

DiagnosticRule* DiagnosticRuleRegistry::get_rule(const std::string& rule_name) {
  auto it = rules_.find(rule_name);
  if (it != rules_.end()) {
    return &it->second;
  }
  return nullptr;
}

std::vector<std::string> DiagnosticRuleRegistry::list_rules() const {
  std::vector<std::string> names;
  for (const auto& pair : rules_) {
    names.push_back(pair.first);
  }
  return names;
}

std::vector<DiagnosticRule> DiagnosticRuleRegistry::get_rules_by_category(
    DiagnosticCategory category) const {
  std::vector<DiagnosticRule> filtered;

  for (const auto& pair : rules_) {
    if (pair.second.category == category) {
      filtered.push_back(pair.second);
    }
  }

  return filtered;
}

// ==================== HARDWARE RULES ====================

void DiagnosticRuleRegistry::register_thermal_throttling_rule() {
  DiagnosticRule rule;
  rule.name = "thermal_throttling";
  rule.category = DiagnosticCategory::HARDWARE;
  rule.priority = 1;  // Highest priority
  rule.enabled = true;

  rule.condition = [](const ProbeResult& result, const RegressionReport& regression) {
    // Trigger if temp > 85C and performance degraded
    return result.telemetry.valid &&
           result.telemetry.temp_junction_mc > 85000 &&  // 85C in millidegrees
           regression.degradation_pct > 10.0;
  };

  rule.diagnose = [](const ProbeResult& result, const RegressionReport& regression) {
    DiagnosticFinding finding;
    finding.rule_name = "thermal_throttling";
    finding.category = DiagnosticCategory::HARDWARE;
    finding.severity = "high";
    finding.confidence = 90;

    std::ostringstream desc;
    desc << "Thermal throttling detected: GPU temperature "
         << (result.telemetry.temp_junction_mc / 1000.0) << "C exceeds safe threshold";
    finding.description = desc.str();

    std::ostringstream evidence;
    evidence << "Temperature: " << (result.telemetry.temp_junction_mc / 1000.0) << "C, "
             << "Performance degradation: " << regression.degradation_pct << "%";
    finding.evidence = evidence.str();

    finding.recommendation = "Improve cooling or reduce GPU workload";
    finding.remediation_steps.push_back("Check datacenter cooling system");
    finding.remediation_steps.push_back("Verify GPU fan operation");
    finding.remediation_steps.push_back("Clean dust from heatsinks");
    finding.remediation_steps.push_back("Reduce ambient temperature");

    return finding;
  };

  register_rule(rule);
}

void DiagnosticRuleRegistry::register_power_limiting_rule() {
  DiagnosticRule rule;
  rule.name = "power_limiting";
  rule.category = DiagnosticCategory::HARDWARE;
  rule.priority = 1;
  rule.enabled = true;

  rule.condition = [](const ProbeResult& result, const RegressionReport& regression) {
    // Trigger if power near max and performance degraded
    // MI300X has ~750W TDP
    uint64_t power_limit_uw = 750000000;  // 750W in microwatts
    return result.telemetry.valid &&
           result.telemetry.power_usage_uw > (power_limit_uw * 0.95) &&
           regression.degradation_pct > 10.0;
  };

  rule.diagnose = [](const ProbeResult& result, const RegressionReport& regression) {
    DiagnosticFinding finding;
    finding.rule_name = "power_limiting";
    finding.category = DiagnosticCategory::HARDWARE;
    finding.severity = "medium";
    finding.confidence = 85;

    finding.description = "Power limiting detected: GPU hitting power cap";

    std::ostringstream evidence;
    evidence << "Power usage: " << (result.telemetry.power_usage_uw / 1000000.0) << "W";
    finding.evidence = evidence.str();

    finding.recommendation = "Increase power limit or optimize workload";
    finding.remediation_steps.push_back("Check PSU capacity");
    finding.remediation_steps.push_back("Adjust power limit with rocm-smi");
    finding.remediation_steps.push_back("Optimize kernel efficiency");

    return finding;
  };

  register_rule(rule);
}

void DiagnosticRuleRegistry::register_memory_errors_rule() {
  DiagnosticRule rule;
  rule.name = "memory_errors";
  rule.category = DiagnosticCategory::HARDWARE;
  rule.priority = 1;
  rule.enabled = true;
  rule.required_probes.push_back("ecc_error_rate");

  rule.condition = [](const ProbeResult& result, const RegressionReport& regression) {
    // Check if ECC error rate probe detected errors
    return result.probe_name == "ecc_error_rate" && result.value > 0.0;
  };

  rule.diagnose = [](const ProbeResult& result, const RegressionReport& regression) {
    DiagnosticFinding finding;
    finding.rule_name = "memory_errors";
    finding.category = DiagnosticCategory::HARDWARE;
    finding.severity = "critical";
    finding.confidence = 95;

    finding.description = "Memory errors detected via ECC";

    std::ostringstream evidence;
    evidence << "ECC error rate: " << result.value << " errors/hour";
    finding.evidence = evidence.str();

    finding.recommendation = "RMA GPU or reseat memory modules";
    finding.remediation_steps.push_back("Check dmesg for amdgpu errors");
    finding.remediation_steps.push_back("Run memory stress test");
    finding.remediation_steps.push_back("Contact hardware support");

    return finding;
  };

  register_rule(rule);
}

void DiagnosticRuleRegistry::register_pcie_degradation_rule() {
  DiagnosticRule rule;
  rule.name = "pcie_degradation";
  rule.category = DiagnosticCategory::HARDWARE;
  rule.priority = 2;
  rule.enabled = true;
  rule.required_probes.push_back("pcie_bandwidth");

  rule.condition = [](const ProbeResult& result, const RegressionReport& regression) {
    return result.probe_name == "pcie_bandwidth" &&
           regression.severity >= RegressionSeverity::MAJOR;
  };

  rule.diagnose = [](const ProbeResult& result, const RegressionReport& regression) {
    DiagnosticFinding finding;
    finding.rule_name = "pcie_degradation";
    finding.category = DiagnosticCategory::HARDWARE;
    finding.severity = "medium";
    finding.confidence = 80;

    finding.description = "PCIe bandwidth degradation detected";
    finding.evidence = "PCIe transfer rate below expected baseline";
    finding.recommendation = "Check PCIe link status and configuration";
    finding.remediation_steps.push_back("Verify PCIe Gen4/5 link active");
    finding.remediation_steps.push_back("Check for PCIe errors in dmesg");
    finding.remediation_steps.push_back("Reseat GPU in PCIe slot");

    return finding;
  };

  register_rule(rule);
}

void DiagnosticRuleRegistry::register_cache_thrashing_rule() {
  DiagnosticRule rule;
  rule.name = "cache_thrashing";
  rule.category = DiagnosticCategory::HARDWARE;
  rule.priority = 3;
  rule.enabled = true;
  rule.required_probes.push_back("cache_performance");

  rule.condition = [](const ProbeResult& result, const RegressionReport& regression) {
    return result.probe_name == "cache_performance" &&
           regression.degradation_pct > 20.0;
  };

  rule.diagnose = [](const ProbeResult& result, const RegressionReport& regression) {
    DiagnosticFinding finding;
    finding.rule_name = "cache_thrashing";
    finding.category = DiagnosticCategory::HARDWARE;
    finding.severity = "low";
    finding.confidence = 70;

    finding.description = "Cache thrashing detected: poor cache locality";
    finding.recommendation = "Optimize memory access patterns in workload";
    finding.remediation_steps.push_back("Review kernel memory access patterns");
    finding.remediation_steps.push_back("Improve data locality");
    finding.remediation_steps.push_back("Consider tiling optimizations");

    return finding;
  };

  register_rule(rule);
}

void DiagnosticRuleRegistry::register_ecc_errors_rule() {
  register_memory_errors_rule();  // Alias to memory_errors_rule
}

void DiagnosticRuleRegistry::register_clock_instability_rule() {
  DiagnosticRule rule;
  rule.name = "clock_instability";
  rule.category = DiagnosticCategory::HARDWARE;
  rule.priority = 2;
  rule.enabled = true;
  rule.required_probes.push_back("clock_stability");

  rule.condition = [](const ProbeResult& result, const RegressionReport& regression) {
    return result.probe_name == "clock_stability" &&
           result.secondary_metrics.count("jitter") > 0 &&
           result.secondary_metrics.at("jitter") > 50.0;  // > 50 MHz jitter
  };

  rule.diagnose = [](const ProbeResult& result, const RegressionReport& regression) {
    DiagnosticFinding finding;
    finding.rule_name = "clock_instability";
    finding.category = DiagnosticCategory::HARDWARE;
    finding.severity = "medium";
    finding.confidence = 75;

    finding.description = "GPU clock frequency unstable (high jitter)";
    finding.recommendation = "Check power delivery and thermal conditions";

    return finding;
  };

  register_rule(rule);
}

void DiagnosticRuleRegistry::register_numa_misconfiguration_rule() {
  DiagnosticRule rule;
  rule.name = "numa_misconfiguration";
  rule.category = DiagnosticCategory::CONFIGURATION;
  rule.priority = 2;
  rule.enabled = true;

  rule.condition = [](const ProbeResult& result, const RegressionReport& regression) {
    // Check memory bandwidth probe for NUMA issues
    return result.probe_name == "memory_bandwidth" &&
           regression.degradation_pct > 15.0;
  };

  rule.diagnose = [](const ProbeResult& result, const RegressionReport& regression) {
    DiagnosticFinding finding;
    finding.rule_name = "numa_misconfiguration";
    finding.category = DiagnosticCategory::CONFIGURATION;
    finding.severity = "medium";
    finding.confidence = 65;

    finding.description = "Possible NUMA misconfiguration affecting memory bandwidth";
    finding.recommendation = "Verify NUMA node configuration and CPU affinity";
    finding.remediation_steps.push_back("Check numactl settings");
    finding.remediation_steps.push_back("Bind process to correct NUMA node");
    finding.remediation_steps.push_back("Review PCIe-to-CPU mapping");

    return finding;
  };

  register_rule(rule);
}

// ==================== SOFTWARE RULES ====================

void DiagnosticRuleRegistry::register_driver_regression_rule() {
  DiagnosticRule rule;
  rule.name = "driver_regression";
  rule.category = DiagnosticCategory::SOFTWARE;
  rule.priority = 1;
  rule.enabled = true;

  rule.condition = [](const ProbeResult& result, const RegressionReport& regression) {
    // Check driver overhead probe
    return result.probe_name == "driver_overhead" &&
           regression.degradation_pct > 20.0;
  };

  rule.diagnose = [](const ProbeResult& result, const RegressionReport& regression) {
    DiagnosticFinding finding;
    finding.rule_name = "driver_regression";
    finding.category = DiagnosticCategory::SOFTWARE;
    finding.severity = "high";
    finding.confidence = 85;

    finding.description = "Driver overhead increased significantly";
    finding.recommendation = "Check for driver updates or rollback";
    finding.remediation_steps.push_back("Check amdgpu driver version");
    finding.remediation_steps.push_back("Review recent driver updates");
    finding.remediation_steps.push_back("Consider driver rollback");

    return finding;
  };

  register_rule(rule);
}

void DiagnosticRuleRegistry::register_rocm_version_mismatch_rule() {
  DiagnosticRule rule;
  rule.name = "rocm_version_mismatch";
  rule.category = DiagnosticCategory::SOFTWARE;
  rule.priority = 2;
  rule.enabled = true;

  rule.condition = [](const ProbeResult& result, const RegressionReport& regression) {
    // This would require metadata comparison
    return false;  // Placeholder
  };

  rule.diagnose = [](const ProbeResult& result, const RegressionReport& regression) {
    DiagnosticFinding finding;
    finding.rule_name = "rocm_version_mismatch";
    finding.category = DiagnosticCategory::SOFTWARE;
    finding.severity = "medium";
    finding.confidence = 80;

    finding.description = "ROCm version mismatch between baseline and current";
    finding.recommendation = "Use matching ROCm version or create new baseline";

    return finding;
  };

  register_rule(rule);
}

void DiagnosticRuleRegistry::register_runtime_overhead_rule() {
  DiagnosticRule rule;
  rule.name = "runtime_overhead";
  rule.category = DiagnosticCategory::SOFTWARE;
  rule.priority = 3;
  rule.enabled = true;

  rule.condition = [](const ProbeResult& result, const RegressionReport& regression) {
    return result.probe_name == "kernel_launch" &&
           regression.degradation_pct > 15.0;
  };

  rule.diagnose = [](const ProbeResult& result, const RegressionReport& regression) {
    DiagnosticFinding finding;
    finding.rule_name = "runtime_overhead";
    finding.category = DiagnosticCategory::SOFTWARE;
    finding.severity = "low";
    finding.confidence = 60;

    finding.description = "Kernel launch overhead increased";
    finding.recommendation = "Check HIP runtime and system load";

    return finding;
  };

  register_rule(rule);
}

void DiagnosticRuleRegistry::register_library_version_mismatch_rule() {
  DiagnosticRule rule;
  rule.name = "library_version_mismatch";
  rule.category = DiagnosticCategory::SOFTWARE;
  rule.priority = 3;
  rule.enabled = true;

  rule.condition = [](const ProbeResult& result, const RegressionReport& regression) {
    return result.probe_name == "gemm" &&
           regression.degradation_pct > 10.0;
  };

  rule.diagnose = [](const ProbeResult& result, const RegressionReport& regression) {
    DiagnosticFinding finding;
    finding.rule_name = "library_version_mismatch";
    finding.category = DiagnosticCategory::SOFTWARE;
    finding.severity = "medium";
    finding.confidence = 70;

    finding.description = "GEMM performance regression may indicate rocBLAS version issue";
    finding.recommendation = "Verify rocBLAS version matches baseline";

    return finding;
  };

  register_rule(rule);
}

void DiagnosticRuleRegistry::register_firmware_bug_rule() {
  DiagnosticRule rule;
  rule.name = "firmware_bug";
  rule.category = DiagnosticCategory::SOFTWARE;
  rule.priority = 2;
  rule.enabled = false;  // Disabled by default (rare)

  rule.condition = [](const ProbeResult& result, const RegressionReport& regression) {
    return false;  // Placeholder - would check firmware-specific metrics
  };

  rule.diagnose = [](const ProbeResult& result, const RegressionReport& regression) {
    DiagnosticFinding finding;
    finding.rule_name = "firmware_bug";
    finding.category = DiagnosticCategory::SOFTWARE;
    finding.severity = "high";
    finding.confidence = 50;

    finding.description = "Possible firmware bug affecting performance";
    finding.recommendation = "Check for firmware updates";

    return finding;
  };

  register_rule(rule);
}

void DiagnosticRuleRegistry::register_cpu_bottleneck_rule() {
  DiagnosticRule rule;
  rule.name = "cpu_bottleneck";
  rule.category = DiagnosticCategory::SOFTWARE;
  rule.priority = 3;
  rule.enabled = true;

  rule.condition = [](const ProbeResult& result, const RegressionReport& regression) {
    // Low GPU utilization + performance regression = CPU bottleneck
    return result.telemetry.valid &&
           result.telemetry.gpu_utilization_pct < 80 &&
           regression.degradation_pct > 10.0;
  };

  rule.diagnose = [](const ProbeResult& result, const RegressionReport& regression) {
    DiagnosticFinding finding;
    finding.rule_name = "cpu_bottleneck";
    finding.category = DiagnosticCategory::SOFTWARE;
    finding.severity = "medium";
    finding.confidence = 75;

    finding.description = "CPU bottleneck detected: low GPU utilization";

    std::ostringstream evidence;
    evidence << "GPU utilization: " << result.telemetry.gpu_utilization_pct << "%";
    finding.evidence = evidence.str();

    finding.recommendation = "Optimize CPU code or increase batch size";

    return finding;
  };

  register_rule(rule);
}

// ==================== CONFIGURATION RULES ====================

void DiagnosticRuleRegistry::register_numa_node_misconfiguration_rule() {
  register_numa_misconfiguration_rule();  // Same as hardware NUMA rule
}

void DiagnosticRuleRegistry::register_bios_settings_rule() {
  DiagnosticRule rule;
  rule.name = "bios_settings";
  rule.category = DiagnosticCategory::CONFIGURATION;
  rule.priority = 4;
  rule.enabled = false;  // Manual check

  rule.condition = [](const ProbeResult& result, const RegressionReport& regression) {
    return false;  // Placeholder
  };

  rule.diagnose = [](const ProbeResult& result, const RegressionReport& regression) {
    DiagnosticFinding finding;
    finding.rule_name = "bios_settings";
    finding.category = DiagnosticCategory::CONFIGURATION;
    finding.severity = "low";
    finding.confidence = 40;

    finding.description = "BIOS settings may not be optimal";
    finding.recommendation = "Review BIOS power and performance settings";

    return finding;
  };

  register_rule(rule);
}

void DiagnosticRuleRegistry::register_power_policy_rule() {
  DiagnosticRule rule;
  rule.name = "power_policy";
  rule.category = DiagnosticCategory::CONFIGURATION;
  rule.priority = 3;
  rule.enabled = true;

  rule.condition = [](const ProbeResult& result, const RegressionReport& regression) {
    return result.telemetry.valid &&
           result.telemetry.gpu_clock_mhz < 1500 &&  // Low clock
           regression.degradation_pct > 10.0;
  };

  rule.diagnose = [](const ProbeResult& result, const RegressionReport& regression) {
    DiagnosticFinding finding;
    finding.rule_name = "power_policy";
    finding.category = DiagnosticCategory::CONFIGURATION;
    finding.severity = "medium";
    finding.confidence = 70;

    finding.description = "GPU running at reduced clock frequency";
    finding.recommendation = "Check power policy settings";
    finding.remediation_steps.push_back("Set GPU to performance mode");
    finding.remediation_steps.push_back("rocm-smi --setperflevel high");

    return finding;
  };

  register_rule(rule);
}

void DiagnosticRuleRegistry::register_scheduler_inefficiency_rule() {
  DiagnosticRule rule;
  rule.name = "scheduler_inefficiency";
  rule.category = DiagnosticCategory::CONFIGURATION;
  rule.priority = 4;
  rule.enabled = false;

  rule.condition = [](const ProbeResult& result, const RegressionReport& regression) {
    return false;  // Placeholder
  };

  rule.diagnose = [](const ProbeResult& result, const RegressionReport& regression) {
    DiagnosticFinding finding;
    finding.rule_name = "scheduler_inefficiency";
    finding.category = DiagnosticCategory::CONFIGURATION;
    finding.severity = "low";
    finding.confidence = 50;

    finding.description = "OS scheduler may not be optimal for GPU workloads";
    finding.recommendation = "Check CPU governor settings";

    return finding;
  };

  register_rule(rule);
}

// ==================== WORKLOAD RULES ====================

void DiagnosticRuleRegistry::register_batch_size_sensitivity_rule() {
  DiagnosticRule rule;
  rule.name = "batch_size_sensitivity";
  rule.category = DiagnosticCategory::WORKLOAD;
  rule.priority = 3;
  rule.enabled = true;
  rule.required_probes.push_back("ml_inference");

  rule.condition = [](const ProbeResult& result, const RegressionReport& regression) {
    return result.probe_name == "ml_inference" &&
           regression.degradation_pct > 15.0;
  };

  rule.diagnose = [](const ProbeResult& result, const RegressionReport& regression) {
    DiagnosticFinding finding;
    finding.rule_name = "batch_size_sensitivity";
    finding.category = DiagnosticCategory::WORKLOAD;
    finding.severity = "low";
    finding.confidence = 60;

    finding.description = "ML inference performance sensitive to batch size";
    finding.recommendation = "Optimize batch size for GPU architecture";

    return finding;
  };

  register_rule(rule);
}

void DiagnosticRuleRegistry::register_mixed_precision_issue_rule() {
  DiagnosticRule rule;
  rule.name = "mixed_precision_issue";
  rule.category = DiagnosticCategory::WORKLOAD;
  rule.priority = 3;
  rule.enabled = true;

  rule.condition = [](const ProbeResult& result, const RegressionReport& regression) {
    return result.probe_name == "gemm" &&
           regression.degradation_pct > 20.0;
  };

  rule.diagnose = [](const ProbeResult& result, const RegressionReport& regression) {
    DiagnosticFinding finding;
    finding.rule_name = "mixed_precision_issue";
    finding.category = DiagnosticCategory::WORKLOAD;
    finding.severity = "low";
    finding.confidence = 55;

    finding.description = "GEMM performance may be affected by precision choice";
    finding.recommendation = "Consider using FP16/BF16 for better performance";

    return finding;
  };

  register_rule(rule);
}

void DiagnosticRuleRegistry::register_kernel_interference_rule() {
  DiagnosticRule rule;
  rule.name = "kernel_interference";
  rule.category = DiagnosticCategory::WORKLOAD;
  rule.priority = 3;
  rule.enabled = true;

  rule.condition = [](const ProbeResult& result, const RegressionReport& regression) {
    // High variance in results indicates interference
    return result.secondary_metrics.count("std_dev") > 0 &&
           result.secondary_metrics.at("std_dev") > (result.value * 0.15);
  };

  rule.diagnose = [](const ProbeResult& result, const RegressionReport& regression) {
    DiagnosticFinding finding;
    finding.rule_name = "kernel_interference";
    finding.category = DiagnosticCategory::WORKLOAD;
    finding.severity = "medium";
    finding.confidence = 65;

    finding.description = "High performance variance suggests kernel interference";
    finding.recommendation = "Check for concurrent GPU workloads";

    return finding;
  };

  register_rule(rule);
}

void DiagnosticRuleRegistry::register_memory_pressure_rule() {
  DiagnosticRule rule;
  rule.name = "memory_pressure";
  rule.category = DiagnosticCategory::WORKLOAD;
  rule.priority = 2;
  rule.enabled = true;

  rule.condition = [](const ProbeResult& result, const RegressionReport& regression) {
    return result.probe_name == "memory_bandwidth" &&
           regression.degradation_pct > 25.0;
  };

  rule.diagnose = [](const ProbeResult& result, const RegressionReport& regression) {
    DiagnosticFinding finding;
    finding.rule_name = "memory_pressure";
    finding.category = DiagnosticCategory::WORKLOAD;
    finding.severity = "medium";
    finding.confidence = 75;

    finding.description = "Memory bandwidth pressure detected";
    finding.recommendation = "Reduce memory-intensive operations or optimize memory access";
    finding.remediation_steps.push_back("Check GPU memory utilization");
    finding.remediation_steps.push_back("Optimize memory access patterns");
    finding.remediation_steps.push_back("Consider memory tiling");

    return finding;
  };

  register_rule(rule);
}

}  // namespace gpim
