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
#include "include/diagnostic_engine.h"
#include <algorithm>
#include <sstream>

namespace gpim {

DiagnosticEngine::DiagnosticEngine() {
  initialize_default_rules();
}

DiagnosticEngine::~DiagnosticEngine() {
}

std::vector<DiagnosticFinding> DiagnosticEngine::diagnose(
    const ProbeResult& result,
    const RegressionReport& regression) {
  std::vector<DiagnosticFinding> findings;

  // Apply each enabled rule
  for (const auto& rule : rules_) {
    auto it = rules_enabled_.find(rule.name);
    if (it != rules_enabled_.end() && !it->second) {
      continue;  // Rule disabled
    }

    // Check if rule condition applies
    if (rule.condition(result, regression)) {
      DiagnosticFinding finding = rule.generate_finding(result, regression);
      findings.push_back(finding);
    }
  }

  // Rank findings by confidence
  rank_findings(findings);

  return findings;
}

void DiagnosticEngine::set_rule_enabled(const std::string& rule_name,
                                        bool enabled) {
  rules_enabled_[rule_name] = enabled;
}

std::vector<std::string> DiagnosticEngine::list_rules() const {
  std::vector<std::string> names;
  for (const auto& rule : rules_) {
    names.push_back(rule.name);
  }
  return names;
}

void DiagnosticEngine::initialize_default_rules() {
  add_thermal_throttling_rule();
  add_power_limit_rule();
  add_memory_errors_rule();
  add_pcie_degradation_rule();
  add_runtime_overhead_rule();
  add_clock_frequency_rule();
  add_memory_bandwidth_rule();
  add_ml_workload_shift_rule();
}

void DiagnosticEngine::add_thermal_throttling_rule() {
  DiagnosticRule rule;
  rule.name = "thermal_throttling";
  rule.description = "GPU temperature exceeds threshold, causing clock reduction";

  rule.condition = [](const ProbeResult& result,
                     const RegressionReport& regression) {
    if (!result.telemetry.valid) return false;

    // Check for high temperature (>85C) and regression
    int64_t temp_c = result.telemetry.temp_junction_mc / 1000;
    return (temp_c > 85 && regression.degradation_pct > 10.0);
  };

  rule.generate_finding = [](const ProbeResult& result,
                            const RegressionReport& regression) {
    DiagnosticFinding finding;
    finding.rule_name = "thermal_throttling";
    finding.description = "Thermal throttling detected";
    finding.confidence = 85;

    std::ostringstream oss;
    oss << "Temperature: " << (result.telemetry.temp_junction_mc / 1000.0)
        << "C (high), Clock: " << result.telemetry.gpu_clock_mhz << " MHz";
    finding.evidence = oss.str();

    finding.recommendation =
        "Check cooling system:\n"
        "  - Verify fans are running: amd-smi monitor\n"
        "  - Check ambient temperature\n"
        "  - Clean dust from heatsinks\n"
        "  - Consider reducing power limit: amd-smi set --power-cap 250";

    return finding;
  };

  rules_.push_back(rule);
  rules_enabled_[rule.name] = true;
}

void DiagnosticEngine::add_power_limit_rule() {
  DiagnosticRule rule;
  rule.name = "power_limit";
  rule.description = "Power usage near cap, causing performance reduction";

  rule.condition = [](const ProbeResult& result,
                     const RegressionReport& regression) {
    if (!result.telemetry.valid) return false;
    if (result.telemetry.power_usage_uw == 0) return false;

    // Check if power usage is high and we have regression
    uint64_t power_w = result.telemetry.power_usage_uw / 1000000;
    return (power_w > 250 && regression.degradation_pct > 10.0);
  };

  rule.generate_finding = [](const ProbeResult& result,
                            const RegressionReport& regression) {
    DiagnosticFinding finding;
    finding.rule_name = "power_limit";
    finding.description = "Power limiting detected";
    finding.confidence = 80;

    std::ostringstream oss;
    oss << "Power usage: " << (result.telemetry.power_usage_uw / 1000000.0)
        << " W (near cap)";
    finding.evidence = oss.str();

    finding.recommendation =
        "Increase power limit or reduce workload:\n"
        "  - Check current cap: amd-smi static\n"
        "  - Increase limit: amd-smi set --power-cap 300\n"
        "  - Reduce batch size or workload intensity";

    return finding;
  };

  rules_.push_back(rule);
  rules_enabled_[rule.name] = true;
}

void DiagnosticEngine::add_memory_errors_rule() {
  DiagnosticRule rule;
  rule.name = "memory_errors";
  rule.description = "ECC errors detected, possible memory degradation";

  rule.condition = [](const ProbeResult& result,
                     const RegressionReport& regression) {
    // This would require ECC error tracking in telemetry
    // For now, check if memory bandwidth probe shows degradation
    return (result.probe_name.find("bandwidth") != std::string::npos &&
            regression.degradation_pct > 20.0);
  };

  rule.generate_finding = [](const ProbeResult& result,
                            const RegressionReport& regression) {
    DiagnosticFinding finding;
    finding.rule_name = "memory_errors";
    finding.description = "Memory bandwidth degradation detected";
    finding.confidence = 60;

    finding.evidence = "Significant bandwidth drop without other indicators";

    finding.recommendation =
        "Check memory health:\n"
        "  - Run memory test: rvs -c gst.conf\n"
        "  - Check for ECC errors: amd-smi metric --ecc\n"
        "  - Consider GPU replacement if errors persist";

    return finding;
  };

  rules_.push_back(rule);
  rules_enabled_[rule.name] = true;
}

void DiagnosticEngine::add_pcie_degradation_rule() {
  DiagnosticRule rule;
  rule.name = "pcie_degradation";
  rule.description = "PCIe bandwidth reduced, possible NUMA topology issue";

  rule.condition = [](const ProbeResult& result,
                     const RegressionReport& regression) {
    // Check for H2D transfer or any probe showing significant degradation
    return regression.degradation_pct > 20.0;
  };

  rule.generate_finding = [](const ProbeResult& result,
                            const RegressionReport& regression) {
    DiagnosticFinding finding;
    finding.rule_name = "pcie_degradation";
    finding.description = "Possible PCIe or NUMA issue";
    finding.confidence = 50;

    finding.evidence = "Significant performance degradation detected";

    finding.recommendation =
        "Check PCIe and NUMA configuration:\n"
        "  - Check PCIe link: lspci -vv | grep LnkSta\n"
        "  - Verify NUMA affinity: numactl --hardware\n"
        "  - Check GPU topology: rocm-smi --showtopo";

    return finding;
  };

  rules_.push_back(rule);
  rules_enabled_[rule.name] = true;
}

void DiagnosticEngine::add_runtime_overhead_rule() {
  DiagnosticRule rule;
  rule.name = "runtime_overhead";
  rule.description = "Kernel launch latency increased, possible driver/runtime issue";

  rule.condition = [](const ProbeResult& result,
                     const RegressionReport& regression) {
    return (result.probe_name == "kernel_launch_latency" &&
            regression.degradation_pct > 30.0);
  };

  rule.generate_finding = [](const ProbeResult& result,
                            const RegressionReport& regression) {
    DiagnosticFinding finding;
    finding.rule_name = "runtime_overhead";
    finding.description = "Runtime overhead detected";
    finding.confidence = 75;

    std::ostringstream oss;
    oss << "Launch latency increased by " << regression.degradation_pct
        << "% (" << regression.baseline_value << " -> "
        << regression.current_value << " us)";
    finding.evidence = oss.str();

    finding.recommendation =
        "Check for runtime or system issues:\n"
        "  - Check ROCm version: apt list --installed | grep rocm\n"
        "  - Check system load: top -b -n 1 | head -20\n"
        "  - Check CPU governor: cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor\n"
        "  - Set performance mode: echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor";

    return finding;
  };

  rules_.push_back(rule);
  rules_enabled_[rule.name] = true;
}

void DiagnosticEngine::add_clock_frequency_rule() {
  DiagnosticRule rule;
  rule.name = "clock_frequency";
  rule.description = "GPU clock frequency below expected";

  rule.condition = [](const ProbeResult& result,
                     const RegressionReport& regression) {
    if (!result.telemetry.valid) return false;

    // Check if clocks are low without thermal justification
    int64_t temp_c = result.telemetry.temp_junction_mc / 1000;
    uint64_t clock_mhz = result.telemetry.gpu_clock_mhz;

    return (clock_mhz < 1400 && temp_c < 75 && regression.degradation_pct > 10.0);
  };

  rule.generate_finding = [](const ProbeResult& result,
                            const RegressionReport& regression) {
    DiagnosticFinding finding;
    finding.rule_name = "clock_frequency";
    finding.description = "Low GPU clock frequency";
    finding.confidence = 70;

    std::ostringstream oss;
    oss << "Clock: " << result.telemetry.gpu_clock_mhz << " MHz (low), "
        << "Temp: " << (result.telemetry.temp_junction_mc / 1000.0) << "C (normal)";
    finding.evidence = oss.str();

    finding.recommendation =
        "Check GPU power state and settings:\n"
        "  - Check power profile: amd-smi metric --power\n"
        "  - Set performance mode: amd-smi set --perf-level high\n"
        "  - Check BIOS settings for PCIe power management";

    return finding;
  };

  rules_.push_back(rule);
  rules_enabled_[rule.name] = true;
}

void DiagnosticEngine::add_memory_bandwidth_rule() {
  DiagnosticRule rule;
  rule.name = "memory_bandwidth";
  rule.description = "Memory bandwidth bottleneck detected";

  rule.condition = [](const ProbeResult& result,
                     const RegressionReport& regression) {
    return (result.probe_name == "memory_bandwidth" &&
            regression.degradation_pct > 15.0);
  };

  rule.generate_finding = [](const ProbeResult& result,
                            const RegressionReport& regression) {
    DiagnosticFinding finding;
    finding.rule_name = "memory_bandwidth";
    finding.description = "Memory bandwidth degraded";
    finding.confidence = 65;

    std::ostringstream oss;
    oss << "Bandwidth: " << regression.current_value << " GB/s "
        << "(baseline: " << regression.baseline_value << " GB/s)";
    finding.evidence = oss.str();

    finding.recommendation =
        "Check memory configuration:\n"
        "  - Verify memory clocks: amd-smi metric --clock\n"
        "  - Check for memory errors: amd-smi metric --ecc\n"
        "  - Run extended memory test: rvs -c gst.conf";

    return finding;
  };

  rules_.push_back(rule);
  rules_enabled_[rule.name] = true;
}

void DiagnosticEngine::add_ml_workload_shift_rule() {
  DiagnosticRule rule;
  rule.name = "ml_workload_shift";
  rule.description = "ML workload characteristics changed (compute vs memory bound)";

  rule.condition = [](const ProbeResult& result,
                     const RegressionReport& regression) {
    return (result.probe_name == "gemm_throughput" &&
            regression.degradation_pct > 15.0);
  };

  rule.generate_finding = [](const ProbeResult& result,
                            const RegressionReport& regression) {
    DiagnosticFinding finding;
    finding.rule_name = "ml_workload_shift";
    finding.description = "GEMM throughput degraded";
    finding.confidence = 70;

    std::ostringstream oss;
    oss << "GEMM: " << regression.current_value << " TFLOPS "
        << "(baseline: " << regression.baseline_value << " TFLOPS)";
    finding.evidence = oss.str();

    finding.recommendation =
        "Check compute performance:\n"
        "  - Verify GPU utilization: amd-smi metric --utilization\n"
        "  - Check for thermal/power throttling\n"
        "  - Verify mixed precision settings (FP32/FP16/BF16)\n"
        "  - Check for competing workloads: rocm-smi --showpids";

    return finding;
  };

  rules_.push_back(rule);
  rules_enabled_[rule.name] = true;
}

void DiagnosticEngine::rank_findings(std::vector<DiagnosticFinding>& findings) {
  std::sort(findings.begin(), findings.end(),
           [](const DiagnosticFinding& a, const DiagnosticFinding& b) {
             return a.confidence > b.confidence;
           });
}

}  // namespace gpim
