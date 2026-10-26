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
#include "include/rvsloglp.h"
#include "include/gpu_util.h"

/**
 * @defgroup GPIM GPIM Module
 *
 * @brief GPU Performance Integrity Monitor
 *
 * The GPU Performance Integrity Monitor (GPIM) periodically
 * monitors GPU performance through lightweight microbenchmark probes.
 * It compares results against established baselines to detect performance
 * regressions and provides diagnostic insights into root causes.
 *
 * Key features:
 * - Lightweight performance probes (kernel launch latency, GEMM, memory
 *   bandwidth, runtime overhead)
 * - Per-GPU baseline creation and storage
 * - Regression detection with configurable thresholds
 * - Rule-based diagnostics using telemetry captured with each probe
 *
 * Operational modes:
 * - baseline: Create performance baselines from multiple runs
 * - single: One-time validation against stored baselines
 */

extern "C" int rvs_module_has_interface(int iid) {
  int sts = 0;
  switch (iid) {
  case 0:
  case 1:
    sts = 1;
  }
  return sts;
}

extern "C" const char* rvs_module_get_description(void) {
  return "ROCm Validation Suite GPU Performance Integrity Monitor (GPIM) module";
}

extern "C" const char* rvs_module_get_config(void) {
  return "mode (string: baseline|single), "\
    "baseline_path (string), "\
    "baseline_runs (int), "\
    "baseline_auto_create (bool), "\
    "baseline_record_hostname (bool), "\
    "probe_kernel_launch_enabled (bool), "\
    "probe_gemm_enabled (bool), "\
    "probe_memory_bandwidth_enabled (bool), "\
    "probe_rocm_version_enabled (bool), "\
    "probe_driver_overhead_enabled (bool), "\
    "regression_detection_enabled (bool), "\
    "regression_threshold_warning (float), "\
    "regression_threshold_minor (float), "\
    "regression_threshold_major (float), "\
    "regression_threshold_critical (float), "\
    "regression_consecutive_failures (int), "\
    "diagnostics_enabled (bool), "\
    "fail_on_regression (bool)";
}

extern "C" const char* rvs_module_get_output(void) {
  return "pass (bool), regressions_detected (int), "\
    "probe_results (array), diagnostic_findings (array)";
}

extern "C" int rvs_module_init(void* pMi) {
  rvs::lp::Initialize(static_cast<T_MODULE_INIT*>(pMi));
  rvs::gpulist::Initialize();
  return 0;
}

extern "C" int rvs_module_terminate(void) {
  amdsmi_shut_down();
  return 0;
}

extern "C" void* rvs_module_action_create(void) {
  return static_cast<void*>(new gpim_action);
}

extern "C" int rvs_module_action_destroy(void* pAction) {
  delete static_cast<rvs::actionbase*>(pAction);
  return 0;
}

extern "C" int rvs_module_action_property_set(void* pAction, const char* Key,
                                               const char* Val) {
  return static_cast<rvs::actionbase*>(pAction)->property_set(Key, Val);
}

extern "C" int rvs_module_action_callback_set(void* pAction,
                                               rvs::callback_t callback,
                                               void* user_param) {
  return static_cast<rvs::actionbase*>(pAction)->callback_set(callback,
                                                               user_param);
}

extern "C" int rvs_module_action_run(void* pAction) {
  return static_cast<rvs::actionbase*>(pAction)->run();
}
