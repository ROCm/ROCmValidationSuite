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
#include "include/rvs_module.h"
#include "include/action.h"
#include "include/rvsloglp.h"
#include "include/gpu_util.h"

/**
 * @defgroup PULSE PULSE Module
 *
 * @brief GPU Power Pulse Stress Test
 *
 * The Pulse Test fluctuates GPU power usage by alternating between
 * high-compute and idle phases at configurable rates, creating rapid
 * current spikes that stress the power supply and voltage regulators.
 * Work across GPUs is synchronized via a two-level barrier (CPU + GPU
 * fine-grained atomics) to create maximum aggregate current transients.
 *
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
  return "GPU Power Pulse Stress Test - creates rapid power fluctuations "
         "to stress the GPU power supply and voltage regulators.";
}

extern "C" const char* rvs_module_get_config(void) {
  return "pulse_rate (int), high_phase_ratio (float), "
    "matrix_size (int), ops_type (string), "
    "tolerance (float), sample_interval (int), "
    "workload_iterations (int), halt_on_error (bool), "
    "gpu_sync_wait (int), verify_mode (string)";
}

extern "C" const char* rvs_module_get_output(void) {
  return "pass (bool)";
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
  return static_cast<void*>(new pulse_action);
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
    void * user_param) {
  return static_cast<rvs::actionbase*>(pAction)->callback_set(callback, user_param);
}

extern "C" int rvs_module_action_run(void* pAction) {
  return static_cast<rvs::actionbase*>(pAction)->run();
}
