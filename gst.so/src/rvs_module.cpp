/********************************************************************************
 *
 * Copyright (c) 2018-2022 Advanced Micro Devices, Inc. All rights reserved.
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
 * @defgroup GST GST Module
 *
 * @brief performs GPU Stress Test
 *
 * The GPU Stress Test runs a Graphics Stress test or SGEMM/DGEMM
 * (Single/Double-precision General Matrix Multiplication) workload
 * on one, some or all GPUs. The GPUs can be of the same or different types.
 * The duration of the benchmark should be configurable, both in terms of time
 * (how long to run) and iterations (how many times to run).
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
  return "The GPU Stress Test runs a Graphics Stress test or SGEMM/DGEMM/HGEMM (Single/Double/Half-precision General Matrix Multiplication) \n\tworkload on one, some or all GPUs.";
}

extern "C" const char* rvs_module_get_config(void) {
  return "target_stress (float), copy_matrix (bool), "\
    "ramp_interval (int), tolerance (float), "\
    "max_violations (int), \n\tlog_interval (int), "\
    "matrix_size (int)";
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
  gst_action::cleanup_logs();
  return 0;
}

extern "C" void* rvs_module_action_create(void) {
  return static_cast<void*>(new gst_action);
}

extern "C" int   rvs_module_action_destroy(void* pAction) {
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
