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

/**
 * @defgroup RCQT RCQT Module
 *
 * @brief ROCm Configuration Qualification Tools
 *
 * This module implements set of features that target and qualify the
 * configuration of the platform. Many of the checks can be done manually using
 * the operating systems command line tools and general knowledge about ROCm’s
 * requirements. The purpose of the RCQT modules is to provide an extensible, OS
 * independent and scriptable interface capable for performing the configuration
 * checks required for ROCm support.
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
  return "The ROCm Configuration Qualification Tool ensures the platform is capable of running ROCm applications and is configured correctly.";
}

extern "C" const char* rvs_module_get_config(void) {
  return "package (string), version (string), installed (bool), user (string), "
  "groups (collection of strings), file (string),\n\towner (string), "
  "group (string), permission (int), type (int), exists (bool)";
}

extern "C" const char* rvs_module_get_output(void) {
  return "pass (bool)";
}

extern "C" int   rvs_module_init(void* pMi) {
  rvs::lp::Initialize(static_cast<T_MODULE_INIT*>(pMi));
  return 0;
}

extern "C" int   rvs_module_terminate(void) {
  return 0;
}

extern "C" void* rvs_module_action_create(void) {
  return static_cast<void*>(new rcqt_action);
}

extern "C" int   rvs_module_action_destroy(void* pAction) {
  delete static_cast<rvs::actionbase*>(pAction);
  return 0;
}

extern "C" int rvs_module_action_property_set\
(void* pAction, const char* Key, const char* Val) {
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
