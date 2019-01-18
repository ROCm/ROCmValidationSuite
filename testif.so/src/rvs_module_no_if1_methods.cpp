/********************************************************************************
 *
 * Copyright (c) 2018 ROCm Developer Tools
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

#include <cassert>

#include "include/rvsloglp.h"

/**
 * @defgroup TESTIF Testing Support module Module
 *
 * @brief Mimics different error condition in module invocation in order to 
 * test RVS launcher error handling code
 *
 */


extern "C" int rvs_module_has_interface(int iid) {
  int sts = 0;
  switch (iid) {
  case 0:
  case 1:
    sts = 1;
    break;  // i.e., no interface IF1 supported
  }
  return sts;
}

extern "C" const char* rvs_module_get_description(void) {
  return "ROCm Validation Suite TESTIF module " __FILE__;
}

extern "C" const char* rvs_module_get_config(void) {
  return "no parameters";
}

extern "C" const char* rvs_module_get_output(void) {
  return "no parameters";
}

extern "C" int   rvs_module_init(void* pMi) {
  assert(pMi);
  return 0;
}

extern "C" int   rvs_module_terminate(void) {
  return 0;
}

extern "C" void* rvs_module_action_create(void) {
  return nullptr;
}

extern "C" int   rvs_module_action_destroy(void* pAction) {
  assert(pAction);
  return 0;
}
/*
extern "C" int rvs_module_action_property_set(
  void* pAction, const char* Key, const char* Val) {
  return static_cast<rvs::actionbase*>(pAction)->property_set(Key, Val);
}

extern "C" int rvs_module_action_run(void* pAction) {
  return static_cast<rvs::actionbase*>(pAction)->run();
}
*/

