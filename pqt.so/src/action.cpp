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
#include "action.h"

#include "rvs_module.h"
#include "worker.h"

extern "C"
{
#include <pci/pci.h>
#include <linux/pci.h>
}
#include <iostream>
#include <algorithm>

#include "pci_caps.h"
#include "gpu_util.h"
#include "rvs_module.h"
#include "rvsloglp.h"


using namespace std;

static Worker* pworker;

//! Default constructor
action::action() {
}

//! Default destructor
action::~action() {
  property.clear();
}

/**
 * @brief Implements action functionality
 *
 * Functionality:
 *
 * - If "do_gpu_list" property is set, it lists all AMD GPUs present in the system and exits
 * - If "monitor" property is set to "true", it creates Worker thread and initiates monitoring and exits
 * - If "monitor" property is not set or is not set to "true", it stops the Worker thread and exits
 *
 * @return 0 - success. non-zero otherwise
 *
 * */
int action::run(void) {
  log("[PQT] in run()", rvs::logdebug);

  return 0;
}

/**
 * @brief Lists AMD GPUs
 *
 * Functionality:
 *
 * Lists all AMD GPUs present in the system.
 *
 * @return 0 - success. non-zero otherwise
 *
 * */
int action::do_gpu_list() {
  log("[PQT] in do_gpu_list()", rvs::logdebug);
  
  return 0;
}
