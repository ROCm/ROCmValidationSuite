/********************************************************************************
 *
 * Copyright (c) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
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

extern "C" {
#include <pci/pci.h>
#include <linux/pci.h>
}

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <algorithm>
#include <iomanip>

#include "include/rvs_key_def.h"
#include "include/rvs_module.h"
#include "include/worker.h"
#include "include/pci_caps.h"
#include "include/gpu_util.h"
#include "include/rvs_util.h"
#include "include/rvsloglp.h"
#define RVS_CONF_DBGWAIT_KEY "debugwait"

static constexpr auto MODULE_NAME = "pesm";
static constexpr auto MODULE_NAME_CAPS = "PESM";
using std::string;
using std::cout;
using std::endl;
using std::hex;


extern Worker* pworker;

//! Default constructor
pesm_action::pesm_action() {
  bjson = false;
  prop_monitor = true;
  module_name = MODULE_NAME;
}

//! Default destructor
pesm_action::~pesm_action() {
  property.clear();
}

/**
 * @brief reads all common configuration keys from
 * the module's properties collection
 * @return true if no fatal error occured, false otherwise
 */
bool pesm_action::get_all_common_config_keys(void) {
    string msg;

    bool sts = true;

    if (property_get(RVS_CONF_NAME_KEY, &action_name)) {
      rvs::lp::Err("Action name missing", MODULE_NAME_CAPS);
      return false;
    }

    // check if  -j flag is passed
    if (has_property("cli.-j")) {
      bjson = true;
    }

    // get <device> property value (a list of gpu id)
    if (int ists = property_get_device()) {
      switch (ists) {
      case 1:
        msg = "Invalid 'device' key value.";
        break;
      case 2:
        msg = "Missing 'device' key.";
        break;
      }
      rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
      sts = false;
    }

    // get the <deviceid> property value if provided
    if (property_get_int<uint16_t>(RVS_CONF_DEVICEID_KEY,
                                  &property_device_id, 0u)) {
      msg = "Invalid 'deviceid' key value.";
      rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
      sts = false;
    }

    // get <device_index> property value (a list of device indexes)
    if (int sts = property_get_device_index()) {
      switch (sts) {
        case 1:
          msg = "Invalid 'device_index' key value.";
          break;
        case 2:
          msg = "Missing 'device_index' key.";
          break;
      }
      // default set as true
      property_device_index_all = true;
      rvs::lp::Log(msg, rvs::loginfo);
    }

    return sts;
}

/**
 * @brief reads all PESM specific configuration keys from
 * the module's properties collection
 * @return true if no fatal error occured, false otherwise
 */
bool pesm_action::get_all_pesm_config_keys(void) {
    string msg;

    bool sts = true;

    // get the <monitor> property value if provided
    if (property_get<bool>(RVS_CONF_MONITOR_KEY, &prop_monitor, true)) {
      msg = "Invalid '" RVS_CONF_MONITOR_KEY "' key value.";
      rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
      sts = false;
    }

    // get the <debugwait> property value if provided
    if (property_get_int<int>(RVS_CONF_DBGWAIT_KEY, &prop_debugwait, 0)) {
      msg = "Invalid '" RVS_CONF_DBGWAIT_KEY "' key value.";
      rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
      sts = false;
    }

    return sts;
}


/**
 * @brief Implements action functionality
 *
 * Functionality:
 *
 * - If "do_gpu_list" property is set,
 *   it lists all AMD GPUs present in the system and exits
 * - If "monitor" property is set to "true",
 *   it creates Worker thread and initiates monitoring and exits
 * - If "monitor" property is not set or is not set to "true",
 *   it stops the Worker thread and exits
 *
 * @return 0 - success. non-zero otherwise
 *
 * */
int pesm_action::run(void) {
  string msg;
  RVSTRACE_

  // this module implements --listGpu command line option
  // if this option is set, an internal input key 'do_gpu_list' is passed
  // to this action
  if (has_property("do_gpu_list")) {
    return do_gpu_list();
  }

  // get commong configuration keys
  if (!get_all_common_config_keys()) {
    return 1;
  }

  // get PESM specific configuration keys
  if (!get_all_pesm_config_keys()) {
    return 1;
  }

  // debugging help
  if (prop_debugwait) {
    sleep(prop_debugwait);
  }
  if (bjson){
    json_add_primary_fields(std::string(MODULE_NAME), action_name);
  }
  // end of monitoring requested?
  if (!prop_monitor) {
    RVSTRACE_
    if (pworker) {
      RVSTRACE_
      // (give thread chance to start)
      sleep(2);
      pworker->set_stop_name(action_name);
      pworker->stop();
      delete pworker;
      pworker = nullptr;
    }
    if (bjson){
      rvs::lp::JsonActionEndNodeCreate();
     }
    RVSTRACE_
    return 0;
  }

  RVSTRACE_
  if (pworker) {
    rvs::lp::Log("[" + property["name"]+ "] pesm monitoring already started",
                rvs::logdebug);
    if (bjson){
      rvs::lp::JsonActionEndNodeCreate();
     }

    return 0;
  }

  RVSTRACE_
  // create worker thread
  pworker = new Worker();
  pworker->set_name(action_name);
  pworker->set_action(*this);
  pworker->json(bjson);
  pworker->set_gpuids(property_device);
  pworker->set_gpuidx(property_device_index);
  pworker->set_deviceid(property_device_id);

  // start worker thread
  RVSTRACE_
  pworker->start();
  sleep(2);
  if (bjson){
    rvs::lp::JsonActionEndNodeCreate();
  }
  RVSTRACE_
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
int pesm_action::do_gpu_list() {
  return display_gpu_info(get_gpu_info());
}

