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
#define MODULE_NAME_CAPS "PESM"
#define RVS_CONF_DBGWAIT_KEY "debugwait"

using std::string;
using std::cout;
using std::endl;
using std::hex;


extern Worker* pworker;

//! Default constructor
pesm_action::pesm_action() {
  bjson = false;
  prop_monitor = true;
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

    // get the <deviceid> property value if provided
    if (property_get<bool>(RVS_CONF_MONITOR_KEY, &prop_monitor, true)) {
      msg = "Invalid '" RVS_CONF_MONITOR_KEY "' key value.";
      rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
      sts = false;
    }

    // get the <deviceid> property value if provided
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
    RVSTRACE_
    return 0;
  }

  RVSTRACE_
  if (pworker) {
    rvs::lp::Log("[" + property["name"]+ "] pesm monitoring already started",
                rvs::logdebug);
    return 0;
  }

  RVSTRACE_
  // create worker thread
  pworker = new Worker();
  pworker->set_name(action_name);
  pworker->json(bjson);
  pworker->set_gpuids(property_device);
  pworker->set_deviceid(property_device_id);

  // start worker thread
  RVSTRACE_
  pworker->start();
  sleep(2);

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
  rvs::lp::Log("pesm in do_gpu_list()", rvs::logtrace);

  std::map<std::string, std::string>::iterator it;

  struct device_info {
    std::string bus;
    std::string name;
    int32_t node_id;
    int32_t gpu_id;
    int32_t device_id;
  };

  std::vector<struct device_info> gpu_info_list;

  struct pci_access* pacc;
  struct pci_dev*    dev;
  char buff[1024];
  char devname[1024];

  // get the pci_access structure
  pacc = pci_alloc();
  // initialize the PCI library
  pci_init(pacc);
  // get the list of devices
  pci_scan_bus(pacc);

  int  ix = 0;
  // iterate over devices
  for (dev = pacc->devices; dev; dev = dev->next) {
    // fil in the info
    pci_fill_info(dev, PCI_FILL_IDENT | PCI_FILL_BASES | PCI_FILL_CLASS
    | PCI_FILL_EXT_CAPS | PCI_FILL_CAPS | PCI_FILL_PHYS_SLOT);

    // computes the actual dev's location_id (sysfs entry)
    uint16_t dev_location_id =
      ((((uint16_t)(dev->bus)) << 8) | (dev->dev));

    // if not AMD GPU just continue
    uint16_t node_id;
    if (rvs::gpulist::location2node(dev_location_id, &node_id)) {
      continue;
    }

    uint16_t gpu_id;
    if (rvs::gpulist::location2gpu(dev_location_id, &gpu_id)) {
      continue;
    }

    snprintf(buff, sizeof(buff), "%02X:%02X.%d", dev->bus, dev->dev, dev->func);

    string name;
    name = pci_lookup_name(pacc, devname, sizeof(devname), PCI_LOOKUP_DEVICE,
                           dev->vendor_id, dev->device_id);

    struct device_info info;
    info.bus       = buff;
    info.name      = name;
    info.node_id   = node_id;
    info.gpu_id    = gpu_id;
    info.device_id = dev->device_id;
    gpu_info_list.push_back(info);

    ++ix;
  }

  std::sort(gpu_info_list.begin(), gpu_info_list.end(),
           [](const struct device_info& a, const struct device_info& b) {
             return a.node_id < b.node_id; });

  if (!gpu_info_list.empty()) {
    cout << "Supported GPUs available:\n";
    for (const auto& info : gpu_info_list) {
      cout << info.bus  << " - GPU[" << std::setw(2) << info.node_id
      << " - " << std::setw(5) << info.gpu_id << "] " << info.name
      << " (Device " << info.device_id << ")\n";
    }
  } else {
    cout << endl << "No supported GPUs available.\n";
  }

  pci_cleanup(pacc);

  return 0;
}
