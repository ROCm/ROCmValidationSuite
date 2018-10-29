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

#include "rvs_key_def.h"
#include "rvs_module.h"
#include "worker.h"
#include "pci_caps.h"
#include "gpu_util.h"
#include "rvs_util.h"
#include "rvsloglp.h"
#define MODULE_NAME "PESM"

using std::string;
using std::cout;
using std::endl;
using std::hex;


extern Worker* pworker;

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
int action::run(void) {
  int error = 0;
  string msg;
  log("[PESM] in run()", rvs::logdebug);

  // get the action name
  rvs::actionbase::property_get_action_name(&error);
  if (error == 2) {
    msg = "action field is missing in gst module";
    rvs::lp::Err(msg, MODULE_NAME);
    return -1;
  }
  rvs::lp::Log("[" + property["name"]+ "] pesm in run()", rvs::logtrace);

  // debugging help
  string val;
  if (has_property("debugwait", &val)) {
    sleep(std::stoi(val));
  }
  // this module implements --listGpu command line option
  // if this option is set, an internal input key 'do_gpu_list' is passed
  // to this action
  if (has_property("do_gpu_list")) {
    return do_gpu_list();
  }

  // start of monitoring?
  if (property["monitor"] == "true") {
    if (pworker) {
      rvs::lp::Log("[" + property["name"]+ "] pesm monitoring already started",
                  rvs::logresults);
      return 0;
    }

    rvs::lp::Log("[" + property["name"]+
    "] pesm property[\"monitor\"] == \"true\"", rvs::logtrace);

    // create worker thread object
    rvs::lp::Log("[" + property["name"]+ "] pesm creating Worker",
                 rvs::logtrace);

    pworker = new Worker();
    pworker->set_name(property["name"]);

    // check if  -j flag is passed
    if (has_property("cli.-j")) {
      pworker->json(true);
    }

    // checki if deviceid filtering is required
    string sdevid;
    if (has_property("deviceid", &sdevid)) {
      if (::is_positive_integer(sdevid)) {
        try {
          pworker->set_deviceid(std::stoi(sdevid));
        }
        catch(...) {
          msg = property["name"] +
          "  invalide 'deviceid' key value: " + sdevid;
          rvs::lp::Err(msg, MODULE_NAME, action_name);
          return -1;
        }
      } else {
        msg = property["name"] +
        "  invalide 'deviceid' key value: " + sdevid;
        rvs::lp::Err(msg, MODULE_NAME, action_name);
        return -1;
      }
    }

    // check if GPU id filtering is requied
    string sdev;
    if (has_property("device", &sdev)) {
      pworker->set_strgpuids(sdev);
      if (sdev != "all") {
        vector<string> sarr = str_split(sdev, YAML_DEVICE_PROP_DELIMITER);
        vector<int> iarr;
        int sts = rvs_util_strarr_to_intarr(sarr, &iarr);
        if (sts < 0) {
          msg = property["name"] +
          "  invalide 'device' key value: " + sdev;
          rvs::lp::Err(msg, MODULE_NAME, action_name);
          return -1;
        }
        pworker->set_gpuids(iarr);
      }
    } else {
          msg = property["name"] +
          "  key 'device' not found";
          rvs::lp::Err(msg, MODULE_NAME, action_name);
          return -1;
    }

    // start worker thread
    rvs::lp::Log("[" + property["name"]+ "] pesm starting Worker",
                 rvs::logtrace);
    pworker->start();
    sleep(2);

    rvs::lp::Log("[" + property["name"]+ "] pesm Monitoring started",
                 rvs::logtrace);
  } else {
    rvs::lp::Log("[" + property["name"]+
    "] pesm property[\"monitor\"] != \"true\"", rvs::logtrace);
    if (pworker) {
      // (give thread chance to start)
      sleep(2);
      pworker->set_stop_name(property["name"]);
      pworker->stop();
      delete pworker;
      pworker = nullptr;
    }
    rvs::lp::Log("[" + property["name"]+ "] pesm Monitoring stopped",
                 rvs::logtrace);
  }
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
  log("pesm in do_gpu_list()", rvs::logtrace);

  std::map<string, string>::iterator it;

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
      ((((uint16_t)(dev->bus)) << 8) | (dev->func));

    // if not and AMD GPU just continue
    int32_t node_id = rvs::gpulist::GetNodeIdFromLocationId(dev_location_id);
    if (node_id < 0)
      continue;

    int32_t gpu_id = rvs::gpulist::GetGpuId(dev_location_id);
    if (gpu_id < 0)
      continue;

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
