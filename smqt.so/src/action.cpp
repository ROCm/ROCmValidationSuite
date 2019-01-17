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

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <string>
#ifdef __cplusplus
extern "C" {
  #endif
  #include <pci/pci.h>
  #include <linux/pci.h>
  #ifdef __cplusplus
}
#endif

#include "include/rvs_key_def.h"
#include "include/rvs_module.h"
#include "include/pci_caps.h"
#include "include/gpu_util.h"
#include "include/rvsloglp.h"
#define MODULE_NAME "SMQT"

using std::string;
using std::vector;
using std::cerr;
using std::cout;
using std::endl;


// config
ulong bar1_req_size, bar1_base_addr_min, bar1_base_addr_max;
ulong bar2_req_size, bar2_base_addr_min, bar2_base_addr_max;
ulong bar4_req_size, bar4_base_addr_min, bar4_base_addr_max, bar5_req_size;
bool keysts = true;
// Prints to the provided buffer a nice number of bytes (KB, MB, GB, etc)
string smqt_action::pretty_print(ulong bytes, uint16_t gpu_id,
                            string action_name, string bar_name) {
  std::string suffix[5] = { " B", " KB", " MB", " GB", " TB"};
  std::stringstream ss;

  uint s = 0;  // which suffix to use
  double count = bytes;
  while (count >= 1024 && s < 5) {
    s++;
    count /= 1024;
  }
  ss << "[" << action_name << "]  smqt " << gpu_id << " " <<
  bar_name << "      "
  << bytes << " (" << std::fixed << std::setprecision(2) <<
  count << suffix[s] << ")";

  return ss.str();
}

smqt_action::smqt_action() {
}

smqt_action::~smqt_action() {
  property.clear();
}

/**
 * @brief reads all common configuration keys from
 * the module's properties collection
 * @return true if no fatal error occured, false otherwise
 */
bool smqt_action::get_all_common_config_keys() {
  string msg, sdevid, sdev;


  // get the action name
  if (property_get(RVS_CONF_NAME_KEY, &action_name)) {
    rvs::lp::Err("Action name missing", MODULE_NAME);
    keysts = false;
  }

  // get <device> property value (a list of gpu id)
  if (int sts = property_get_device()) {
    switch (sts) {
    case 1:
      msg = "Invalid 'device' key value.";
      break;
    case 2:
      msg = "Missing 'device' key.";
      break;
    }
    rvs::lp::Err(msg, MODULE_NAME, action_name);
    keysts = false;
  }

  // get the <deviceid> property value if provided
  if (property_get_int<uint16_t>(RVS_CONF_DEVICEID_KEY,
                                &property_device_id, 0u) != 0) {
    msg = "Invalid 'deviceid' key value.";
    rvs::lp::Err(msg, MODULE_NAME, action_name);
    keysts = false;
  }


  return keysts;
}

#define SMQT_FETCH_AND_CHECK(bar) \
err = property_get_int<ulong>(#bar, & bar); \
switch (err) { \
  case 1: msg = "Invalid #bar key"; \
    rvs::lp::Err(msg, MODULE_NAME, action_name); \
    return false; \
  case 2: msg = "Missing #bar key"; \
    rvs::lp::Err(msg, MODULE_NAME, action_name); \
    return false; \
}

bool smqt_action::get_all_smqt_config_keys() {
  int err = 0;
  std::string msg;

  SMQT_FETCH_AND_CHECK(bar1_req_size)
  SMQT_FETCH_AND_CHECK(bar2_req_size)
  SMQT_FETCH_AND_CHECK(bar4_req_size)
  SMQT_FETCH_AND_CHECK(bar5_req_size)
  SMQT_FETCH_AND_CHECK(bar1_base_addr_min)
  SMQT_FETCH_AND_CHECK(bar2_base_addr_min)
  SMQT_FETCH_AND_CHECK(bar4_base_addr_min)
  SMQT_FETCH_AND_CHECK(bar1_base_addr_max)
  SMQT_FETCH_AND_CHECK(bar2_base_addr_max)
  SMQT_FETCH_AND_CHECK(bar4_base_addr_max)

  return true;
}
/**
 * @brief Implements action functionality
 * Check if the sizes and addresses of BARs match the given ones
 * @return 0 - success, non-zero otherwise
 * */ 

int smqt_action::run(void) {
  bool global_pass = true;
  string msg;
  struct pci_access *pacc;
  bool devid_found = false;

  if (!get_all_common_config_keys()) {
    msg = "Couldn't fetch common config keys from the configuration file!";
    rvs::lp::Err(msg, MODULE_NAME, action_name);
    return -1;
  }

  if (!get_all_smqt_config_keys()) {
    msg = "Couldn't fetch bar config keys from the configuration file!";
    rvs::lp::Err(msg, MODULE_NAME, action_name);
    return -1;
  }

  // get the pci_access structure
  pacc = pci_alloc();
  // initialize the PCI library
  pci_init(pacc);
  // get the list of devices
  pci_scan_bus(pacc);

  struct pci_dev *dev;
  dev = pacc->devices;

  // iterate over devices
  for (dev = pacc->devices; dev; dev = dev->next) {
    bool pass = true;
    // fil in the info
    pci_fill_info(dev, PCI_FILL_IDENT | PCI_FILL_BASES \
    | PCI_FILL_CLASS | PCI_FILL_EXT_CAPS | PCI_FILL_CAPS | PCI_FILL_PHYS_SLOT);

    // computes the actual dev's location_id (sysfs entry)
    uint16_t dev_location_id = ((((uint16_t)(dev->bus)) << 8) | (dev->func));

    uint16_t gpu_id;
    // if not and AMD GPU just continue
    if (rvs::gpulist::location2gpu(dev_location_id, &gpu_id))
      continue;

#ifdef  RVS_UNIT_TEST
    on_set_device_gpu_id();
#endif

    // filter by device id if needed
    if (property_device_id > 0) {
      rvs::gpulist::gpu2device(gpu_id, &dev_id);
      if (property_device_id != dev_id) {
        continue;
        keysts = false;
      }
    }

    devid_found = true;

    // filter by list of devices if needed
    if (!property_device_all) {
      if (property_device.end() ==
          std::find(property_device.begin(), property_device.end(), gpu_id))
        continue;
    }

    // get actual values
    bar1_base_addr = dev->base_addr[0];
    bar1_size = dev->size[0];
    bar2_base_addr = dev->base_addr[2];
    bar2_size = dev->size[2];
    bar4_base_addr = dev->base_addr[5];
    bar4_size = dev->size[5];
    bar5_size = dev->rom_size;

#ifdef  RVS_UNIT_TEST
    on_bar_data_read();
#endif

    // check if values are as expected
    if (bar1_base_addr < bar1_base_addr_min ||
        bar1_base_addr > bar1_base_addr_max)
      pass = false;
    if (bar2_base_addr < bar2_base_addr_min ||
        bar2_base_addr > bar2_base_addr_max)
      pass = false;
    if (bar4_base_addr < bar4_base_addr_min ||
        bar4_base_addr > bar4_base_addr_max)
      pass = false;

    if (bar1_req_size > bar1_size ||
        bar2_req_size < bar2_size ||
        bar4_req_size < bar4_size ||
        bar5_req_size < bar5_size)
      pass = false;

    // loginfo
    unsigned int sec;
    unsigned int usec;
    rvs::lp::get_ticks(&sec, &usec);
    string msgs1, msgs2, msgs4, msgs5, msga1, msga2, msga4, pmsg, str, pass_str;
    char hex_value[30];

    if (pass)
      pass_str = "true";
    else
      pass_str = "false";

    // formating bar1 size for print
    msgs1 = pretty_print(bar1_size, gpu_id, action_name, "bar1_size");

    // formating bar2 size for print
    msgs2 = pretty_print(bar2_size, gpu_id, action_name, "bar2_size");

    // formating bar4 size for print
    msgs4 = pretty_print(bar4_size, gpu_id, action_name, "bar4_size");

    // formating bar5 size for print
    msgs5 = pretty_print(bar5_size, gpu_id, action_name, "bar5_size");

    snprintf(hex_value, sizeof(hex_value), "%lX", bar1_base_addr);
    msga1 = "[" + action_name + "] " + " smqt " + std::to_string(gpu_id) +
    " bar1_base_addr " + hex_value;
    snprintf(hex_value, sizeof(hex_value), "%lX", bar2_base_addr);
    msga2 = "[" + action_name + "] " + " smqt " + std::to_string(gpu_id) +
    " bar2_base_addr " + hex_value;
    snprintf(hex_value, sizeof(hex_value), "%lX", bar4_base_addr);
    msga4 = "[" + action_name + "] " + " smqt " + std::to_string(gpu_id) +
    " bar4_base_addr " + hex_value;
    pmsg = "[" + action_name + "] " + " smqt "  + std::to_string(gpu_id) +
    " " +pass_str;

    void* r = rvs::lp::LogRecordCreate("SMQT", action_name.c_str(),
                                       rvs::loginfo, sec, usec);

    void* res = rvs::lp::LogRecordCreate("SMQT", action_name.c_str(),
                                       rvs::logresults, sec, usec);

    rvs::lp::Log(msgs1, rvs::loginfo, sec, usec);
    rvs::lp::Log(msga1, rvs::loginfo, sec, usec);
    rvs::lp::Log(msgs2, rvs::loginfo, sec, usec);
    rvs::lp::Log(msga2, rvs::loginfo, sec, usec);
    rvs::lp::Log(msgs4, rvs::loginfo, sec, usec);
    rvs::lp::Log(msga4, rvs::loginfo, sec, usec);
    rvs::lp::Log(msgs5, rvs::loginfo, sec, usec);
    rvs::lp::Log(pmsg, rvs::logresults);
    rvs::lp::AddInt(r, "gpu", gpu_id);
    rvs::lp::AddString(r, "bar1_size", std::to_string(bar1_size));
    rvs::lp::AddString(r, "bar1_base_addr", std::to_string(bar1_base_addr));
    rvs::lp::AddString(r, "bar2_size", std::to_string(bar2_size));
    rvs::lp::AddString(r, "bar2_base_addr", std::to_string(bar2_base_addr));
    rvs::lp::AddString(r, "bar4_size", std::to_string(bar4_size));
    rvs::lp::AddString(r, "bar4_base_addr", std::to_string(bar4_base_addr));
    rvs::lp::AddString(r, "bar5_size", std::to_string(bar4_size));
    rvs::lp::AddString(res, "pass", std::to_string(pass));
    rvs::lp::LogRecordFlush(r);
    rvs::lp::LogRecordFlush(res);
    if (!pass)
      global_pass = false;
  }
  if (!devid_found) {
    global_pass = false;
    msg = "No devices match criteria from the test configuation.";
    rvs::lp::Err(msg, MODULE_NAME, action_name);
    return -1;
  }
  return global_pass ? 0 : -1;
}
