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

#include "rvs_module.h"
#include "pci_caps.h"
#include "gpu_util.h"
#include "rvsloglp.h"

using std::string;
using std::vector;
using std::cerr;

// Prints to the provided buffer a nice number of bytes (KB, MB, GB, etc)
string action::pretty_print(ulong bytes, string action_name, string bar_name) {
  std::string suffix[5] = { " B", " KB", " MB", " GB", " TB"};
  std::stringstream ss;

  uint s = 0;  // which suffix to use
  double count = bytes;
  while (count >= 1024 && s < 5) {
    s++;
    count /= 1024;
  }
  ss << "[" << action_name << "]  smqt " << bar_name << "      "\
  << bytes << " (" << std::fixed << std::setprecision(2) <<
  count << suffix[s] << ")";

  return ss.str();
}

ulong action::get_property(string property) {
  string value;
  ulong result;

  if (has_property(property, &value)) {
    result = std::stoul(value);
  } else {
    std::cerr << "RVS-SMQT: Error fetching " << property
              << ". Cannot continue without it. Exiting!\n";
    exit(EXIT_FAILURE);
  }
  return result;
}

action::action() {
}

action::~action() {
  property.clear();
}

/**
 * @brief reads all common configuration keys from
 * the module's properties collection
 * @return true if no fatal error occured, false otherwise
 */
bool action::get_all_common_config_keys() {
  string msg, sdevid, sdev;
  int error;

  // get <device> property value (a list of gpu id)
  if (has_property("device", &sdev)) {
    property_get_device(&error);
    if (error) {  // log the error & abort GST
      cerr << "RVS-SMQT: action: " << action_name <<
            "  invalid 'device' key value " << sdev << '\n';
      return false;
    }
  } else {
    cerr << "RVS-SMQT: action: " << action_name <<
        "  key 'device' was not found" << std::endl;
    return false;
  }

  return true;
}

/**
 * @brief Implements action functionality
 * Check if the sizes and addresses of BARs match the given ones
 * @return 0 - success, non-zero otherwise
 * */ 

int action::run(void) {
  // config
  ulong bar1_req_size, bar1_base_addr_min, bar1_base_addr_max;
  ulong bar2_req_size, bar2_base_addr_min, bar2_base_addr_max;
  ulong bar4_req_size, bar4_base_addr_min, bar4_base_addr_max, bar5_req_size;
  // output
  ulong bar1_size, bar1_base_addr, bar2_size, bar2_base_addr;
  ulong bar4_size, bar4_base_addr, bar5_size;
  bool pass;
  int error = 0;
  string msg;
  struct pci_access *pacc;
  // get the action name
  rvs::actionbase::property_get_action_name(&error);
  if (error == 2) {
    msg = "action field is missing in smqt module";
    cerr << "RVS-SMQT: " << msg;
    return -1;
  }

  if (!get_all_common_config_keys()) {
    std::cerr << "RVS-SMQT: Couldn't fetch common config keys "
              << "from the configuration file!\n";
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

  if (!has_property("name", &action_name)) {
    std::cerr << "RVS-SMQT: Error fetching action_name\n";
    return false;
  }
  bar1_req_size      = get_property("bar1_req_size");
  bar2_req_size      = get_property("bar2_req_size");
  bar4_req_size      = get_property("bar4_req_size");
  bar5_req_size      = get_property("bar5_req_size");
  bar1_base_addr_min = get_property("bar1_base_addr_min");
  bar2_base_addr_min = get_property("bar2_base_addr_min");
  bar4_base_addr_min = get_property("bar4_base_addr_min");
  bar1_base_addr_max = get_property("bar1_base_addr_max");
  bar2_base_addr_max = get_property("bar2_base_addr_max");
  bar4_base_addr_max = get_property("bar4_base_addr_max");

  // iterate over devices
  for (dev = pacc->devices; dev; dev = dev->next) {
    // fil in the info
    pci_fill_info(dev, PCI_FILL_IDENT | PCI_FILL_BASES \
    | PCI_FILL_CLASS | PCI_FILL_EXT_CAPS | PCI_FILL_CAPS | PCI_FILL_PHYS_SLOT);

    // computes the actual dev's location_id (sysfs entry)
    uint16_t dev_location_id = ((((uint16_t)(dev->bus)) << 8) | (dev->func));

    int32_t gpu_id = rvs::gpulist::GetGpuId(dev_location_id);

    // if the device is not in the list, move on
    if (-1 == gpu_id)
      continue;

    // get actual values
    bar1_base_addr = dev->base_addr[0];
    bar1_size = dev->size[0];
    bar2_base_addr = dev->base_addr[2];
    bar2_size = dev->size[2];
    bar4_base_addr = dev->base_addr[5];
    bar4_size = dev->size[5];
    bar5_size = dev->rom_size;

    // check if values are as expected
    pass = true;

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
      pass_str = "pass";
    else
      pass_str = "fail";

    // formating bar1 size for print
    msgs1 = pretty_print(bar1_size, action_name, "bar1_size");

    // formating bar2 size for print
    msgs2 = pretty_print(bar2_size, action_name, "bar2_size");

    // formating bar4 size for print
    msgs4 = pretty_print(bar4_size, action_name, "bar4_size");

    // formating bar5 size for print
    msgs5 = pretty_print(bar5_size, action_name, "bar5_size");

    snprintf(hex_value, 0xFF, "%lX", bar1_base_addr);
    msga1 = "[" + action_name + "] " + " smqt bar1_base_addr " + hex_value;
    snprintf(hex_value, 0xFF, "%lX", bar2_base_addr);
    msga2 = "[" + action_name + "] " + " smqt bar2_base_addr " + hex_value;
    snprintf(hex_value, 0xFF, "%lX", bar4_base_addr);
    msga4 = "[" + action_name + "] " + " smqt bar4_base_addr " + hex_value;
    pmsg = "[" + action_name + "] " + " smqt " + pass_str;

    void* r = rvs::lp::LogRecordCreate("SMQT", action_name.c_str(),
                                       rvs::loginfo, sec, usec);

    rvs::lp::Log(msgs1, rvs::loginfo, sec, usec);
    rvs::lp::Log(msga1, rvs::loginfo, sec, usec);
    rvs::lp::Log(msgs2, rvs::loginfo, sec, usec);
    rvs::lp::Log(msga2, rvs::loginfo, sec, usec);
    rvs::lp::Log(msgs4, rvs::loginfo, sec, usec);
    rvs::lp::Log(msga4, rvs::loginfo, sec, usec);
    rvs::lp::Log(msgs5, rvs::loginfo, sec, usec);
    rvs::lp::Log(pmsg, rvs::logresults);
    rvs::lp::AddString(r, "bar1_size", std::to_string(bar1_size));
    rvs::lp::AddString(r, "bar1_base_addr", std::to_string(bar1_base_addr));
    rvs::lp::AddString(r, "bar2_size", std::to_string(bar2_size));
    rvs::lp::AddString(r, "bar2_base_addr", std::to_string(bar2_base_addr));
    rvs::lp::AddString(r, "bar4_size", std::to_string(bar4_size));
    rvs::lp::AddString(r, "bar4_base_addr", std::to_string(bar4_base_addr));
    rvs::lp::AddString(r, "bar5_size", std::to_string(bar4_size));
    rvs::lp::AddString(r, "pass", std::to_string(pass));
    rvs::lp::LogRecordFlush(r);
  }
  return pass;
}
