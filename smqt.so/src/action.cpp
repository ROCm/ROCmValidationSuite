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

#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <algorithm>
#include <cmath>
#include <vector>
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
using std::cin;
using std::cout;
using std::cerr;
using std::iterator;
using std::endl;
using std::ifstream;
using std::map;
using std::vector;

action::action() {
}

action::~action() {
  property.clear();
}

/**
 * Check if the sizes and addresses of BARs match the given ones
 * @param property config file map fields
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
  string action_name;
  vector<uint16_t> gpus_location_id;
  struct pci_access *pacc;

  // get the action name
  rvs::actionbase::property_get_action_name(&error);
  if (error == 2) {
    msg = "action field is missing in smqt module";
    log(msg.c_str(), rvs::logerror);
    return -1;
  }

  gpu_get_all_location_id(gpus_location_id);
  // get the pci_access structure
  pacc = pci_alloc();
  // initialize the PCI library
  pci_init(pacc);
  // get the list of devices
  pci_scan_bus(pacc);

  struct pci_dev *dev;
  dev = pacc->devices;

  // get requested values
  for (auto it=property.begin(); it != property.end(); ++it) {
    if (it->first == "name")
      action_name = it->second;
    if (it->first == "bar1_req_size")
      bar1_req_size = std::atoi(it->second.c_str());
    if (it->first == "bar2_req_size")
      bar2_req_size = std::atoi(it->second.c_str());
    if (it->first == "bar4_req_size")
      bar4_req_size = std::atoi(it->second.c_str());
    if (it->first == "bar5_req_size")
      bar5_req_size = std::atoi(it->second.c_str());

    if (it->first == "bar1_base_addr_min")
      bar1_base_addr_min = std::atoi(it->second.c_str());
    if (it->first == "bar2_base_addr_min")
      bar2_base_addr_min = std::atoi(it->second.c_str());
    if (it->first == "bar4_base_addr_min")
      bar4_base_addr_min = std::atoi(it->second.c_str());

    if (it->first == "bar1_base_addr_max")
      bar1_base_addr_max = std::atoi(it->second.c_str());
    if (it->first == "bar2_base_addr_max")
      bar2_base_addr_max = std::atoi(it->second.c_str());
    if (it->first == "bar4_base_addr_max")
      bar4_base_addr_max = std::atoi(it->second.c_str());
  }

  // iterate over devices
  for (dev = pacc->devices; dev; dev = dev->next) {
    // fil in the info
    pci_fill_info(dev, PCI_FILL_IDENT | PCI_FILL_BASES \
    | PCI_FILL_CLASS | PCI_FILL_EXT_CAPS | PCI_FILL_CAPS | PCI_FILL_PHYS_SLOT);

    // computes the actual dev's location_id (sysfs entry)
    uint16_t dev_location_id = ((((uint16_t)(dev->bus)) << 8) | (dev->func));

    // check if this pci_dev corresponds to one of AMD GPUs
    auto it_gpu = find(gpus_location_id.begin(), \
    gpus_location_id.end(), dev_location_id);

    if (it_gpu == gpus_location_id.end())
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

    if (bar1_base_addr < bar1_base_addr_min)
      pass = false;
    if (bar1_base_addr > bar1_base_addr_max)
    pass = false;
    if (bar2_base_addr < bar2_base_addr_min)
      pass = false;
    if (bar2_base_addr > bar2_base_addr_max)
      pass = false;
    if (bar4_base_addr < bar4_base_addr_min)
      pass = false;
    if (bar4_base_addr > bar4_base_addr_max)
    pass = false;

    if (bar1_req_size > bar1_size)
      pass = false;
    if (bar2_req_size < bar2_size)
      pass = false;
    if (bar4_req_size < bar4_size)
      pass = false;
    if (bar5_req_size < bar5_size)
      pass = false;

    // loginfo
    unsigned int sec;
    unsigned int usec;
    rvs::lp::get_ticks(sec, usec);
    string msgs1, msgs2, msgs4, msgs5, msga1, msga2, msga4, pmsg, str;
    char hex_value[30];

    // formating bar1 size for print
    if (remainder(bar1_size, pow(2, 30)) == 0)
      str = std::to_string(static_cast<int>(round(bar1_size/pow(2, 30))))+" GB";
    else if (remainder(bar1_size, pow(2, 20)) == 0)
      str = std::to_string(static_cast<int>(round(bar1_size/pow(2, 20))))+" MB";
    else if (remainder(bar1_size, pow(2, 10)) == 0)
      str = std::to_string(static_cast<int>(round(bar1_size/pow(2, 10))))+" KB";
    msgs1 = "[" + action_name + "] " + " smqt bar1_size      "\
    + std::to_string(bar1_size) + " (" + str+ ")";

    // formating bar2 size for print
    if (remainder(bar2_size, pow(2, 30)) == 0)
      str = std::to_string(static_cast<int>(round(bar2_size/pow(2, 30))))+" GB";
    else if (remainder(bar2_size, pow(2, 20)) == 0)
      str = std::to_string(static_cast<int>(round(bar2_size/pow(2, 20))))+" MB";
    else if (remainder(bar2_size, pow(2, 10)) == 0)
      str = std::to_string(static_cast<int>(round(bar2_size/pow(2, 10))))+" KB";
    msgs2 = "[" + action_name + "] " + " smqt bar2_size      "\
    + std::to_string(bar2_size) + " (" + str+ ")";

    // formating bar4 size for print
    if (remainder(bar4_size, pow(2, 30)) == 0)
      str = std::to_string(static_cast<int>(round(bar4_size/pow(2, 30))))+" GB";
    else if (remainder(bar4_size, pow(2, 20)) == 0)
      str = std::to_string(static_cast<int>(round(bar4_size/pow(2, 20))))+" MB";
    else if (remainder(bar4_size, pow(2, 10)) == 0)
      str = std::to_string(static_cast<int>(round(bar4_size/pow(2, 10))))+" KB";
    msgs4 = "[" + action_name + "] " + " smqt bar4_size      "\
    + std::to_string(bar4_size) + " (" + str+ ")";

    // formating bar5 size for print
    if (remainder(bar5_size, pow(2, 30)) == 0)
      str = std::to_string(static_cast<int>(round(bar5_size/pow(2, 30))))+" GB";
    else if (remainder(bar5_size, pow(2, 20)) == 0)
      str = std::to_string(static_cast<int>(round(bar5_size/pow(2, 20))))+" MB";
    else if (remainder(bar5_size, pow(2, 10)) == 0)
      str = std::to_string(static_cast<int>(round(bar5_size/pow(2, 10))))+" KB";
    msgs5 = "[" + action_name + "] " + " smqt bar5_size      "\
    + std::to_string(bar5_size) + " (" + str+ ")";

    snprintf(hex_value, 0xFF, "%lX", bar1_base_addr);
    msga1 = "[" + action_name + "] " + " smqt bar1_base_addr " + hex_value;
    snprintf(hex_value, 0xFF, "%lX", bar2_base_addr);
    msga2 = "[" + action_name + "] " + " smqt bar2_base_addr " + hex_value;
    snprintf(hex_value, 0xFF, "%lX", bar4_base_addr);
    msga4 = "[" + action_name + "] " + " smqt bar4_base_addr " + hex_value;
    pmsg = "[" + action_name + "] " + " smqt " + std::to_string(pass);
    void* r = rvs::lp::LogRecordCreate("SMQT", action_name.c_str(), \
    rvs::loginfo, sec, usec);
    rvs::lp::Log(msgs1.c_str(), rvs::loginfo, sec, usec);
    rvs::lp::Log(msga1.c_str(), rvs::loginfo, sec, usec);
    rvs::lp::Log(msgs2.c_str(), rvs::loginfo, sec, usec);
    rvs::lp::Log(msga2.c_str(), rvs::loginfo, sec, usec);
    rvs::lp::Log(msgs4.c_str(), rvs::loginfo, sec, usec);
    rvs::lp::Log(msga4.c_str(), rvs::loginfo, sec, usec);
    rvs::lp::Log(msgs5.c_str(), rvs::loginfo, sec, usec);
    rvs::lp::Log(pmsg.c_str(), rvs::logresults);
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
