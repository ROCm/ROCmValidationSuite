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
#include "include/worker.h"

#include <chrono>
#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>

#ifdef __cplusplus
extern "C" {
#endif
#include <pci/pci.h>
#include <linux/pci.h>
#ifdef __cplusplus
}
#endif

#include "include/rvs_module.h"
#include "include/pci_caps.h"
#include "include/gpu_util.h"
#include "include/rvsloglp.h"
#define MODULE_NAME "PESM"

using std::string;
using std::vector;
using std::map;

Worker::Worker() {
  bfiltergpu = false;
}
Worker::~Worker() {}

/**
 * @brief Sets GPU IDs for filtering
 * @arg GpuIds Array of GPU GpuIds
 */
void Worker::set_gpuids(const std::vector<uint16_t>& GpuIds) {
  gpuids = GpuIds;
  if (gpuids.size()) {
    bfiltergpu = true;
  }
}

/**
 * @brief Thread function
 *
 * Loops while brun == TRUE and performs polled monitoring every 1msec.
 *
 * */
void Worker::run() {
  char buff[1024];

  map<string, string>::iterator it;
  vector<uint16_t> gpus_location_id;
  map<uint16_t, string> old_speed_val;
  map<uint16_t, string> old_pwr_val;

  struct pci_access *pacc;
  struct pci_dev *dev;

  unsigned int sec;
  unsigned int usec;
  void* r;
  rvs::action_result_t action_result;
  map<uint16_t, string> speed_change;
  map<uint16_t, string> power_change;
  string msg;

  brun = true;

  // get timestamp
  rvs::lp::get_ticks(&sec, &usec);

  // add string output
  msg = "[" + action_name + "] " + "PCIe link speed and power monitoring started ...";
  rvs::lp::Log(msg, rvs::logresults, sec, usec);

  if (bjson) {
    // add JSON output
    r = rvs::lp::LogRecordCreate("pesm", action_name.c_str(), rvs::logresults,
        sec, usec);
    rvs::lp::AddString(r, "msg", "started");
    rvs::lp::LogRecordFlush(r);
  }

  // worker thread has started
  while (brun) {
    rvs::lp::Log("[" + action_name + "] pesm worker thread is running...",
                 rvs::logtrace);

    // get the pci_access structure
    pacc = pci_alloc();
    // initialize the PCI library
    pci_init(pacc);
    // get the list of devices
    pci_scan_bus(pacc);

    // iterate over devices
    for (dev = pacc->devices; dev; dev = dev->next) {

      int known_fields = pci_fill_info(dev, PCI_FILL_IDENT | PCI_FILL_BASES | PCI_FILL_CLASS
      | PCI_FILL_EXT_CAPS | PCI_FILL_CAPS | PCI_FILL_PHYS_SLOT);  // fil in the info

      // computes the actual dev's location_id (sysfs entry)
      uint16_t dev_location_id =
        ((((uint16_t)(dev->bus)) << 8) | ((uint16_t)(dev->dev)) << 3);

      uint16_t gpu_id;
      // if not and AMD GPU just continue
      if (rvs::gpulist::location2gpu(dev_location_id, &gpu_id))
        continue;

      // device_id filtering
      if ( device_id != 0 && dev->device_id != device_id)
        continue;

      // gpu id filtering
      if (bfiltergpu) {
        auto itgpuid = find(gpuids.begin(), gpuids.end(), gpu_id);
        if (itgpuid == gpuids.end())
          continue;
      }

      rvs::lp::get_ticks(&sec, &usec);

      // get current speed for the link
      get_link_stat_cur_speed(dev, buff);
      string new_speed_val(buff);
      if(old_speed_val[gpu_id].empty()) {
        old_speed_val[gpu_id] = new_speed_val;
      }

      // get current power state for GPU
      get_pwr_curr_state(dev, buff);
      string new_pwr_val(buff);
      if(old_pwr_val[gpu_id].empty()) {
        old_pwr_val[gpu_id] = new_pwr_val;
        continue;
      }

      // link speed changed
      if (old_speed_val[gpu_id] != new_speed_val) {
        // new value is different, so store it;
        old_speed_val[gpu_id] = new_speed_val;

        msg = "[" + action_name + "] " + std::to_string(gpu_id) +
          " PCIe link speed changed " + new_speed_val;
        rvs::lp::Log(msg, rvs::loginfo, sec, usec);

        action_result.state = rvs::actionstate::ACTION_RUNNING;
        action_result.status = rvs::actionstatus::ACTION_SUCCESS;
        action_result.output = msg.c_str();
        action.action_callback(&action_result);

        speed_change[gpu_id] = "true";
      }
      else {
        msg = "[" + action_name + "] " + "pesm " +
          std::to_string(gpu_id) + " PCIe link speed unchanged " + new_speed_val;
        rvs::lp::Log(msg, rvs::loginfo, sec, usec);

        if(speed_change[gpu_id].empty()) {
          speed_change[gpu_id] = "false";
        }
      }

      // power state changed
      if (old_pwr_val[gpu_id] != new_pwr_val) {
        // new value is different, so store it;
        old_pwr_val[gpu_id] = new_pwr_val;

        msg = "[" + action_name + "] " + std::to_string(gpu_id)
          + " PCIe power state changed " + new_pwr_val;
        rvs::lp::Log(msg, rvs::loginfo, sec, usec);

        action_result.state = rvs::actionstate::ACTION_RUNNING;
        action_result.status = rvs::actionstatus::ACTION_SUCCESS;
        action_result.output = msg.c_str();
        action.action_callback(&action_result);

        power_change[gpu_id] = "true";
      }
      else {
        msg = "[" + action_name + "] " + 
          std::to_string(gpu_id) + " PCIe power state unchanged " + new_pwr_val;
        rvs::lp::Log(msg, rvs::loginfo, sec, usec);

        if(power_change[gpu_id].empty()) {
          power_change[gpu_id] = "false";
        }
      }
    }

    pci_cleanup(pacc);

    sleep(1);
  }

  // get timestamp
  rvs::lp::get_ticks(&sec, &usec);

  if (bjson) {
    // add JSON output
    r = rvs::lp::LogRecordCreate("PESM",
        stop_action_name.c_str(), rvs::logresults,
        sec, usec);
  }

  string gpu_json;
  string speed_json;
  string power_json;

  for(auto i : speed_change) {

    msg = "[" + stop_action_name + "]" +
      " GPU " + std::to_string(i.first) + " PCIe speed change " +  i.second;

    rvs::lp::Log(msg, rvs::logresults, sec, usec);

    if (bjson) {
      gpu_json += std::to_string(i.first) + ", ";
      speed_json += i.second + ", ";
    }
  }

  if (bjson) {
    gpu_json = gpu_json.substr(0, (gpu_json.size() - 2));
    speed_json = speed_json.substr(0, (speed_json.size() - 2));

    rvs::lp::AddString(r, "gpu", gpu_json);
    rvs::lp::AddString(r, "speed_change", speed_json);
  }

  for(auto i : power_change) {

    msg = "[" + stop_action_name + "]" +
      " GPU " + std::to_string(i.first) + " PCIe power change " +  i.second;
    rvs::lp::Log(msg, rvs::logresults, sec, usec);

    if (bjson) {
      power_json += i.second + ", ";
    }
  }

  if (bjson) {
    power_json = power_json.substr(0, (power_json.size() - 2));

    rvs::lp::AddString(r, "power_change", power_json);
    rvs::lp::AddString(r, "msg", "stopped");
    rvs::lp::AddString(r, "result", "true");

    rvs::lp::LogRecordFlush(r);
  }

  // add string output
  msg = "[" + stop_action_name + "] PCIe monitoring ended after wait duration.";
  rvs::lp::Log(msg, rvs::logresults, sec, usec);

  action_result.state = rvs::actionstate::ACTION_COMPLETED;
  action_result.status = rvs::actionstatus::ACTION_SUCCESS;
  action_result.output = msg.c_str();
  action.action_callback(&action_result);


  rvs::lp::Log("[" + stop_action_name + "] pesm worker thread has finished",
      rvs::logdebug);
}

/**
 * @brief Stops monitoring
 *
 * Sets brun member to FALSE thus signaling end of monitoring.
 * Then it waits for std::thread to exit before returning.
 *
 * */
void Worker::stop() {
  rvs::lp::Log("[" + stop_action_name + "] pesm in Worker::stop()",
               rvs::logtrace);
  // reset "run" flag
  brun = false;
  // (give thread chance to finish processing and exit)
  sleep(200);

  // wait a bit to make sure thread has exited
  try {
    if (t.joinable())
      t.join();
  }
  catch(...) {
  }
}
