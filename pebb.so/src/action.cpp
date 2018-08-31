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
#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "pci_caps.h"
#include "gpu_util.h"
#include "rvs_util.h"
#include "rvsloglp.h"
#include "rvshsa.h"
#include "rvstimer.h"
#include "hsa/hsa.h"


#include "rvs_module.h"
#include "worker.h"

#define RVS_CONF_LOG_INTERVAL_KEY "log_interval"
#define DEFAULT_LOG_INTERVAL 10
#define DEFAULT_DURATION 10000

#define MODULE_NAME "pebb"
#define JSON_CREATE_NODE_ERROR "JSON cannot create node"

using std::cout;
using std::endl;
using std::cerr;
using std::string;
using std::vector;

//! Default constructor
pebbaction::pebbaction() {
}

//! Default destructor
pebbaction::~pebbaction() {
  property.clear();
}

/**
 *  * @brief reads the module's properties collection to see if
 * device to host transfers will be considered
 */
void pebbaction::property_get_h2d() {
  prop_h2d = true;
  auto it = property.find("host_to_device");
  if (it != property.end()) {
    if (it->second == "false")
      prop_h2d = false;
  }
}

/**
 *  * @brief reads the module's properties collection to see if
 * device to host transfers will be considered
 */
void pebbaction::property_get_d2h() {
  prop_d2h = true;
  auto it = property.find("device_to_host");
  if (it != property.end()) {
    if (it->second == "false")
      prop_d2h = false;
  }
}

/**
 * @brief reads the log interval from the module's properties collection
 * @param error pointer to a memory location where the error code will be stored
 */
void pebbaction::property_get_log_interval(int *error) {
  *error = 0;
  prop_log_interval = DEFAULT_LOG_INTERVAL;
  auto it = property.find(RVS_CONF_LOG_INTERVAL_KEY);
  if (it != property.end()) {
    if (is_positive_integer(it->second)) {
      prop_log_interval = std::stoul(it->second);
      if (prop_log_interval == 0)
        prop_log_interval = DEFAULT_LOG_INTERVAL;
    } else {
      *error = 1;
    }
  }
}

/**
 * @brief reads all PQT related configuration keys from
 * the module's properties collection
 * @return true if no fatal error occured, false otherwise
 */
bool pebbaction::get_all_pebb_config_keys(void) {;
  string msg;
  int error;

  property_get_log_interval(&error);
  if (error) {
    cerr << "RVS-PEBB: action: " << action_name <<
    "  invalid '" << RVS_CONF_LOG_INTERVAL_KEY << "'" << std::endl;
    return false;
  }

  property_get_h2d();
  property_get_d2h();

  return true;
}

/**
 * @brief reads all common configuration keys from
 * the module's properties collection
 * @return true if no fatal error occured, false otherwise
 */
bool pebbaction::get_all_common_config_keys(void) {
  string msg, sdevid, sdev;
  int error;
  // get the action name
  property_get_action_name(&error);
  if (error) {
    msg = "pqt [] action field is missing";
    log(msg.c_str(), rvs::logerror);
    return false;
  }

  // get <device> property value (a list of gpu id)
  if (has_property("device", &sdev)) {
    prop_device_all_selected = property_get_device(&error);
    if (error) {  // log the error & abort GST
      cerr << "RVS-PQT: action: " << action_name <<
      "  invalid 'device' key value " << sdev << std::endl;
      return false;
    }
  } else {
    cerr << "RVS-PQT: action: " << action_name <<
    "  key 'device' was not found" << std::endl;
    return false;
  }

  // get the <deviceid> property value
  if (has_property("deviceid", &sdevid)) {
    int devid = property_get_deviceid(&error);
    if (!error) {
      if (devid != -1) {
        prop_deviceid = static_cast<uint16_t>(devid);
        prop_device_id_filtering = true;
      }
    } else {
      cerr << "RVS-PQT: action: " << action_name <<
      "  invalid 'deviceid' key value " << sdevid << std::endl;
      return false;
    }
  } else {
    prop_device_id_filtering = false;
  }

  // get the other action/GST related properties
  rvs::actionbase::property_get_run_parallel(&error);
  if (error == 1) {
    cerr << "RVS-PQT: action: " << action_name <<
    "  invalid '" << RVS_CONF_PARALLEL_KEY <<
    "' key value" << std::endl;
    return false;
  }
  rvs::actionbase::property_get_run_duration(&error);
  if (error == 1) {
    cerr << "RVS-PQT: action: " << action_name <<
    "  invalid '" << RVS_CONF_DURATION_KEY <<
    "' key value" << std::endl;
    return false;
  }
  if (gst_run_duration_ms == 0) {
    gst_run_duration_ms = DEFAULT_DURATION;
  }
  return true;
}

/**
 * @brief Create thread objects based on action description in configuation
 * file.
 *
 * Threads are created but are not started. Execution, one by one of parallel,
 * depends on "parallel" key in configuration file. Pointers to created objects
 * are stored in "test_array" member
 *
 * @return 0 - if successfull, non-zero otherwise
 *
 * */
int pebbaction::create_threads() {
  std::string msg;
  std::vector<uint16_t> gpu_id;
  std::vector<uint16_t> gpu_device_id;

  gpu_get_all_gpu_id(&gpu_id);
  gpu_get_all_device_id(&gpu_device_id);

  for (size_t i = 0; i < gpu_id.size(); i++) {
    if (prop_device_id_filtering) {
      if (prop_deviceid != gpu_device_id[i]) {
        continue;
      }
    }
    // filter out by listed sources
    if (!prop_device_all_selected) {
      const auto it = std::find(device_prop_gpu_id_list.cbegin(),
                                device_prop_gpu_id_list.cend(),
                                std::to_string(gpu_id[i]));
      if (it == device_prop_gpu_id_list.cend()) {
        continue;
      }
    }

    int dstnode;
    int srcnode;

    for (uint cpu_index = 0;
         cpu_index < rvs::hsa::Get()->cpu_list.size(); cpu_index++) {
        // GPUs are peers, create transaction for them
        if (prop_h2d) {
          dstnode = rvs::gpulist::GetNodeIdFromGpuId(gpu_id[i]);
          if (dstnode < 0) {
            std::cerr << "RVS-PEBB: no node found for destination GPU ID "
            //<< std::to_string(gpu_id[i]);
            << std::to_string(i);
            return -1;
          }
          srcnode = rvs::hsa::Get()->cpu_list[cpu_index].node;
          pebbworker* p = new pebbworker;
          p->initialize(srcnode, dstnode, false);
          test_array.push_back(p);
        }
        if (prop_d2h) {
          srcnode = rvs::gpulist::GetNodeIdFromGpuId(gpu_id[i]);
          if (srcnode < 0) {
            std::cerr << "RVS-PEBB: no node found for source GPU ID "
            << std::to_string(gpu_id[i]);
            return -1;
          }
          dstnode = rvs::hsa::Get()->cpu_list[cpu_index].node;
          pebbworker* p = new pebbworker;
          p->initialize(srcnode, dstnode, false);
          test_array.push_back(p);
        }
    }
  }
  return 0;
}

/**
 * @brief Delete test thread objects at the end of action execution
 *
 * @return 0 - if successfull, non-zero otherwise
 *
 * */
int pebbaction::destroy_threads() {
  for (auto it = test_array.begin(); it != test_array.end(); ++it) {
    (*it)->stop();
    delete *it;
  }
  return 0;
}

/**
 * @brief Main action execution entry point. Implements test logic.
 *
 * @return 0 - if successfull, non-zero otherwise
 *
 * */
int pebbaction::run() {
  int sts;
  string msg;
  // check for -j flag (json logging)
  if (property.find("cli.-j") != property.end()) {
    unsigned int sec;
    unsigned int usec;

    rvs::lp::get_ticks(&sec, &usec);
    bjson = true;
    json_rcqt_node = rvs::lp::LogRecordCreate(MODULE_NAME,
                              action_name.c_str(), rvs::loginfo, sec, usec);
    if (json_rcqt_node == NULL) {
      // log the error
      msg =
      action_name + " " + MODULE_NAME + " "
      + JSON_CREATE_NODE_ERROR;
      log(msg.c_str(), rvs::logerror);
    }
  }

  if (!get_all_common_config_keys())
    return -1;
  if (!get_all_pebb_config_keys())
    return -1;
  sts = create_threads();

  if (sts != 0) {
    return sts;
  }

  if (gst_runs_parallel) {
    sts = run_parallel();
  } else {
    sts = run_single();
  }

  destroy_threads();

  return sts;
}

/**
 * @brief Execute test transfers one by one, in round robin fashion, for the
 * duration of the action.
 *
 * @return 0 - if successfull, non-zero otherwise
 *
 * */
int pebbaction::run_single() {
  // define timers
  rvs::timer<pebbaction> timer_running(&pebbaction::do_running_average, this);
  rvs::timer<pebbaction> timer_final(&pebbaction::do_final_average, this);

  // let the test run
  brun = true;

  // start timers
  timer_final.start(gst_run_duration_ms, true);  // ticks only once
  timer_running.start(prop_log_interval);        // ticks continuously

  int i = 0;
  // iterate through test array and invoke tests one by one
  do {
    for (auto it = test_array.begin(); brun && it != test_array.end(); ++it) {
      (*it)->do_transfer();
      cout << "do transfer " << i++ << endl;
      sleep(1);
    }
  } while (brun);

  timer_running.stop();
  timer_final.stop();

  print_final_average();

  return 0;
}

/**
 * @brief Execute test transfers all at once, for the
 * duration of the action.
 *
 * @return 0 - if successfull, non-zero otherwise
 *
 * */
int pebbaction::run_parallel() {
  // define timers
  rvs::timer<pebbaction> timer_running(&pebbaction::do_running_average, this);
  rvs::timer<pebbaction> timer_final(&pebbaction::do_final_average, this);

  // let the test run
  brun = true;

  int i = 0;
  // start all worker threads
  for (auto it = test_array.begin(); it != test_array.end(); ++it) {
    (*it)->start();
    cout << "do transfer " << i++ << endl;
  }

  // start timers
  timer_final.start(gst_run_duration_ms, true);  // ticks only once
  timer_running.start(prop_log_interval);        // ticks continuously

  // wait for test to complete
  while (brun) {
    sleep(1);
  }

  timer_running.stop();
  timer_final.stop();

  print_final_average();

  return 0;
}

/**
 * @brief Collect running average bandwidth data for all the tests and prints
 * them on cout every log_interval msec.
 *
 * @return 0 - if successfull, non-zero otherwise
 *
 * */
int pebbaction::print_running_average() {
  int src_node, dst_node;
  int src_id, dst_id;
  bool bidir;
  size_t current_size;
  double duration;
  std::string msg;
  for (auto it = test_array.begin(); it != test_array.end() ; ++it) {
    (*it)->get_running_data(&src_node, &dst_node, &bidir,
                            &current_size, &duration);

    double bandiwdth = current_size/duration/(1024*1024*1024);
    if (bidir) {
      bandiwdth *=2;
    }
    char buff[64];
    snprintf( buff, sizeof(buff), "%.2f GBps", bandiwdth);
    src_id = rvs::gpulist::GetGpuIdFromNodeId(src_node);
    dst_id = rvs::gpulist::GetGpuIdFromNodeId(dst_node);

    msg = "[" + action_name + "] bandwidth  " +
    std::to_string(src_id) + " " + std::to_string(dst_id) + " :" +buff;
    rvs::lp::Log(msg, rvs::loginfo);
  }

  return 0;
}

/**
 * @brief Collect bandwidth totals for all the tests and prints
 * them on cout at the end of action execution
 *
 * @return 0 - if successfull, non-zero otherwise
 *
 * */
int pebbaction::print_final_average() {
  int src_node, dst_node;
  int src_id, dst_id;
  bool bidir = false;
  size_t current_size;
  double duration;
  std::string msg;
  for (auto it = test_array.begin(); it != test_array.end(); ++it) {
    (*it)->get_final_data(&src_node, &dst_node, &bidir,
                          &current_size, &duration);

    double bandiwdth = current_size/duration/(1024*1024*1024);

    char buff[64];
    snprintf( buff, sizeof(buff), "%.2f GBps", bandiwdth);
    src_id = rvs::gpulist::GetGpuIdFromNodeId(src_node);
    dst_id = rvs::gpulist::GetGpuIdFromNodeId(dst_node);
    if (src_id == 0)
      src_id = src_node;
    if (dst_id == 0)
      dst_id = dst_node;

    msg = "[" + action_name + "] bandwidth  " +
    std::to_string(src_id) + " " + std::to_string(dst_id) +
    " :" + buff +
    "  duration: " + std::to_string(duration) + " ms";

    rvs::lp::Log(msg, rvs::logresults);
  }
  return 0;
}

void pebbaction::do_final_average() {
  rvs::lp::Log("pqt in do_final_average", rvs::logdebug);
  brun = false;
}

void pebbaction::do_running_average() {
  rvs::lp::Log("in do_running_average", rvs::logdebug);
  print_running_average();
}
