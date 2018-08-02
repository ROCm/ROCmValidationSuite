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

#include "rvs_module.h"
#include "worker.h"

#define RVS_CONF_LOG_INTERVAL_KEY "log_interval"
#define DEFAULT_LOG_INTERVAL 500

using std::cerr;
using std::string;
using std::vector;

//! Default constructor
action::action() {
}

//! Default destructor
action::~action() {
  property.clear();
}

/**
 * gets the peer gpu_id list from the module's properties collection
 * @param error pointer to a memory location where the error code will be stored
 * @return true if "all" is selected, false otherwise
 */
bool action::property_get_peers(int *error) {
    *error = 0;  // init with 'no error'
    auto it = property.find("peers");
    if (it != property.end()) {
        if (it->second == "all") {
            return true;
        } else {
            // split the list of gpu_id
            prop_peers = str_split(it->second,
                    YAML_DEVICE_PROP_DELIMITER);
            if (prop_peers.empty()) {
                *error = 1;  // list of gpu_id cannot be empty
            } else {
                for (vector<string>::iterator it_gpu_id =
                        prop_peers.begin();
                        it_gpu_id != prop_peers.end(); ++it_gpu_id)
                    if (!is_positive_integer(*it_gpu_id)) {
                        *error = 1;
                        break;
                    }
            }
            return false;
        }

    } else {
        *error = 1;
        // when error is set, it doesn't really matter whether the method
        // returns true or false
        return false;
    }
}

/**
 * gets the peer deviceid from the module's properties collection
 * @param error pointer to a memory location where the error code will be stored
 * @return deviceid value if valid, -1 otherwise
 */
int action::property_get_peer_deviceid(int *error) {
    auto it = property.find("peer_deviceid");
    int deviceid = -1;
    *error = 0;  // init with 'no error'

    if (it != property.end()) {
        if (it->second != "") {
            if (is_positive_integer(it->second)) {
                deviceid = std::stoi(it->second);
            } else {
                *error = 1;  // we have something but it's not a number
            }
        } else {
            *error = 1;  // we have an empty string
        }
    }
    return deviceid;
}

/**
 * @brief reads the module's properties collection to see whether bandiwdth
 * tests should be run after peer check
 */
void action::property_get_test_bandwidth(int *error) {
  prop_test_bandwidth = false;
  auto it = property.find("test_bandwidth");
  if (it != property.end()) {
    if (it->second == "true") {
      prop_test_bandwidth = true;
      *error = 0;
    } else if (it->second == "false") {
      *error = 0;
    } else {
      *error = 1;
    }
  } else {
    *error = 2;
  }
}

/**
 * @brief reads the module's properties collection to see whether bandiwdth
 * tests should be run in both directions
 */
void action::property_get_bidirectional(int *error) {
  prop_bidirectional = false;
  auto it = property.find("bidirectional");
  if (it != property.end()) {
    if (it->second == "true") {
      prop_bidirectional = true;
      *error = 0;
    } else if (it->second == "false") {
      *error = 0;
    } else {
      *error = 1;
    }
  } else {
    *error = 2;
  }
}

/**
 * @brief reads the log interval from the module's properties collection
 * @param error pointer to a memory location where the error code will be stored
 */
void action::property_get_log_interval(int *error) {
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
bool action::get_all_pqt_config_keys(void) {
  int    error;
  string msg;

  property_get_peers(&error);
  if (error) {
    cerr << "RVS-PQT: action: " << action_name <<
        "  invalid '" << "peers" << "'" << std::endl;
    return false;
  }

  prop_peer_deviceid = property_get_peer_deviceid(&error);
  if (error) {
    cerr << "RVS-PQT: action: " << action_name <<
        "  invalid 'peer_deviceid '" << std::endl;
    return false;
  }

  property_get_test_bandwidth(&error);
  if (error) {
    cerr << "RVS-PQT: action: " << action_name <<
        "  invalid 'test_bandwidth'" << std::endl;
    return false;
  }

  property_get_log_interval(&error);
  if (error) {
    cerr << "RVS-PQT: action: " << action_name <<
        "  invalid '" << RVS_CONF_LOG_INTERVAL_KEY << "'" << std::endl;
    return false;
  }

  property_get_bidirectional(&error);
  if (error) {
    cerr << "RVS-PQT: action: " << action_name <<
        "  invalid 'bidirectional'" << std::endl;
    return false;
  }

  return true;
}

/**
 * @brief reads all common configuration keys from
 * the module's properties collection
 * @return true if no fatal error occured, false otherwise
 */
bool action::get_all_common_config_keys(void) {
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
    if (has_property("device", sdev)) {
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
    if (has_property("deviceid", sdevid)) {
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
    }

    // get the other action/GST related properties
    rvs::actionbase::property_get_run_parallel(&error);
    if (error == 1) {
        cerr << "RVS-PQT: action: " << action_name <<
            "  invalid '" << RVS_CONF_PARALLEL_KEY <<
            "' key value" << std::endl;
        return false;
    }

    rvs::actionbase::property_get_run_count(&error);
    if (error == 1) {
        cerr << "RVS-PQT: action: " << action_name <<
            "  invalid '" << RVS_CONF_COUNT_KEY << "' key value" << std::endl;
        return false;
    }

    rvs::actionbase::property_get_run_wait(&error);
    if (error == 1) {
        cerr << "RVS-PQT: action: " << action_name <<
            "  invalid '" << RVS_CONF_WAIT_KEY << "' key value" << std::endl;
        return false;
    }

    rvs::actionbase::property_get_run_duration(&error);
    if (error == 1) {
        cerr << "RVS-PQT: action: " << action_name <<
            "  invalid '" << RVS_CONF_DURATION_KEY <<
            "' key value" << std::endl;
        return false;
    }

    return true;
}

int action::create_threads() {
  Worker* p = new Worker;
  p->initialize(4, 5, false);

  test_array.push_back(p);

  return 0;
}

int action::destroy_threads() {
  for (auto it = test_array.begin(); it != test_array.end(); ++it) {
    delete *it;
  }

  return 0;
}

int action ::run() {
  if (!get_all_common_config_keys())
    return -1;
  if (!get_all_pqt_config_keys())
    return -1;

  create_threads();

  if (gst_runs_parallel) {
    run_parallel();
  } else {
    run_single();
  }

  destroy_threads();

  return 0;
}

int action::run_single() {
  // define timers
  rvs::timer<action> timer_running(&action::do_running_average, this);
  rvs::timer<action> timer_final(&action::do_final_average, this);

  // let the test run
  brun = true;

  // start timers
  timer_final.start(10000, true);  // ticks only once
  timer_running.start(500);        // ticks continuously

  // iterate through test array and invoke tests one by one
  do {
    for (auto it = test_array.begin(); brun && it != test_array.end(); ++it) {
      (*it)->do_transfer();
     sleep(1);
    }
  } while (brun);

  timer_running.stop();
  timer_final.stop();

  print_final_average();

  return 0;
}

int action::run_parallel() {
  return 0;
}

int action::print_running_average() {
  int src_node, dst_node;
  int src_id, dst_id;
  bool bidir;
  size_t current_size;
  double duration;
  std::string msg;

  for (auto it = test_array.begin(); brun && it != test_array.end(); ++it) {
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

    msg = "[" + action_name + "] p2p-bandwidth  " +
           std::to_string(src_id) + " " + std::to_string(dst_id) +
           "  bidirectional: " +
           std::string(bidir ? "true" : "false") +
           "  " + buff;
    rvs::lp::Log(msg, rvs::loginfo);
    sleep(1);
  }

  return 0;
}

int action::print_final_average() {
  int src_node, dst_node;
  int src_id, dst_id;
  bool bidir;
  size_t current_size;
  double duration;
  std::string msg;

  for (auto it = test_array.begin(); it != test_array.end(); ++it) {
    (*it)->get_final_data(&src_node, &dst_node, &bidir,
                            &current_size, &duration);

    double bandiwdth = current_size/duration/(1024*1024*1024);
    if (bidir) {
      bandiwdth *=2;
    }
    char buff[64];
    snprintf( buff, sizeof(buff), "%.2f GBps", bandiwdth);
    src_id = rvs::gpulist::GetGpuIdFromNodeId(src_node);
    dst_id = rvs::gpulist::GetGpuIdFromNodeId(dst_node);

    msg = "[" + action_name + "] p2p-bandwidth  " +
           std::to_string(src_id) + " " + std::to_string(dst_id) +
           "  bidirectional: " +
           std::string(bidir ? "true" : "false") +
           "  " + buff +
           "  duration: " + std::to_string(duration) + " ms";

    rvs::lp::Log(msg, rvs::logresults);
    sleep(1);
  }

  return 0;
}

void action::do_final_average() {
  rvs::lp::Log("pqt in do_final_average", rvs::logdebug);
  brun = false;
}

void action::do_running_average() {
  rvs::lp::Log("in do_running_average", rvs::logdebug);

  print_running_average();
}


