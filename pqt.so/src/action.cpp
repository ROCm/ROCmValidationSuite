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
#define DEFAULT_LOG_INTERVAL 1000
#define DEFAULT_DURATION 10000

#define MODULE_NAME "pqt"
#define JSON_CREATE_NODE_ERROR "JSON cannot create node"
#define RVS_CONF_BLOCK_SIZE_KEY "block_size"

using std::cerr;
using std::string;
using std::vector;

//! Default constructor
pqtaction::pqtaction() {
  prop_deviceid = -1;
  prop_device_id_filtering = false;
  prop_peer_deviceid = -1;
}

//! Default destructor
pqtaction::~pqtaction() {
  property.clear();
}

/**
 * gets the peer gpu_id list from the module's properties collection
 * @param error pointer to a memory location where the error code will be stored
 * @return true if "all" is selected, false otherwise
 */
bool pqtaction::property_get_peers(int *error) {
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
int pqtaction::property_get_peer_deviceid(int *error) {
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
 * @brief reads the module's properties collection to see whether bandwidth
 * tests should be run after peer check
 */
void pqtaction::property_get_test_bandwidth(int *error) {
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
 * @brief reads the module's properties collection to see whether bandwidth
 * tests should be run in both directions
 */
void pqtaction::property_get_bidirectional(int *error) {
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
 * @brief reads all PQT related configuration keys from
 * the module's properties collection
 * @return true if no fatal error occured, false otherwise
 */
bool pqtaction::get_all_pqt_config_keys(void) {
  int    error;
  string msg;

  prop_peer_device_all_selected = property_get_peers(&error);
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

  prop_log_interval = property_get_log_interval(&error);
  if (error == 1) {
    cerr << "RVS-PQT: action: " << action_name <<
        "  invalid '" << RVS_CONF_LOG_INTERVAL_KEY << "'" << std::endl;
    return false;
  } else if (error == 2) {
    prop_log_interval = DEFAULT_LOG_INTERVAL;
  }

  property_get_bidirectional(&error);
  if (error) {
    if (prop_test_bandwidth == true) {
      cerr << "RVS-PQT: action: " << action_name <<
          "  invalid 'bidirectional'" << std::endl;
      return false;
    }
  }

  property_get_uint_list(RVS_CONF_BLOCK_SIZE_KEY, YAML_DEVICE_PROP_DELIMITER,
                         &block_size, &b_block_size_all, &error);
  if (error == 1) {
      cerr << "RVS-PQT: action: " << action_name << "  invalid '"
           << RVS_CONF_BLOCK_SIZE_KEY << "' key" << std::endl;
      return false;
  } else if (error == 2) {
    b_block_size_all = true;
    block_size.clear();
  }

  return true;
}

/**
 * @brief reads all common configuration keys from
 * the module's properties collection
 * @return true if no fatal error occured, false otherwise
 */
bool pqtaction::get_all_common_config_keys(void) {
  string msg, sdevid, sdev;
  int error;

  // get the action name
  property_get_action_name(&error);
  if (error) {
    msg = "action field is missing";
    cerr << "RVS-PQT: " << msg;
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
int pqtaction::create_threads() {
  std::string msg;

  std::vector<uint16_t> gpu_id;
  std::vector<uint16_t> gpu_device_id;
  uint16_t transfer_ix = 0;
  bool bmatch_found = false;

  gpu_get_all_gpu_id(&gpu_id);
  gpu_get_all_device_id(&gpu_device_id);

  for (size_t i = 0; i < gpu_id.size(); i++) {    // all possible sources
    // filter out by source device id
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

    for (size_t j = 0; j < gpu_id.size(); j++) {  // all possible peers
      // filter out by peer id
      if (prop_peer_deviceid > 0) {
        if (prop_peer_deviceid != gpu_device_id[j]) {
          continue;
        }
      }

      // filter out by listed peers
      if (!prop_peer_device_all_selected) {
        const auto it = std::find(prop_peers.cbegin(),
                                  prop_peers.cend(),
                                  std::to_string(gpu_id[j]));
        if (it == prop_peers.cend()) {
          continue;
        }
      }

      // signal that at lease one matching src-dst combination
      // has been found:
      bmatch_found = true;

      // perform peer check
      if (is_peer(gpu_id[i], gpu_id[j])) {
        msg = "[" + action_name + "] p2p "
            + std::to_string(gpu_id[i]) + " "
            + std::to_string(gpu_id[j]) + " true";
        rvs::lp::Log(msg, rvs::logresults);
        if (bjson) {
          unsigned int sec;
          unsigned int usec;
          rvs::lp::get_ticks(&sec, &usec);
          json_rcqt_node = rvs::lp::LogRecordCreate(MODULE_NAME,
                              action_name.c_str(), rvs::logresults, sec, usec);
          if (json_rcqt_node != NULL) {
            rvs::lp::AddString(json_rcqt_node, "src",
                               std::to_string(gpu_id[i]));
            rvs::lp::AddString(json_rcqt_node, "dst",
                               std::to_string(gpu_id[j]));
            rvs::lp::AddString(json_rcqt_node, "p2p", "true");
            rvs::lp::LogRecordFlush(json_rcqt_node);
          }
        }

        // GPUs are peers, create transaction for them
        int srcnode = rvs::gpulist::GetNodeIdFromGpuId(gpu_id[i]);
        if (srcnode < 0) {
          std::cerr << "RVS-PQT: no node found for GPU ID "
          << std::to_string(gpu_id[i]);
          return -1;
        }

        int dstnode = rvs::gpulist::GetNodeIdFromGpuId(gpu_id[j]);
        if (srcnode < 0) {
          std::cerr << "RVS-PQT: no node found for GPU ID "
          << std::to_string(gpu_id[j]);
          return -1;
        }

        if (prop_test_bandwidth) {
          transfer_ix += 1;
          pqtworker* p = new pqtworker;
          p->initialize(srcnode, dstnode, prop_bidirectional);
          p->set_name(action_name);
          p->set_stop_name(action_name);
          p->set_transfer_ix(transfer_ix);
          p->set_block_sizes(block_size);
          test_array.push_back(p);
        }

      } else {
        msg = "[" + action_name + "] p2p "
            + std::to_string(gpu_id[i]) + " "
            + std::to_string(gpu_id[j]) + " false";
        rvs::lp::Log(msg, rvs::logresults);
        if (bjson) {
          unsigned int sec;
          unsigned int usec;
          rvs::lp::get_ticks(&sec, &usec);
          json_rcqt_node = rvs::lp::LogRecordCreate(MODULE_NAME,
                              action_name.c_str(), rvs::logresults, sec, usec);
          if (json_rcqt_node != NULL) {
            rvs::lp::AddString(json_rcqt_node,
                               "src", std::to_string(gpu_id[i]));
            rvs::lp::AddString(json_rcqt_node,
                               "dst", std::to_string(gpu_id[j]));
            rvs::lp::AddString(json_rcqt_node,
                               "p2p", "false");
            rvs::lp::LogRecordFlush(json_rcqt_node);
          }
        }
      }
    }
  }

  if (prop_test_bandwidth && test_array.size() < 1) {
    std::string diag;
    if (bmatch_found) {
      diag = "No peers found";
    } else {
      diag = "No devices match criteria from the test configuation";
    }
    msg = "[" + action_name + "] p2p-bandwidth " + diag;
    rvs::lp::Log(msg, rvs::logerror);
    if (bjson) {
      unsigned int sec;
      unsigned int usec;
      rvs::lp::get_ticks(&sec, &usec);
      json_rcqt_node = rvs::lp::LogRecordCreate("p2p-bandwidth",
                              action_name.c_str(), rvs::logerror, sec, usec);
      if (json_rcqt_node != NULL) {
        rvs::lp::AddString(json_rcqt_node,
          "message",
          diag);
        rvs::lp::LogRecordFlush(json_rcqt_node);
      }
    }
    return 0;
  }

  for (auto it = test_array.begin(); it != test_array.end(); ++it) {
    (*it)->set_transfer_num(test_array.size());
  }

  return 0;
}

/**
 * @brief Delete test thread objects at the end of action execution
 *
 * @return 0 - if successfull, non-zero otherwise
 *
 * */
int pqtaction::destroy_threads() {
  for (auto it = test_array.begin(); it != test_array.end(); ++it) {
    (*it)->set_stop_name(action_name);
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
int pqtaction::run() {
  int sts;
  string msg;

  rvs::lp::Log("int pqtaction::run()", rvs::logtrace);

  if (!get_all_common_config_keys()) {
    cerr << "RVS-PQT: " << "Error in get_all_common_config_keys()"<< std::endl;
    return -1;
  }
  if (!get_all_pqt_config_keys()) {
    cerr << "RVS-PQT: " << "Error in get_all_pqt_config_keys()"<< std::endl;
    return -1;
  }

  // log_interval must be less than duration
  if (prop_log_interval > 0 && gst_run_duration_ms > 0) {
    if (static_cast<uint64_t>(prop_log_interval) > gst_run_duration_ms) {
      cerr << "RVS-PQT: action: " << action_name <<
          "  log_interval must be less than duration" << std::endl;
      return -1;
    }
  }

  sts = create_threads();
  if (sts)
    return sts;

  if (prop_test_bandwidth && test_array.size() > 0) {
    if (gst_runs_parallel) {
      sts = run_parallel();
    } else {
      sts = run_single();
    }
  }

  // do cleanup
  destroy_threads();

  return sts;
}

/**
 * @brief Check if two GPU can access each other memory
 *
 * @param Src GPU ID of the source GPU
 * @param Dst GPU ID of the destination GPU
 *
 * @return 0 - no access, 1 - Src can acces Dst, 2 - both have access
 *
 * */
int pqtaction::is_peer(uint16_t Src, uint16_t Dst) {
  //! ptr to RVS HSA singleton wrapper
  rvs::hsa* pHsa;

  if (Src == Dst) {
    return 0;
  }
  pHsa = rvs::hsa::Get();

  // GPUs are peers, create transaction for them
  int srcnode = rvs::gpulist::GetNodeIdFromGpuId(Src);
  if (srcnode < 0) {
    std::cerr << "RVS-PQT: no node found for GPU ID "
    << std::to_string(Src);
    return 0;
  }

  int dstnode = rvs::gpulist::GetNodeIdFromGpuId(Dst);
  if (srcnode < 0) {
    std::cerr << "RVS-PQT: no node found for GPU ID "
    << std::to_string(Dst);
    return 0;
  }

  return pHsa->rvs::hsa::GetPeerStatus(srcnode, dstnode);
}

/**
 * @brief Execute test transfers one by one, in round robin fashion, for the
 * duration of the action.
 *
 * @return 0 - if successfull, non-zero otherwise
 *
 * */
int pqtaction::run_single() {
  RVSTRACE_
  // define timers
  rvs::timer<pqtaction> timer_running(&pqtaction::do_running_average, this);
  rvs::timer<pqtaction> timer_final(&pqtaction::do_final_average, this);

  unsigned int iter = gst_run_count > 0 ? gst_run_count : 1;
  unsigned int step = gst_run_count == 0 ? 0 : 1;

  // let the test run
  brun = true;

  // start timers
  if (gst_run_duration_ms) {
    RVSTRACE_
    timer_final.start(gst_run_duration_ms, true);  // ticks only once
  }

  if (prop_log_interval) {
    RVSTRACE_
    timer_running.start(prop_log_interval);        // ticks continuously
  }

  // iterate through test array and invoke tests one by one
  RVSTRACE_
  do {
    RVSTRACE_
    for (auto it = test_array.begin(); brun && it != test_array.end(); ++it) {
      RVSTRACE_
      (*it)->do_transfer();

      // if log interval is zero, print current results immediately
      if (prop_log_interval == 0) {
        print_running_average(*it);
      }
      sleep(1);

      if (rvs::lp::Stopping()) {
        RVSTRACE_
        brun = false;
        break;
      }
    }
    RVSTRACE_

    iter -= step;

    // insert wait between runs if needed
    if (iter > 0 && gst_run_wait_ms > 0) {
      RVSTRACE_
      sleep(gst_run_wait_ms);
    }
  } while (brun && iter);

  RVSTRACE_
  timer_running.stop();
  timer_final.stop();

  print_final_average();

  RVSTRACE_
  return rvs::lp::Stopping() ? -1 : 0;
}

/**
 * @brief Execute test transfers all at once, for the
 * duration of the action.
 *
 * @return 0 - if successfull, non-zero otherwise
 *
 * */
int pqtaction::run_parallel() {
  // define timers
  rvs::timer<pqtaction> timer_running(&pqtaction::do_running_average, this);
  rvs::timer<pqtaction> timer_final(&pqtaction::do_final_average, this);

  // let the test run
  brun = true;

  // start all worker threads
  for (auto it = test_array.begin(); it != test_array.end(); ++it) {
    (*it)->start();
  }

  // start timers
  timer_final.start(gst_run_duration_ms, true);  // ticks only once
  timer_running.start(prop_log_interval);        // ticks continuously

  // wait for test to complete
  while (brun) {
    sleep(1);
    if (rvs::lp::Stopping()) {
      RVSTRACE_
      brun = false;
    }
  }

  timer_running.stop();
  timer_final.stop();

  // signal all worker threads to stop
  for (auto it = test_array.begin(); it != test_array.end(); ++it) {
    (*it)->stop();
  }
  sleep(10);

  // join all worker threads
  for (auto it = test_array.begin(); it != test_array.end(); ++it) {
    (*it)->join();
  }

  print_final_average();

  return rvs::lp::Stopping() ? -1 : 0;
}

/**
 * @brief Collect running average bandwidth data for all the tests and prints
 * them out every log_interval msecs.
 *
 * @return 0 - if successfull, non-zero otherwise
 *
 * */
int pqtaction::print_running_average() {
  for (auto it = test_array.begin(); brun && it != test_array.end(); ++it) {
    print_running_average(*it);
  }

  return 0;
}

/**
 * @brief Collect running average for this particular transfer.
 *
 * @param pWorker ptr to a pqtworker class
 *
 * @return 0 - if successfull, non-zero otherwise
 *
 * */
int pqtaction::print_running_average(pqtworker* pWorker) {
  int         src_node, dst_node;
  int         src_id, dst_id;
  bool        bidir;
  size_t      current_size;
  double      duration;
  std::string msg;
  char        buff[64];
  double      bandwidth;
  uint16_t    transfer_ix;
  uint16_t    transfer_num;

  // get running average
  pWorker->get_running_data(&src_node, &dst_node, &bidir,
                            &current_size, &duration);

  if (duration > 0) {
    bandwidth = current_size/duration/(1024*1024*1024);
    if (bidir) {
      bandwidth *=2;
    }
    snprintf( buff, sizeof(buff), "%.3f GBps", bandwidth);
  } else {
    // no running average in this iteration, try getting total so far
    // (do not reset final totals as this is just intermediate query)
    pWorker->get_final_data(&src_node, &dst_node, &bidir,
                            &current_size, &duration, false);
    if (duration > 0) {
      bandwidth = current_size/duration/(1024*1024*1024);
      if (bidir) {
        bandwidth *=2;
      }
      snprintf( buff, sizeof(buff), "%.3f GBps (*)", bandwidth);
    } else {
      // not transfers at all - print "pending"
      snprintf( buff, sizeof(buff), "(pending)");
    }
  }

  src_id = rvs::gpulist::GetGpuIdFromNodeId(src_node);
  dst_id = rvs::gpulist::GetGpuIdFromNodeId(dst_node);
  transfer_ix = pWorker->get_transfer_ix();
  transfer_num = pWorker->get_transfer_num();

  msg = "[" + action_name + "] p2p-bandwidth  ["
      + std::to_string(transfer_ix) + "/" + std::to_string(transfer_num)
      + "] " + std::to_string(src_id) + " " + std::to_string(dst_id)
      + "  bidirectional: " + std::string(bidir ? "true" : "false")
      + "  " + buff;
  rvs::lp::Log(msg, rvs::loginfo);
  if (bjson) {
    unsigned int sec;
    unsigned int usec;
    rvs::lp::get_ticks(&sec, &usec);
    json_rcqt_node = rvs::lp::LogRecordCreate(MODULE_NAME,
                            action_name.c_str(), rvs::loginfo, sec, usec);
    if (json_rcqt_node != NULL) {
      rvs::lp::AddString(json_rcqt_node,
                          "transfer_ix", std::to_string(transfer_ix));
      rvs::lp::AddString(json_rcqt_node,
                          "transfer_num", std::to_string(transfer_num));
      rvs::lp::AddString(json_rcqt_node, "src", std::to_string(src_id));
      rvs::lp::AddString(json_rcqt_node, "dst", std::to_string(dst_id));
      rvs::lp::AddString(json_rcqt_node, "p2p", "true");
      rvs::lp::AddString(json_rcqt_node, "bidirectional",
                          std::string(bidir ? "true" : "false"));
      rvs::lp::AddString(json_rcqt_node, "bandwidth (GBs)", buff);
      rvs::lp::LogRecordFlush(json_rcqt_node);
    }
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
int pqtaction::print_final_average() {
  int         src_node, dst_node;
  int         src_id, dst_id;
  bool        bidir;
  size_t      current_size;
  double      duration;
  std::string msg;
  double      bandwidth;
  char        buff[128];
  uint16_t    transfer_ix;
  uint16_t    transfer_num;

  for (auto it = test_array.begin(); it != test_array.end(); ++it) {
    (*it)->get_final_data(&src_node, &dst_node, &bidir,
                            &current_size, &duration);

    if (duration) {
      bandwidth = current_size/duration/(1024*1024*1024);
      if (bidir) {
        bandwidth *=2;
      }
      snprintf( buff, sizeof(buff), "%.3f GBps", bandwidth);
    } else {
      snprintf( buff, sizeof(buff), "(not measured)");
    }
    src_id = rvs::gpulist::GetGpuIdFromNodeId(src_node);
    dst_id = rvs::gpulist::GetGpuIdFromNodeId(dst_node);
    transfer_ix = (*it)->get_transfer_ix();
    transfer_num = (*it)->get_transfer_num();

    msg = "[" + action_name + "] p2p-bandwidth  ["
        + std::to_string(transfer_ix) + "/" + std::to_string(transfer_num)
        + "] " + std::to_string(src_id) + " " + std::to_string(dst_id)
        + "  bidirectional: " + std::string(bidir ? "true" : "false")
        + "  " + buff + "  duration: " + std::to_string(duration) + " sec";

    rvs::lp::Log(msg, rvs::logresults);
    if (bjson) {
      unsigned int sec;
      unsigned int usec;
      rvs::lp::get_ticks(&sec, &usec);
      json_rcqt_node = rvs::lp::LogRecordCreate(MODULE_NAME,
                              action_name.c_str(), rvs::logresults, sec, usec);
      if (json_rcqt_node != NULL) {
        rvs::lp::AddString(json_rcqt_node,
                            "transfer_ix", std::to_string(transfer_ix));
        rvs::lp::AddString(json_rcqt_node,
                            "transfer_num", std::to_string(transfer_num));
        rvs::lp::AddString(json_rcqt_node, "src", std::to_string(src_id));
        rvs::lp::AddString(json_rcqt_node, "dst", std::to_string(dst_id));
        rvs::lp::AddString(json_rcqt_node, "p2p", "true");
        rvs::lp::AddString(json_rcqt_node, "bidirectional",
                           std::string(bidir ? "true" : "false"));
        rvs::lp::AddString(json_rcqt_node, "bandwidth (GBps)", buff);
        rvs::lp::AddString(json_rcqt_node, "duration (sec)",
                           std::to_string(duration));
        rvs::lp::LogRecordFlush(json_rcqt_node);
      }
    }
    sleep(1);
  }

  return 0;
}

/**
 * @brief timer callback used to signal end of test
 *
 * timer callback used to signal end of test and to initiate
 * calculation of final average
 *
 * */
void pqtaction::do_final_average() {
  std::string msg;
  unsigned int sec;
  unsigned int usec;
  rvs::lp::get_ticks(&sec, &usec);

  msg = "[" + action_name + "] pqt in do_final_average";
  rvs::lp::Log(msg, rvs::logtrace, sec, usec);

  if (bjson) {
    json_rcqt_node = rvs::lp::LogRecordCreate(MODULE_NAME,
                            action_name.c_str(), rvs::logtrace, sec, usec);
    if (json_rcqt_node != NULL) {
      rvs::lp::AddString(json_rcqt_node, "message", "pqt in do_final_average");
      rvs::lp::LogRecordFlush(json_rcqt_node);
    }
  }

  brun = false;
}

/**
 * @brief timer callback used to signal end of log interval
 *
 * timer callback used to signal end of log interval and to initiate
 * calculation of moving average
 *
 * */
void pqtaction::do_running_average() {
  unsigned int sec;
  unsigned int usec;
  std::string msg;

  rvs::lp::get_ticks(&sec, &usec);
  msg = "[" + action_name + "] pqt in do_running_average";
  rvs::lp::Log(msg, rvs::logtrace, sec, usec);
  if (bjson) {
    json_rcqt_node = rvs::lp::LogRecordCreate(MODULE_NAME,
                            action_name.c_str(), rvs::logtrace, sec, usec);
    if (json_rcqt_node != NULL) {
      rvs::lp::AddString(json_rcqt_node,
                         "message",
                         "in do_running_average");
      rvs::lp::LogRecordFlush(json_rcqt_node);
    }
  }
  print_running_average();
}


