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

#include "hsa/hsa.h"

#include "pci_caps.h"
#include "gpu_util.h"
#include "rvs_util.h"
#include "rvsloglp.h"
#include "rvshsa.h"
#include "rvstimer.h"

#include "rvs_key_def.h"
#include "rvs_module.h"
#include "worker_b2b.h"

#define MODULE_NAME "pebb"
#define MODULE_NAME_CAPS "PEBB"
#define JSON_CREATE_NODE_ERROR "JSON cannot create node"

using std::string;
using std::vector;

//! Default constructor
pebbaction::pebbaction() {
  prop_deviceid = -1;
  prop_device_id_filtering = false;
  bjson = false;
  b2b_block_size = 0;
  link_type = -1;
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
 *  @brief reads the module's properties collection to see if
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
 * @brief reads all PQT related configuration keys from
 * the module's properties collection
 * @return true if no fatal error occured, false otherwise
 */
bool pebbaction::get_all_pebb_config_keys(void) {;
  string msg;
  int error;


  RVSTRACE_
  prop_log_interval = property_get_log_interval(&error);
  if (error == 1) {
    msg = "invalid '" + std::string(RVS_CONF_LOG_INTERVAL_KEY) + "' key";
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
    return false;
  } else if (error == 2) {
    prop_log_interval = DEFAULT_LOG_INTERVAL;
  }

  property_get_h2d();
  property_get_d2h();

  property_get_uint_list(RVS_CONF_BLOCK_SIZE_KEY, YAML_DEVICE_PROP_DELIMITER,
                         &block_size, &b_block_size_all, &error);
  if (error == 1) {
      msg = "invalid '" + std::string(RVS_CONF_BLOCK_SIZE_KEY) + "' key";
      rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
      return false;
  } else if (error == 2) {
    b_block_size_all = true;
    block_size.clear();
  }

  b2b_block_size = property_get_b2b_size(&error);
  if (error == 1) {
    msg = "invalid '" + std::string(RVS_CONF_B2B_BLOCK_SIZE_KEY) + "' key";
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
    return false;
  }

  link_type = property_get_link_type(&error);
  if (error == 1) {
    msg = "invalid '" + std::string(RVS_CONF_LINK_TYPE_KEY) + "' key";
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
    return false;
  }

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

  RVSTRACE_
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
      msg = "invalid 'device' key value " + sdev;
      rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
      return false;
    }
  } else {
    msg = "key 'device' was not found";
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
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

      msg = "invalid 'deviceid' key value " + std::string(sdevid);
      rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);

      return false;
    }
  } else {
    prop_device_id_filtering = false;
  }

  // get the other action/GST related properties
  rvs::actionbase::property_get_run_parallel(&error);
  if (error == 1) {
    msg = "invalid '" + std::string(RVS_CONF_PARALLEL_KEY) +
    "' key value";
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
    return false;
  }

  //! number of GST stress test iterations to run
  uint64_t run_count = 1;
  //! stress test run delay
  uint64_t run_wait_ms = 0;
  //! stress test run duration
  uint64_t run_duration_ms = 0;
  
  error = property_get_int<uint64_t>(RVS_CONF_COUNT_KEY, &run_count);
  if (error != 0) {
      msg ="invalid '" + std::string(RVS_CONF_COUNT_KEY) +"' key value";
      rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
      return false;
  }

  error = property_get_int<uint64_t>(RVS_CONF_WAIT_KEY, &run_wait_ms);
  if (error != 0) {
      msg = "invalid '" + std::string(RVS_CONF_WAIT_KEY) + "' key value";
      rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
      return false;
  }

  error = property_get_int<uint64_t>(RVS_CONF_DURATION_KEY, &run_duration_ms);
  if (error != 0) {
    msg = "invalid '" + std::string(RVS_CONF_DURATION_KEY) +
    "' key value";
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
    return false;
  } else if (run_duration_ms == 0) {
    msg = "'" + std::string(RVS_CONF_DURATION_KEY) +
    "' key must be greater then zero";
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
    return false;
  }

  property_get_log_level(&error);
  if (error == 1) {
    msg = "invalid logging level value";
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
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
int pebbaction::create_threads() {
  std::string msg;
  std::vector<uint16_t> gpu_id;
  std::vector<uint16_t> gpu_device_id;
  uint16_t transfer_ix = 0;
  bool bmatch_found = false;

  RVSTRACE_
  gpu_get_all_gpu_id(&gpu_id);
  gpu_get_all_device_id(&gpu_device_id);

  RVSTRACE_
  for (size_t i = 0; i < gpu_id.size(); i++) {
    RVSTRACE_
    if (prop_device_id_filtering) {
      RVSTRACE_
      if (prop_deviceid != gpu_device_id[i]) {
        RVSTRACE_
        continue;
      }
    }

    // filter out by listed sources
    RVSTRACE_
    if (!prop_device_all_selected) {
      RVSTRACE_
      const auto it = std::find(device_prop_gpu_id_list.cbegin(),
                                device_prop_gpu_id_list.cend(),
                                std::to_string(gpu_id[i]));
      if (it == device_prop_gpu_id_list.cend()) {
        RVSTRACE_
        continue;
      }
    }

    int dstnode;
    int srcnode;

    RVSTRACE_
    for (uint cpu_index = 0;
         cpu_index < rvs::hsa::Get()->cpu_list.size();
         cpu_index++) {
      RVSTRACE_

      dstnode = rvs::gpulist::GetNodeIdFromGpuId(gpu_id[i]);
      if (dstnode < 0) {
        RVSTRACE_
        msg = "no node found for destination GPU ID "
          + std::to_string(gpu_id[i]);
        rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
        return -1;
      }
      RVSTRACE_
      srcnode = rvs::hsa::Get()->cpu_list[cpu_index].node;

      // get link info regardless of peer status (just in case...)
      uint32_t distance = 0;
      bool b_reverse = false;

      std::vector<rvs::linkinfo_t> arr_linkinfo;
      rvs::hsa::Get()->GetLinkInfo(srcnode, dstnode,
                                         &distance, &arr_linkinfo);
      if (distance == rvs::hsa::NO_CONN) {
        RVSTRACE_
        rvs::hsa::Get()->GetLinkInfo(dstnode, srcnode,
                                    &distance, &arr_linkinfo);
        if (distance != rvs::hsa::NO_CONN) {
          RVSTRACE_
          // there is a path if transfer is initiated by
          // destination agent:
          b_reverse = true;
        }
      }

      // if link type is specified, check that it matches
      if (!rvs::hsa::check_link_type(arr_linkinfo, link_type))
        continue;

      bmatch_found = true;
      transfer_ix += 1;

      print_link_info(srcnode, dstnode, gpu_id[i],
                      distance, arr_linkinfo, b_reverse);

      // if GPUs are peers, create transaction for them
      if (rvs::hsa::Get()->GetPeerStatus(srcnode, dstnode)) {
        RVSTRACE_
        pebbworker* p = nullptr;
        if (gst_runs_parallel && b2b_block_size > 0) {
          RVSTRACE_
          pebbworker_b2b* pb2b = new pebbworker_b2b;
          if (pb2b == nullptr) {
            RVSTRACE_
            msg = "internal error";
            rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
            return -1;
          }
          pb2b->initialize(srcnode, dstnode,
                           prop_h2d, prop_d2h, b2b_block_size);
          p = pb2b;
        } else {
          RVSTRACE_
          p = new pebbworker;
          if (p == nullptr) {
            RVSTRACE_
            msg = "internal error";
            rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
            return -1;
          }
          p->initialize(srcnode, dstnode, prop_h2d, prop_d2h);
        }
        RVSTRACE_
        p->set_name(action_name);
        p->set_stop_name(action_name);
        p->set_transfer_ix(transfer_ix);
        p->set_block_sizes(block_size);
        p->set_loglevel(property_log_level);
        test_array.push_back(p);
      }
    }
  }

  RVSTRACE_
  if (test_array.size() < 1) {
    std::string diag;
    if (bmatch_found) {
      diag = "No peers found";
    } else {
      diag = "No devices match criteria from the test configuation";
    }
    msg = "[" + action_name + "] pcie-bandwidth  " + diag;
    rvs::lp::Log(msg, rvs::logerror);
    if (bjson) {
      unsigned int sec;
      unsigned int usec;
      rvs::lp::get_ticks(&sec, &usec);
      void* pjson = rvs::lp::LogRecordCreate("pcie-bandwidth",
                              action_name.c_str(), rvs::logerror, sec, usec);
      if (pjson != NULL) {
        rvs::lp::AddString(pjson,
          "message",
          diag);
        rvs::lp::LogRecordFlush(pjson);
      }
    }
    return 0;
  }

  for (auto it = test_array.begin(); it != test_array.end(); ++it) {
    RVSTRACE_
    (*it)->set_transfer_num(test_array.size());
  }

  RVSTRACE_
  return 0;
}

/**
 * @brief Delete test thread objects at the end of action execution
 *
 * @return 0 - if successfull, non-zero otherwise
 *
 * */
int pebbaction::destroy_threads() {
  RVSTRACE_
  for (auto it = test_array.begin(); it != test_array.end(); ++it) {
    (*it)->set_stop_name(action_name);
    (*it)->stop();
    delete *it;
  }
  return 0;
}

/**
 * @brief Collect running average bandwidth data for all the tests and prints
 * them out.
 *
 * @return 0 - if successfull, non-zero otherwise
 *
 * */
int pebbaction::print_running_average() {
  for (auto it = test_array.begin(); brun && it != test_array.end(); ++it) {
    print_running_average(*it);
  }

  return 0;
}

/**
 * @brief Collect running average for this particular transfer.
 *
 * @param pWorker ptr to a pebbworker class
 *
 * @return 0 - if successfull, non-zero otherwise
 *
 * */
int pebbaction::print_running_average(pebbworker* pWorker) {
  int         src_node, dst_node;
  int         dst_id;
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

  dst_id = rvs::gpulist::GetGpuIdFromNodeId(dst_node);
  transfer_ix = pWorker->get_transfer_ix();
  transfer_num = pWorker->get_transfer_num();

  msg = "[" + action_name + "] pcie-bandwidth  ["
      + std::to_string(transfer_ix) + "/" + std::to_string(transfer_num)
      + "] "
      + std::to_string(src_node) + " " + std::to_string(dst_id)
      + "  h2d: " + (prop_h2d ? "true" : "false")
      + "  d2h: " + (prop_d2h ? "true" : "false") + "  "
      + buff;

  rvs::lp::Log(msg, rvs::loginfo);

  if (bjson) {
    unsigned int sec;
    unsigned int usec;
    rvs::lp::get_ticks(&sec, &usec);
    void* pjson = rvs::lp::LogRecordCreate(MODULE_NAME,
                        action_name.c_str(), rvs::loginfo, sec, usec);
    if (pjson != NULL) {
      rvs::lp::AddString(pjson,
                          "transfer_ix", std::to_string(transfer_ix));
      rvs::lp::AddString(pjson,
                          "transfer_num", std::to_string(transfer_num));
      rvs::lp::AddString(pjson, "src", std::to_string(src_node));
      rvs::lp::AddString(pjson, "dst", std::to_string(dst_id));
      rvs::lp::AddString(pjson, "pcie-bandwidth (GBps)", buff);
      rvs::lp::LogRecordFlush(pjson);
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
int pebbaction::print_final_average() {
  int         src_node, dst_node;
  int         dst_id;
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

    dst_id = rvs::gpulist::GetGpuIdFromNodeId(dst_node);
    transfer_ix = (*it)->get_transfer_ix();
    transfer_num = (*it)->get_transfer_num();

    msg = "[" + action_name + "] pcie-bandwidth  ["
        + std::to_string(transfer_ix) + "/" + std::to_string(transfer_num)
        + "] "
        + std::to_string(src_node) + " " + std::to_string(dst_id)
        + "  h2d: " + (prop_h2d ? "true" : "false")
        + "  d2h: " + (prop_d2h ? "true" : "false")
        + "  " + buff
        + "  duration: " + std::to_string(duration) + " sec";

    rvs::lp::Log(msg, rvs::logresults);
    if (bjson) {
      unsigned int sec;
      unsigned int usec;
      rvs::lp::get_ticks(&sec, &usec);
      void* pjson = rvs::lp::LogRecordCreate(MODULE_NAME,
                          action_name.c_str(), rvs::logresults, sec, usec);
      if (pjson != NULL) {
        rvs::lp::AddString(pjson,
                            "transfer_ix", std::to_string(transfer_ix));
        rvs::lp::AddString(pjson,
                            "transfer_num", std::to_string(transfer_num));
        rvs::lp::AddString(pjson, "src", std::to_string(src_node));
        rvs::lp::AddString(pjson, "dst", std::to_string(dst_id));
        rvs::lp::AddString(pjson, "bandwidth (GBps)", buff);
        rvs::lp::AddString(pjson, "duration (sec)",
                           std::to_string(duration));
        rvs::lp::LogRecordFlush(pjson);
      }
    }
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
void pebbaction::do_final_average() {
  if (property_log_level >= rvs::logtrace) {
    std::string msg;
    unsigned int sec;
    unsigned int usec;
    rvs::lp::get_ticks(&sec, &usec);

    msg = "[" + action_name + "] pebb in do_final_average";
    rvs::lp::Log(msg, rvs::logtrace, sec, usec);

    if (bjson) {
      void* pjson = rvs::lp::LogRecordCreate(MODULE_NAME,
                              action_name.c_str(), rvs::logtrace, sec, usec);
      if (pjson != NULL) {
        rvs::lp::AddString(pjson, "message", "pebb in do_final_average");
        rvs::lp::LogRecordFlush(pjson);
      }
    }
  }

  // signal main thread to stop
  brun = false;

  // signal worker threads to stop
  for (auto it = test_array.begin(); it != test_array.end(); ++it) {
    (*it)->stop();
  }
}

/**
 * @brief timer callback used to signal end of log interval
 *
 * timer callback used to signal end of log interval and to initiate
 * calculation of moving average
 *
 * */
void pebbaction::do_running_average() {
  unsigned int sec;
  unsigned int usec;
  std::string msg;

  if (!brun) {
    return;
  }

  rvs::lp::get_ticks(&sec, &usec);
  msg = "[" + action_name + "] pebb in do_running_average";
  rvs::lp::Log(msg, rvs::logtrace, sec, usec);
  if (bjson) {
    void* pjson = rvs::lp::LogRecordCreate(MODULE_NAME,
                            action_name.c_str(), rvs::logtrace, sec, usec);
    if (pjson != NULL) {
      rvs::lp::AddString(pjson,
                         "message",
                         "in do_running_average");
      rvs::lp::LogRecordFlush(pjson);
    }
  }
  print_running_average();
}

/**
 * @brief Print link information.
 *
 * Print link information as list of "hops" between two NUMA nodes.
 * Each hop is in format \<link_type\>:\<distance\>
 *
 * @param SrcNode starting NUMA node
 * @param DstNode ending NUMA node
 * @param DstGpuID destination GPU id
 * @param Distance NUMA distance between the twonodes
 * @param arrLinkInfo array of hop infos
 * @param bReverse 'true' if info is for DST to SRC direction
 *
 * @return 0 - if successfull, non-zero otherwise
 *
 * */
int pebbaction::print_link_info(int SrcNode, int DstNode, int DstGpuID,
                      uint32_t Distance,
                      const std::vector<rvs::linkinfo_t>& arrLinkInfo,
                      bool bReverse) {
  RVSTRACE_
  std::string msg;

  msg = "[" + action_name + "] pcie-bandwidth "
      + std::to_string(SrcNode)
      + " " + std::to_string(DstNode)
      + " " + std::to_string(DstGpuID);
  if (Distance == rvs::hsa::NO_CONN) {
    msg += "  distance:-1";
  } else {
    msg += "  distance:" + std::to_string(Distance);
  }
  // iterate through individual hops
  for (auto it = arrLinkInfo.begin(); it != arrLinkInfo.end(); it++) {
    msg += " " + it->strtype + ":";
    if (it->distance == rvs::hsa::NO_CONN) {
      msg += "-1";
    } else {
      msg +=std::to_string(it->distance);
    }
  }
  if (bReverse) {
    msg += " (R)";
  }

  rvs::lp::Log(msg, rvs::logresults);

  if (bjson) {
    unsigned int sec;
    unsigned int usec;
    rvs::lp::get_ticks(&sec, &usec);
    void* pjson = rvs::lp::LogRecordCreate(MODULE_NAME,
                        action_name.c_str(), rvs::logresults, sec, usec);
    if (pjson != NULL) {
      RVSTRACE_
      rvs::lp::AddString(pjson, "Src", std::to_string(SrcNode));
      rvs::lp::AddString(pjson, "Dst", std::to_string(DstNode));
      rvs::lp::AddString(pjson, "GPU", std::to_string(DstGpuID));
      if (Distance == rvs::hsa::NO_CONN) {
          rvs::lp::AddInt(pjson, "distance", -1);
      } else {
          rvs::lp::AddInt(pjson, "distance", Distance);
      }
      if (bReverse) {
        rvs::lp::AddInt(pjson, "Reverse", 1);
      } else {
        rvs::lp::AddInt(pjson, "Reverse", 0);
      }

      void* phops = rvs::lp::CreateNode(pjson, "hops");
      rvs::lp::AddNode(pjson, phops);

      // iterate through individual hops
      for (uint i = 0; i < arrLinkInfo.size(); i++) {
        char sbuff[64];
        snprintf(sbuff, sizeof(sbuff), "hop%d", i);
        void* phop = rvs::lp::CreateNode(phops, sbuff);
        rvs::lp::AddString(phop, "type", arrLinkInfo[i].strtype);
        if (arrLinkInfo[i].distance == rvs::hsa::NO_CONN) {
          rvs::lp::AddInt(phop, "distance", -1);
        } else {
          rvs::lp::AddInt(phop, "distance", arrLinkInfo[i].distance);
        }
        rvs::lp::AddNode(phops, phop);
      }
      rvs::lp::LogRecordFlush(pjson);
    }
  }

  return 0;
}
