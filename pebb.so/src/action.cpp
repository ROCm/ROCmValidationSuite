/********************************************************************************
 * 
 * Copyright (c) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "hsa/hsa.h"

#include "include/pci_caps.h"
#include "include/gpu_util.h"
#include "include/rvs_util.h"
#include "include/rvsloglp.h"
#include "include/rvshsa.h"
#include "include/rvstimer.h"

#include "include/rvs_key_def.h"
#include "include/rvs_module.h"
#include "include/worker_b2b.h"

#define JSON_CREATE_NODE_ERROR "JSON cannot create node"
static constexpr auto MODULE_NAME = "pebb";
static constexpr auto MODULE_NAME_CAPS = "PEBB";
using std::string;
using std::vector;
//! Default constructor
pebb_action::pebb_action():link_type_string{} {
  bjson = false;
  b2b_block_size = 0;
  link_type = -1;
  module_name = MODULE_NAME;
}

//! Default destructor
pebb_action::~pebb_action() {
  property.clear();
}

/**
 * @brief reads all PEBB related configuration keys from
 * the module's properties collection
 * @return true if no fatal error occured, false otherwise
 */
bool pebb_action::get_all_pebb_config_keys(void) {
  string msg;
  int error;
  bool bsts = true;

  RVSTRACE_

    if (property_get("host_to_device", &prop_h2d, true)) {
      msg = "invalid 'host_to_device' key";
      rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
      bsts = false;
    }

  if (property_get("device_to_host", &prop_d2h, true)) {
    msg = "invalid 'device_to_host' key";
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
    bsts = false;
  }

  error = property_get_uint_list<uint32_t>(RVS_CONF_BLOCK_SIZE_KEY,
      YAML_DEVICE_PROP_DELIMITER,
      &block_size, &b_block_size_all);
  if (error == 1) {
    msg = "invalid '" + std::string(RVS_CONF_BLOCK_SIZE_KEY) + "' key";
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
    bsts = false;
  } else if (error == 2) {
    b_block_size_all = true;
    block_size.clear();
  }

  error = property_get_int<uint32_t>(RVS_CONF_B2B_BLOCK_SIZE_KEY, &b2b_block_size);
  if (error == 1) {
    msg = "invalid '" + std::string(RVS_CONF_B2B_BLOCK_SIZE_KEY) + "' key";
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
    bsts = false;
  }

  error = property_get_int<int>(RVS_CONF_LINK_TYPE_KEY, &link_type);
  if (error == 1) {
    msg = "invalid '" + std::string(RVS_CONF_LINK_TYPE_KEY) + "' key";
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
    bsts = false;
  }

  error = property_get_int<uint32_t>(RVS_CONF_HOT_CALLS_KEY, &hot_calls, DEFAULT_HOT_CALLS);
  if (error == 1) {
    msg = "invalid '" + std::string(RVS_CONF_HOT_CALLS_KEY) + "' key";
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
    bsts = false;
  }

  error = property_get_int<uint32_t>(RVS_CONF_WARM_CALLS_KEY, &warm_calls, DEFAULT_WARM_CALLS);
  if (error == 1) {
    msg = "invalid '" + std::string(RVS_CONF_WARM_CALLS_KEY) + "' key";
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
    bsts = false;
  }

  if (property_get(RVS_CONF_B2B_KEY, &b2b, DEFAULT_B2B)) {
    msg = "invalid '" + std::string(RVS_CONF_B2B_KEY) + "' key";
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
    bsts = false;
  }

  if(!hot_calls) {
    hot_calls = DEFAULT_HOT_CALLS;
  }

  if(!warm_calls) {
    warm_calls = DEFAULT_WARM_CALLS;
  }

  if( link_type == 2)
    link_type_string = "PCIe";
  else if(link_type == 4)
    link_type_string = "XGMI";

  return bsts;
}


/**
 * @brief Create thread objects based on action description in configuration
 * file.
 *
 * Threads are created but are not started. Execution, one by one of parallel,
 * depends on "parallel" key in configuration file. Pointers to created objects
 * are stored in "test_array" member
 *
 * @return 0 - if successfull, non-zero otherwise
 *
 * */
int pebb_action::create_threads() {
  std::string msg;
  std::vector<uint16_t> gpu_id;
  std::vector<uint16_t> gpu_idx;
  std::vector<uint16_t> gpu_device_id;
  uint16_t transfer_ix = 0;
  bool bmatch_found = false;

  RVSTRACE_
    gpu_get_all_gpu_id(&gpu_id);
  gpu_get_all_gpu_idx(&gpu_idx);
  gpu_get_all_device_id(&gpu_device_id);
  RVSTRACE_
    for (size_t i = 0; i < gpu_id.size(); i++) {
      RVSTRACE_
        if (property_device_id > 0) {
          RVSTRACE_
            if (property_device_id != gpu_device_id[i]) {
              RVSTRACE_
                continue;
            }
        }

      // filter out by listed sources
      RVSTRACE_
        if (!property_device_all && property_device.size()) {
          RVSTRACE_
            const auto it = std::find(property_device.cbegin(),
                property_device.cend(),
                gpu_id[i]);
          if (it == property_device.cend()) {
            RVSTRACE_
              continue;
          }
        }

      // filter out by listed sources
      if (!property_device_index_all && property_device_index.size()) {
        const auto it = std::find(property_device_index.cbegin(),
            property_device_index.cend(),
            gpu_idx[i]);
        if (it == property_device_index.cend()) {
          RVSTRACE_
            continue;
        }
      }

      uint16_t dstnode;
      int srcnode;

      RVSTRACE_
        for (uint cpu_index = 0;
            cpu_index < rvs::hsa::Get()->cpu_list.size();
            cpu_index++) {
          RVSTRACE_

            if (rvs::gpulist::gpu2node(gpu_id[i], &dstnode)) {
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
            }else{// if no connection either way, no point in adding to list
              continue;
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
            if (property_parallel && b2b_block_size > 0) {
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
            p->set_hot_calls(hot_calls);
            p->set_warm_calls(warm_calls);
            p->set_b2b(b2b);
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
        diag = "No devices match criteria from the test configuration";
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
      return -1;
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
int pebb_action::destroy_threads() {
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
int pebb_action::print_running_average() {
  for (auto it = test_array.begin(); brun && it != test_array.end(); ++it) {
    print_running_average(*it);
  }

  return 0;
}

/**
 *  * @brief logs a message to JSON
 *   * @param key info type
 *    * @param value message to log
 *     * @param log_level the level of log (e.g.: info, results, error)
 *      */
void* pebb_action::json_base_node(int log_level) {
  void *json_node = json_node_create(std::string(MODULE_NAME),
      action_name.c_str(), log_level);
  if(!json_node){
    // log error
    return nullptr;
  }	
  return json_node;
}

void pebb_action::json_add_kv(void *json_node, const std::string &key, const std::string &value){
  if (json_node) {
    rvs::lp::AddString(json_node, key, value);
  }
}

void pebb_action::json_to_file(void *json_node,int log_level){
  if (json_node)
    rvs::lp::LogRecordFlush(json_node, log_level);
}

void pebb_action::log_json_bandwidth(std::string srcnode, std::string dstnode,
    int log_level, std::string bandwidth){

  if(bjson){
    void *json_node = json_base_node(log_level);
    json_add_kv(json_node, "srccpu", srcnode);
    json_add_kv(json_node, "dstgpu", dstnode);
    json_add_kv(json_node, "intf", link_type_string);
    json_add_kv(json_node, "throughput", bandwidth.empty() ? "NA" : bandwidth);
    json_add_kv(json_node, "pass", "true");
    json_to_file(json_node, log_level);
  }
}

/**
 * @brief Collect running average for this particular transfer.
 *
 * @param pWorker ptr to a pebbworker class
 *
 * @return 0 - if successfull, non-zero otherwise
 *
 * */
int pebb_action::print_running_average(pebbworker* pWorker) {
  uint16_t    src_node, dst_node;
  uint16_t    dst_id;
  bool        bidir;
  size_t      current_size;
  double      duration;
  std::string msg;
  char        buff[64];
  double      bandwidth;
  uint16_t    transfer_ix;
  uint16_t    transfer_num;
  char        transfer_buff[8];
  char        dstgpuid_buff[12];

  RVSTRACE_
    // get running average
    pWorker->get_running_data(&src_node, &dst_node, &bidir,
        &current_size, &duration);

  if (duration > 0) {
    RVSTRACE_
      bandwidth = current_size/duration/1000/1000/1000;
    if (bidir) {
      RVSTRACE_
        bandwidth *=2;
    }
    snprintf( buff, sizeof(buff), "%.3f GBps", bandwidth);
  } else {
    RVSTRACE_
      // no running average in this iteration, try getting total so far
      // (do not reset final totals as this is just intermediate query)
      pWorker->get_final_data(&src_node, &dst_node, &bidir,
          &current_size, &duration, false);
    RVSTRACE_
      bandwidth = current_size/duration/1000/1000/1000;
    if (bidir) {
      RVSTRACE_
        bandwidth *=2;
    }
    snprintf( buff, sizeof(buff), "%.3f GBps (*)", bandwidth);
  }

  RVSTRACE_
    if (rvs::gpulist::node2gpu(dst_node, &dst_id)) {
      RVSTRACE_
        std::string msg = "could not find GPU id for node " +
        std::to_string(dst_node);
      rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
      return -1;
    }
  RVSTRACE_

    std::string pci_bdf;
  if (rvs::gpulist::node2bdf(dst_node, pci_bdf)) {
    RVSTRACE_
      std::string msg = "could not find PCI BDF for node " +
      std::to_string(dst_node);
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
    return -1;
  }
  RVSTRACE_

    transfer_ix = pWorker->get_transfer_ix();
  transfer_num = pWorker->get_transfer_num();

  snprintf(transfer_buff, sizeof(transfer_buff), "%2d", transfer_ix);
  snprintf(dstgpuid_buff, sizeof(dstgpuid_buff), "%5d", dst_id);

  msg = "[" + action_name + "] pcie-bandwidth ["
    + transfer_buff + "/" + std::to_string(transfer_num) + "]"
    + " [CPU:: " + std::to_string(src_node) + "]"
    + " [GPU:: " + std::to_string(dst_node) + " - " + dstgpuid_buff + " - " + pci_bdf + "]"
    + " h2d::" + (prop_h2d ? "true" : "false")
    + " d2h::" + (prop_d2h ? "true" : "false") + " "
    + buff;

  rvs::lp::Log(msg, rvs::loginfo);

  log_json_bandwidth(std::to_string(src_node), std::to_string(dst_id),rvs::loginfo, buff);
  RVSTRACE_
    return 0;
}

/**
 * @brief Collect bandwidth totals for all the tests and prints
 * them on cout at the end of action execution
 *
 * @return 0 - if successfull, non-zero otherwise
 *
 * */
int pebb_action::print_final_average() {
  bandwidth   bw;
  uint16_t    src_node, dst_node;
  uint16_t    dst_id;
  bool        bidir;
  string      str; 
  size_t      current_size;
  double      duration;
  std::string msg;
  double      bandwidth;
  char        buff[128];
  char        transfer_buff[8];
  char        dstgpuid_buff[12];
  uint16_t    transfer_ix;
  uint16_t    transfer_num;
  rvs::action_result_t result;

  for (auto it = test_array.begin(); it != test_array.end(); ++it) {
    RVSTRACE_
      (*it)->get_final_data(&src_node, &dst_node, &bidir,
          &current_size, &duration);

    if (duration) {
      RVSTRACE_
        bandwidth = current_size/duration/1000/1000/1000;
      if (bidir) {
        RVSTRACE_
          bandwidth *=2;
      }
      snprintf( buff, sizeof(buff), "%.3f GBps", bandwidth);
    } else {
      RVSTRACE_
        snprintf( buff, sizeof(buff), "(not measured)");
    }

    RVSTRACE_
      if (rvs::gpulist::node2gpu(dst_node, &dst_id)) {
        RVSTRACE_
          std::string msg = "could not find GPU id for node " +
          std::to_string(dst_node);
        rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
        return -1;
      }
    RVSTRACE_
      transfer_ix = (*it)->get_transfer_ix();
    transfer_num = (*it)->get_transfer_num();

    RVSTRACE_
      std::string pci_bdf;
    if (rvs::gpulist::node2bdf(dst_node, pci_bdf)) {
      RVSTRACE_
        std::string msg = "could not find PCI BDF for node " +
        std::to_string(dst_node);
      rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
      return -1;
    }
    RVSTRACE_

      snprintf(transfer_buff, sizeof(transfer_buff), "%2d", transfer_ix);
    snprintf(dstgpuid_buff, sizeof(dstgpuid_buff), "%5d", dst_id);

    msg = "[" + action_name + "] pcie-bandwidth ["
      + transfer_buff + "/" + std::to_string(transfer_num) + "]"
      + " [CPU:: " + std::to_string(src_node) + "]"
      + " [GPU:: " + std::to_string(dst_node) + " - " + dstgpuid_buff + " - " + pci_bdf + "]"
      + " h2d::" + (prop_h2d ? "true" : "false")
      + " d2h::" + (prop_d2h ? "true" : "false")
      + " " + buff
      + " duration: " + std::to_string(duration) + " secs";

    rvs::lp::Log(msg, rvs::logresults);

    bw.finalBandwith = buff;
    bw.GPUId = dst_id;
    bw.CPUId = src_node;

    resultBandwidth.push_back(bw);
    log_json_bandwidth(std::to_string(src_node), std::to_string(dst_id), rvs::logresults, buff);

    result.state = rvs::actionstate::ACTION_RUNNING;
    result.status = rvs::actionstatus::ACTION_SUCCESS;
    result.output = msg.c_str();
    action_callback(&result);

    RVSTRACE_
  }
  RVSTRACE_
    return 0;
}

/**
 * @brief timer callback used to signal end of test
 *
 * timer callback used to signal end of test and to initiate
 * calculation of final average
 *
 * */
void pebb_action::do_final_average() {
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
void pebb_action::do_running_average() {
  unsigned int sec;
  unsigned int usec;
  std::string msg;

  if (!brun) {
    return;
  }

  rvs::lp::get_ticks(&sec, &usec);
  msg = "[" + action_name + "] pebb in do_running_average";
  rvs::lp::Log(msg, rvs::logtrace, sec, usec);
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
int pebb_action::print_link_info(int SrcNode, int DstNode, int DstGpuID,
    uint32_t Distance,
    const std::vector<rvs::linkinfo_t>& arrLinkInfo,
    bool bReverse) {
  RVSTRACE_
    std::string msg;
  rvs::action_result_t result;
  std::string pci_bdf;
  char dstgpuid_buff[8];

  if (rvs::gpulist::node2bdf(DstNode, pci_bdf)) {
    RVSTRACE_
      std::string msg = "could not find PCI BDF for node " +
      std::to_string(DstNode);
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
    return -1;
  }

  snprintf(dstgpuid_buff, sizeof(dstgpuid_buff), "%5d", DstGpuID);

  msg = "[" + action_name + "] pcie-bandwidth "
    + "[CPU:: " +  std::to_string(SrcNode) + "] "
    + "[GPU:: " + std::to_string(DstNode)
    + " - " + dstgpuid_buff + " - " + pci_bdf + "]";

  if (Distance == rvs::hsa::NO_CONN) {
    msg += " distance:-1";
  } else {
    msg += " distance:" + std::to_string(Distance);
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
  //log_json_bandwidth(std::to_string(SrcNode), std::to_string(DstGpuID),rvs::logresults);

  result.state = rvs::actionstate::ACTION_RUNNING;
  result.status = rvs::actionstatus::ACTION_SUCCESS;
  result.output = msg.c_str();
  action_callback(&result);

  return 0;
}

