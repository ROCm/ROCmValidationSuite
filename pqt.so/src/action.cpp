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

#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>


#include "pci_caps.h"
#include "gpu_util.h"
#include "rvsloglp.h"
#include "rvshsa.h"

#include "rvs_module.h"
#include "worker.h"

//  static Worker* pworker;  //FIXME

//! Default constructor
action::action() {
}

//! Default destructor
action::~action() {
  property.clear();
}

int action ::run() {
  rvs::hsa* phsa = rvs::hsa::Get();

  double duration;
//  size_t size = 1024*1024;
  int sts;

  for (size_t i = 0; i < phsa->size_list.size(); i++) {
    sts = phsa->SendTraffic(4, 5, phsa->size_list[i], true, &duration);

    if (sts) {
      std::cerr << "RVS-PQT: internal error" << std::endl;
    }

    std::string msg = "pqt packet size: " + std::to_string(phsa->size_list[i])
      + "   throughput: "
      + std::to_string(phsa->size_list[i]/duration/(1024*1024*1024)*2);
    rvs::lp::Log(msg, rvs::logresults);
  }

  return 0;
}

/**
 * @brief Implements action functionality
 *
 * @return 0 - success. non-zero otherwise
 *
 * */
// int action::run(void) {
//   hsa_agent_t src_agent, dst_agent;
//   hsa_amd_memory_pool_t src_buff, dst_buff;
//   size_t src_max_size, dst_max_size;
// //   bool bidirectional;
//   string log_msg;
//
//   log("[PQT] in run()", rvs::logdebug);
//
//   // get all the agents
//   GetAgents();
//
//   rvs::lp::Log("[PQT] gpu_list.size() = " + std::to_string(gpu_list.size()), rvs::logdebug);
//   for (uint32_t i = 0; i < gpu_list.size(); i++) {
//     for (uint32_t j = 0; j < gpu_list.size(); j++) {
//       if (i == j) { continue; };
//       for (uint32_t n = 0; n < gpu_list[i].mem_pool_list.size(); n++) {
//         for (uint32_t m = 0; m < gpu_list[j].mem_pool_list.size(); m++) {
//           src_agent    = gpu_list[i].agent;
//           dst_agent    = gpu_list[j].agent;
//           src_buff     = gpu_list[i].mem_pool_list[n];
//           dst_buff     = gpu_list[j].mem_pool_list[m];
//           src_max_size = gpu_list[i].max_size_list[n];
//           dst_max_size = gpu_list[j].max_size_list[m];
//           // print
//           log_msg = "[PQT] src = " + gpu_list[i].agent_name + " / " + gpu_list[i].agent_device_type + " and dst = " + gpu_list[j].agent_name + " / " + gpu_list[j].agent_device_type;
//           log(log_msg.c_str(), rvs::logdebug);
//           // send p2p traffic (unidirectional)
//           send_p2p_traffic(src_agent, dst_agent, src_buff, dst_buff, false, src_max_size, dst_max_size, false);
//           // send p2p traffic (bidirectional)
//           send_p2p_traffic(src_agent, dst_agent, src_buff, dst_buff, true, src_max_size, dst_max_size, false);
//         }
//       }
//     }
//   }
//
//   rvs::lp::Log("[PQT] Test DONE", rvs::logdebug);
//
//   return 0;
// }
//

// TODO info
/**
 * @brief Process hsa_agent memory pool
 *
 * Functionality:
 *
 * Process agents memory pools
 *
 * @return hsa_status_t
 *
 * */
// void action::send_p2p_traffic(hsa_agent_t src_agent, hsa_agent_t dst_agent, hsa_amd_memory_pool_t src_buff, hsa_amd_memory_pool_t dst_buff, bool bidirectional, size_t src_max_size, size_t dst_max_size, bool validate) {
//   hsa_status_t status;
//   void* src_pool_pointer_fwd;
//   void* dst_pool_pointer_fwd;
//   void* src_pool_pointer_rev;
//   void* dst_pool_pointer_rev;
//   string log_msg;
//   hsa_signal_t signal_fwd, signal_rev;
//   char s_buff[256];
//   uint64_t total_size = 0;
//   double total_time = 0;
//   double curr_time;
//   double bandwidth;
//
//   log("[PQT] +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++", rvs::logdebug);
//   log("[PQT] send_p2p_traffic called ... ", rvs::logdebug);
//   log("[PQT] +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++", rvs::logdebug);
//
//   snprintf(s_buff, sizeof(s_buff), "%lX", src_agent.handle);
//   rvs::lp::Log(std::string("src_agent = ") + s_buff, rvs::logdebug);
//
//   snprintf(s_buff, sizeof(s_buff), "%lX", dst_agent.handle);
//   rvs::lp::Log(std::string("dst_agent = ") + s_buff, rvs::logdebug);
//
//   // Initialize size of buffer to equal the largest element of allocation
//   uint32_t size_len = size_list.size();
//
//   // Iterate through the differnt buffer sizes to
//   // compute the bandwidth as determined by copy
//   for (uint32_t idx = 0; idx < size_len; idx++) {
//
//     // This should not be happening
//     uint32_t curr_size = size_list[idx];
//     if (curr_size > src_max_size || curr_size > dst_max_size) {
//       break;
//     }
//
//     // print current size
//     log("[PQT] -----------------------------------------------------------------------------------", rvs::logdebug);
//     log_msg = "[PQT] send_p2p_traffic - curr_size = " + std::to_string(curr_size) + " Bytes";
//     log(log_msg.c_str(), rvs::logdebug);
//     log("[PQT] -----------------------------------------------------------------------------------", rvs::logdebug);
//
//     // Allocate buffers in src and dst pools
//     status = hsa_amd_memory_pool_allocate(src_buff, curr_size, 0, (void**)&src_pool_pointer_fwd);
//     print_hsa_status("[PQT] send_p2p_traffic - hsa_amd_memory_pool_allocate(SRC)", status);
//     snprintf(s_buff, sizeof(s_buff), "%p", src_pool_pointer_fwd);
//     rvs::lp::Log(std::string("src_pool_pointer_fwd = ") + s_buff, rvs::logdebug);
//
//     status = hsa_amd_memory_pool_allocate(dst_buff, curr_size, 0, (void**)&dst_pool_pointer_fwd);
//     print_hsa_status("[PQT] send_p2p_traffic - hsa_amd_memory_pool_allocate(DST)", status);
//     snprintf(s_buff, sizeof(s_buff), "%p", dst_pool_pointer_fwd);
//     rvs::lp::Log(std::string("dst_pool_pointer_fwd = ") + s_buff, rvs::logdebug);
//
//     if (bidirectional == true) {
//       status = hsa_amd_memory_pool_allocate(src_buff, curr_size, 0, (void**)&src_pool_pointer_rev);
//       print_hsa_status("[PQT] send_p2p_traffic BIDIRECTIONAL - hsa_amd_memory_pool_allocate(SRC)", status);
//       snprintf(s_buff, sizeof(s_buff), "%p", src_pool_pointer_rev);
//       rvs::lp::Log(std::string("src_pool_pointer_rev = ") + s_buff, rvs::logdebug);
//
//       status = hsa_amd_memory_pool_allocate(dst_buff, curr_size, 0, (void**)&dst_pool_pointer_rev);
//       print_hsa_status("[PQT] send_p2p_traffic BIDIRECTIONAL - hsa_amd_memory_pool_allocate(DST)", status);
//       snprintf(s_buff, sizeof(s_buff), "%p", dst_pool_pointer_rev);
//       rvs::lp::Log(std::string("dst_pool_pointer_rev = ") + s_buff, rvs::logdebug);
//     }
//
//
//     // Initialize buffers if validate is used
//     if (validate == true) {
//       std::memset(src_pool_pointer_fwd, 0xA5, curr_size);
//       std::memset(dst_pool_pointer_fwd, 0x5A, curr_size);
//
//       if (bidirectional == true) {
//         std::memset(src_pool_pointer_rev, 0xA5, curr_size);
//         std::memset(dst_pool_pointer_rev, 0x5A, curr_size);
//       }
//     }
//
//     // Create a signal to wait on copy operation
//     // hsa_signal_create(hsa_signal_value_t initial_value, uint32_t num_consumers, const hsa_agent_t *consumers, hsa_signal_t *signal)
//     status = hsa_signal_create(1, 0, NULL, &signal_fwd);
//     print_hsa_status("[PQT] send_p2p_traffic - hsa_signal_create()", status);
//
//     // get agent access
//     status = hsa_amd_agents_allow_access(1, &src_agent, NULL, dst_pool_pointer_fwd);
//     print_hsa_status("[PQT] send_p2p_traffic - hsa_amd_agents_allow_access(SRC)", status);
//
//     status = hsa_amd_agents_allow_access(1, &dst_agent, NULL, src_pool_pointer_fwd);
//     print_hsa_status("[PQT] send_p2p_traffic - hsa_amd_agents_allow_access(DST)", status);
//
//     // store signal
//     hsa_signal_store_relaxed(signal_fwd, 1);
//
//     if (bidirectional == true) {
//       status = hsa_signal_create(1, 0, NULL, &signal_rev);
//       print_hsa_status("[PQT] send_p2p_traffic BIDIRECTIONAL - hsa_signal_create()", status);
//
//       status = hsa_amd_agents_allow_access(1, &src_agent, NULL, dst_pool_pointer_rev);
//       print_hsa_status("[PQT] send_p2p_traffic BIDIRECTIONAL - hsa_amd_agents_allow_access(SRC)", status);
//
//       status = hsa_amd_agents_allow_access(1, &dst_agent, NULL, src_pool_pointer_rev);
//       print_hsa_status("[PQT] send_p2p_traffic BIDIRECTIONAL - hsa_amd_agents_allow_access(DST)", status);
//
//       hsa_signal_store_relaxed(signal_rev, 1);
//     }
//
//     // TODO need to check if the combination is unidirectonial or bidirectional
//
//     // Determine if accessibility to dst pool for src agent is not denied
//     hsa_amd_memory_pool_access_t access;
//     status = hsa_amd_agent_memory_pool_get_info(src_agent, dst_buff, HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &access);
//     print_hsa_status("[PQT] send_p2p_traffic - hsa_amd_agent_memory_pool_get_info(SRC->DST)", status);
//
//     if (access == HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) {
//       log_msg = "[PQT] send_p2p_traffic - HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED for SRC -> DST), so skip it ...";
//       log(log_msg.c_str(), rvs::logdebug);
//       return;
//     }
//
//     if (bidirectional == true) {
//       status = hsa_amd_agent_memory_pool_get_info(dst_agent, src_buff, HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &access);
//       print_hsa_status("[PQT] send_p2p_traffic BIDIRECTIONAL - hsa_amd_agent_memory_pool_get_info(DST->SRC)", status);
//
//       if (access == HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) {
//         log_msg = "[PQT] send_p2p_traffic BIDIRECTIONAL - HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED for DST -> SRC), so skip it ...";
//         log(log_msg.c_str(), rvs::logdebug);
//         return;
//       }
//     }
//
//     // Add current transfer size
//     total_size += curr_size;
//
//     // Copy from src into dst buffer
//     // hsa_amd_memory_async_copy(void* dst, hsa_agent_t dst_agent, const void* src, hsa_agent_t src_agent, size_t size, uint32_t num_dep_signals, const hsa_signal_t* dep_signals, hsa_signal_t completion_signal)
//     status = hsa_amd_memory_async_copy(dst_pool_pointer_fwd, dst_agent, src_pool_pointer_fwd, src_agent, curr_size, 0, NULL, signal_fwd);
//     print_hsa_status("[PQT] send_p2p_traffic - hsa_amd_memory_async_copy(SRC -> DST)", status);
//
//     if (bidirectional == true) {
//       status = hsa_amd_memory_async_copy(src_pool_pointer_rev, src_agent, dst_pool_pointer_rev, dst_agent, curr_size, 0, NULL, signal_rev);
//       print_hsa_status("[PQT] send_p2p_traffic BIDIRECTIONAL - hsa_amd_memory_async_copy(DST -> SRC)", status);
//     }
//
//     // Wait for the forward copy operation to complete
//     log_msg = "[PQT] send_p2p_traffic - hsa_signal_wait_acquire(SRC -> DST) before ...";
//     log(log_msg.c_str(), rvs::logdebug);
//     while (hsa_signal_wait_acquire(signal_fwd, HSA_SIGNAL_CONDITION_LT, 1, uint64_t(-1), HSA_WAIT_STATE_ACTIVE));
//     log_msg = "[PQT] send_p2p_traffic - hsa_signal_wait_acquire(SRC -> DST) after ...";
//     log(log_msg.c_str(), rvs::logdebug);
//
//     if (bidirectional == true) {
//       log_msg = "[PQT] send_p2p_traffic BIDIRECTIONAL - hsa_signal_wait_acquire(DST -> SRC) before ...";
//       log(log_msg.c_str(), rvs::logdebug);
//       while (hsa_signal_wait_acquire(signal_rev, HSA_SIGNAL_CONDITION_LT, 1, uint64_t(-1), HSA_WAIT_STATE_ACTIVE));
//       log_msg = "[PQT] send_p2p_traffic BIDIRECTIONAL - hsa_signal_wait_acquire(DST -> SRC) after ...";
//       log(log_msg.c_str(), rvs::logdebug);
//     }
//
//     curr_time = GetCopyTime(bidirectional, signal_fwd, signal_rev)/1000000000;
//     total_time += curr_time;
//     log_msg = "[PQT] send_p2p_traffic - total_time = " + std::to_string(total_time);
//     log(log_msg.c_str(), rvs::logdebug);
//
//     // convert to GB/s
//     if (bidirectional == true) {
//       curr_size *= 2;
//     }
//     bandwidth = (curr_size / curr_time);
//     bandwidth /= (1024*1024*1024);
//     log_msg = "[PQT] send_p2p_traffic - PARTIAL curr_size = " + std::to_string(curr_size) + " Bytes and curr_time = " + std::to_string(curr_time) + " bandwidth = " + std::to_string(bandwidth) + " GBytes/s";
//     log(log_msg.c_str(), rvs::logdebug);
//
//     // Compare output equals input
//     if (validate == true && memcmp(src_pool_pointer_fwd, dst_pool_pointer_fwd, curr_size) != 0) {
//       log("After copy is finished memory content is not the same (forward)", rvs::logerror);
//     }
//     // Compare output equals input
//     if (bidirectional == true && validate == true && memcmp(src_pool_pointer_rev, dst_pool_pointer_rev, curr_size) != 0) {
//       log("After copy is finished memory content is not the same (forward)", rvs::logerror);
//     }
//
//   }
//
//   // convert to GB/s
//   if (bidirectional == true) {
//     total_size *= 2;
//   }
//   bandwidth = (total_size / total_time);
//   bandwidth /= (1024*1024*1024);
//   log_msg = "[PQT] send_p2p_traffic - END total_size = " + std::to_string(total_size) + " Bytes and total_time = " + std::to_string(total_time) + " bandwidth = " + std::to_string(bandwidth) + " GBytes/s";
//   log(log_msg.c_str(), rvs::logdebug);
//
// }

