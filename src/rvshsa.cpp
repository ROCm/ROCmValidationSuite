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
#include "include/rvshsa.h"

#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

#include "include/rvs_util.h"
#include "include/rvsloglp.h"

// ptr to singletone instance
rvs::hsa* rvs::hsa::pDsc;
const uint32_t rvs::hsa::NO_CONN;

/**
 * @brief Initialize RVS HSA wrapper
 *
 * */
void rvs::hsa::Init() {
  if (pDsc == nullptr) {
    pDsc = new rvs::hsa();
    pDsc->InitAgents();
  }
}

/**
 * @brief Terminate RVS HSA wrapper
 *
 * */
void rvs::hsa::Terminate() {
  if (pDsc != nullptr) {
    delete pDsc;
    pDsc = nullptr;
  }
}

/**
 * @brief Fetch RVS HSA wrapper
 * @return pointer to RVS HSA singleton
 *
 * */
rvs::hsa* rvs::hsa::Get() {
  return pDsc;
}

//! Default constructor
rvs::hsa::hsa() {
}

//! Default destructor
rvs::hsa::~hsa() {
}


/**
 * @brief helper method used in debbuging
 *
 */
void rvs::hsa::print_hsa_status(const char* file,
                                int line,
                                const char* function,
                                const char* msg,
                                hsa_status_t st) {
  if (st == HSA_STATUS_SUCCESS) {
    return;
  }
  string log_msg = msg;
  log_msg += "  " + std::string(file)+ "  " + function + ":"
  + std::to_string(line);
  rvs::lp::Log(log_msg, rvs::logdebug);
  print_hsa_status(log_msg.c_str(), st);
}
/**
 * @brief helper method used in debbuging
 *
 */
void rvs::hsa::print_hsa_status(const char* file,
                                int line,
                                const char* function,
                                hsa_status_t st) {
  print_hsa_status(file, line, function, "", st);
}

/**
 * @brief helper method used in debbuging
 *
 */
void rvs::hsa::print_hsa_status(const char* message, hsa_status_t st) {
  // skip successfull messages
  if (st == HSA_STATUS_SUCCESS) {
    return;
  }
  string log_msg = message;
  switch (st) {
    case HSA_STATUS_SUCCESS : {
      log_msg += " The function has been executed successfully.";
      break;
    };
    case HSA_STATUS_INFO_BREAK : {
      log_msg += " A traversal over a list of elements has been "
      "interrupted by the application before completing.";
      break;
    };
    case HSA_STATUS_ERROR : {
      log_msg += " A generic error has occurred.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_ARGUMENT : {
      log_msg += " One of the actual arguments does not meet a "
      "precondition stated in the documentation of the corresponding formal "
      "argument.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_QUEUE_CREATION : {
      log_msg += " The requested queue creation is not valid.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_ALLOCATION : {
      log_msg += " The requested allocation is not valid.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_AGENT : {
      log_msg += " The agent is invalid.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_REGION : {
      log_msg += " The memory region is invalid.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_SIGNAL : {
      log_msg += " The signal is invalid.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_QUEUE : {
      log_msg += " The queue is invalid.";
      break;
    };
    case HSA_STATUS_ERROR_OUT_OF_RESOURCES : {
      log_msg += " The HSA runtime failed to allocate the necessary "
      "resources. This error may also occur when the HSA runtime needs to "
      "spawn threads or create internal OS-specific events.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_PACKET_FORMAT : {
      log_msg += " The AQL packet is malformed.";
      break;
    };
    case HSA_STATUS_ERROR_RESOURCE_FREE : {
      log_msg +=
      " An error has been detected while releasing a resource.";
      break;
    };
    case HSA_STATUS_ERROR_NOT_INITIALIZED : {
      log_msg +=
      " An API other than ::hsa_init has been invoked while the reference "
      "count of the HSA runtime is 0.";
      break;
    };
    case HSA_STATUS_ERROR_REFCOUNT_OVERFLOW : {
      log_msg += " The maximum reference count for the object has "
      "been reached.";
      break;
    };
    case HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS : {
      log_msg += " The arguments passed to a functions are not "
      "compatible.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_INDEX : {
      log_msg += " The index is invalid.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_ISA : {
      log_msg += " The instruction set architecture is invalid.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_ISA_NAME : {
      log_msg += " The instruction set architecture name is invalid.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_CODE_OBJECT : {
      log_msg += " The code object is invalid.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_EXECUTABLE : {
      log_msg += " The executable is invalid.";
      break;
    };
    case HSA_STATUS_ERROR_FROZEN_EXECUTABLE : {
      log_msg += " The executable is frozen.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_SYMBOL_NAME : {
      log_msg += " There is no symbol with the given name.";
      break;
    };
    case HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED : {
      log_msg += " The variable is already defined.";
      break;
    };
    case HSA_STATUS_ERROR_VARIABLE_UNDEFINED : {
      log_msg += " The variable is undefined.";
      break;
    };
    case HSA_STATUS_ERROR_EXCEPTION : {
      log_msg +=
      " An HSAIL operation resulted on a hardware exception.";
      break;
    };
    default : {
      log_msg += " Unknown error.";
      break;
    }
  }
  rvs::lp::Log(log_msg, rvs::logdebug);
}

/**
 * checks if all the links in @p arrLinkInfo are of type @p LinkType
 * @param arrLinkInfo array of links between two HSA nodes
 * @param LinkType type of HSA link (in line with hsa_amd_link_info_type_t)
 * @return 'true' if all links are of type defined by link_type key
 * @return 'false' otherwise
 */
bool rvs::hsa::check_link_type(
  const std::vector<rvs::linkinfo_t>& arrLinkInfo, int LinkType) {
  // nothing to check - just return
  if (LinkType < 0)
    return true;

  // all link segmenst should have the requested type or the test fails
  bool retval = true;
  for (auto it = arrLinkInfo.begin(); it != arrLinkInfo.end(); ++it) {
    if (it->etype != LinkType) {
      retval = false;
    }
  }
  return retval;
}


/**
 * @brief Fetch all HSA agents
 *
 * Functionality:
 *
 * Fetch all CPUs and GPUs present in the system.
 *
 * @return void
 *
 * */
void rvs::hsa::InitAgents() {
  hsa_status_t status;
  string log_msg;

  RVSHSATRACE_
  // Initialize Roc Runtime
  if (HSA_STATUS_SUCCESS != (status = hsa_init()))
    print_hsa_status(__FILE__, __LINE__, __func__, "hsa_init()", status);

  // Initialize profiling
  if (HSA_STATUS_SUCCESS !=
     (status = hsa_amd_profiling_async_copy_enable(true)))
    print_hsa_status(__FILE__, __LINE__, __func__,
                   "hsa_amd_profiling_async_copy_enable()", status);

  // Populate the lists of agents
  if (HSA_STATUS_SUCCESS !=
     (status = hsa_iterate_agents(ProcessAgent, &agent_list)))
    print_hsa_status(__FILE__, __LINE__, __func__,
                   "hsa_iterate_agents()", status);

  for (uint32_t i = 0; i < agent_list.size(); i++) {
    rvs::lp::Log("[RVSHSA] ============================", rvs::logdebug);
    log_msg = "[RVSHSA] InitAgents - agent with name = "  +
      agent_list[i].agent_name + " and device_type = " +
      agent_list[i].agent_device_type;
    rvs::lp::Log(log_msg.c_str(), rvs::logtrace);
    rvs::lp::Log("[RVSHSA] ============================", rvs::logdebug);

    // Populate the list of memory pools
    RVSHSATRACE_
    if (HSA_STATUS_SUCCESS !=
       (status = hsa_amd_agent_iterate_memory_pools(
                  agent_list[i].agent,
                  ProcessMemPool, &agent_list[i])))
      print_hsa_status(__FILE__, __LINE__, __func__,
                     "hsa_amd_agent_iterate_memory_pools()", status);

    // separate the lists
    if (agent_list[i].agent_device_type == "CPU") {
      cpu_list.push_back(agent_list[i]);
    } else if (agent_list[i].agent_device_type == "GPU") {
      gpu_list.push_back(agent_list[i]);
    } else {
      log_msg = "Unexpected agent type: " + agent_list[i].agent_device_type;
      rvs::lp::Log(log_msg, rvs::logdebug);
    }
  }

  // Initialize the list of buffer sizes to use in copy/read/write operations
  // For All Copy operations use only one buffer size
  if (size_list.size() == 0) {
    uint32_t size_len = sizeof(DEFAULT_SIZE_LIST)/sizeof(uint32_t);
    for (uint32_t idx = 0; idx < size_len; idx++) {
      size_list.push_back(DEFAULT_SIZE_LIST[idx]);
    }
  }

  std::sort(size_list.begin(), size_list.end());
}

/**
 * @brief Process individual hsa_agent
 *
 * Functionality:
 *
 * Process a single CPUs or GPUs hsa_agent_t
 *
 * @return hsa_status_t
 *
 * */
hsa_status_t rvs::hsa::ProcessAgent(hsa_agent_t agent, void* data) {
  hsa_status_t status;
  char agent_name[64];
  hsa_device_type_t device_type;
  string log_msg, log_agent_name;
  uint32_t node;
  AgentInformation agent_info;

  // get agent list
  vector<AgentInformation>* agent_l =
  reinterpret_cast<vector<AgentInformation>*>(data);

  RVSHSATRACE_;

  // Get the name of the agent
  if (HSA_STATUS_SUCCESS !=
     (status = hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, agent_name)))
    print_hsa_status(__FILE__, __LINE__, __func__,
                   "[HSA_AGENT_INFO_NAME", status);
  rvs::lp::Log(string("agent_name: ") + agent_name, rvs::logdebug);

  // Get device type
  if (HSA_STATUS_SUCCESS !=
     (status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type)))
    print_hsa_status(__FILE__, __LINE__, __func__,
                   "[RVSHSA] HSA_AGENT_INFO_DEVICE", status);

  if (HSA_STATUS_SUCCESS !=
     (status = hsa_agent_get_info(agent, HSA_AGENT_INFO_NODE, &node)))
    print_hsa_status(__FILE__, __LINE__, __func__,
                   "[RVSHSA] HSA_AGENT_INFO_NODE", status);
  agent_info.node = node;
  rvs::lp::Log("node: " + std::to_string(node), rvs::logdebug);

  log_agent_name = agent_name;
  log_msg = "[RVSHSA] Found agent with name = "  + log_agent_name +
    " and device_type = ";
  switch (device_type) {
    case HSA_DEVICE_TYPE_CPU : {
      agent_info.agent_device_type = "CPU";
      log_msg = log_msg + "CPU.";
      break;
    };
    case HSA_DEVICE_TYPE_GPU : {
      agent_info.agent_device_type = "GPU";
      log_msg = log_msg + "GPU.";
      break;
    };
    case HSA_DEVICE_TYPE_DSP : {
      agent_info.agent_device_type = "DSP";
      log_msg = log_msg + "DSP.";
      break;
    };
  }
  rvs::lp::Log(log_msg.c_str(), rvs::logdebug);

  // add agent to list
  agent_info.agent = agent;
  agent_info.agent_name = log_agent_name;
  agent_l->push_back(agent_info);

  return HSA_STATUS_SUCCESS;
}

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
hsa_status_t rvs::hsa::ProcessMemPool(hsa_amd_memory_pool_t pool, void* data) {
  hsa_status_t status;

  RVSHSATRACE_
  // get current agents memory pools
  AgentInformation* agent_info = reinterpret_cast<AgentInformation*>(data);

  // Query pools' segment, report only pools from global segment
  hsa_amd_segment_t segment;
  if (HSA_STATUS_SUCCESS != (status = hsa_amd_memory_pool_get_info(pool,
                                        HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
                                        &segment)))
    print_hsa_status(__FILE__, __LINE__, __func__,
                   "hsa_amd_memory_pool_get_info()", status);
  if (HSA_AMD_SEGMENT_GLOBAL != segment) {
    RVSHSATRACE_
    return HSA_STATUS_SUCCESS;
  }

  RVSHSATRACE_
  // Determine if allocation is allowed in this pool
  // Report only pools that allow an alloction by user
  bool alloc = false;
  if (HSA_STATUS_SUCCESS != (status = hsa_amd_memory_pool_get_info(pool,
                            HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
                            &alloc)))
    print_hsa_status(__FILE__, __LINE__, __func__,
                   "HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED", status);
  if (alloc != true) {
    RVSHSATRACE_
    return HSA_STATUS_SUCCESS;
  }

  RVSHSATRACE_
  // Query the max allocatable size
  size_t max_size = 0;
  if (HSA_STATUS_SUCCESS != (status = hsa_amd_memory_pool_get_info(pool,
                                        HSA_AMD_MEMORY_POOL_INFO_SIZE,
                                        &max_size)))
    print_hsa_status(__FILE__, __LINE__, __func__,
                   "HSA_AMD_MEMORY_POOL_INFO_SIZE", status);
  agent_info->max_size_list.push_back(max_size);

  // Determine if the pools is accessible to all agents
  bool access_to_all = false;
  if (HSA_STATUS_SUCCESS != (status = hsa_amd_memory_pool_get_info(pool,
                                HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL,
                                &access_to_all)))
    print_hsa_status(__FILE__, __LINE__, __func__,
                   "HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL", status);

  // Determine type of access to owner agent
  hsa_amd_memory_pool_access_t owner_access;
  hsa_agent_t agent = agent_info->agent;
  if (HSA_STATUS_SUCCESS !=
     (status = hsa_amd_agent_memory_pool_get_info(agent, pool,
                                      HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS,
                                      &owner_access)))
    print_hsa_status(__FILE__, __LINE__, __func__, status);

  // Determine if the pool is fine-grained or coarse-grained
  uint32_t flag = 0;
  if (HSA_STATUS_SUCCESS != (status = hsa_amd_memory_pool_get_info(pool,
                                        HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS,
                                        &flag)))
    print_hsa_status(__FILE__, __LINE__, __func__,
                   "HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS", status);
  bool is_kernarg = (HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT & flag);

  // Update the pool handle for system memory if kernarg is true
  rvs::lp::Log("[RVSHSA] ****************************************",
               rvs::logdebug);
  if (is_kernarg) {
    agent_info->sys_pool = pool;
    rvs::lp::Log("[RVSHSA] Found system memory region", rvs::logdebug);
  } else if (owner_access != HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) {
    rvs::lp::Log("[RVSHSA] Found regular memory region", rvs::logdebug);
  }
  rvs::lp::Log("[RVSHSA] ****************************************",
               rvs::logdebug);
  agent_info->mem_pool_list.push_back(pool);

  return HSA_STATUS_SUCCESS;
}


/**
 * @brief Find HSA agent index in RVS HSA wrapper
 * @param Node NUMA node
 * @return index in agent_list vector
 *
 * */
int rvs::hsa::FindAgent(const uint32_t Node) {
  for (size_t i = 0; i < agent_list.size(); i++) {
    if (agent_list[i].node == Node)
      return i;
  }
  RVSHSATRACE_
  return -1;
}

/**
 * @brief Fetch time needed to copy data between two memory pools
 *
 * Uses time obtained from corresponding hsa_signal objects
 *
 * @param bidirectional 'true' for bidirectional transfer
 * @param signal_fwd signal used for direct transfer
 * @param signal_rev signal used for reverse transfer
 * @return time in seconds
 *
 * */
double rvs::hsa::GetCopyTime(bool bidirectional,
                             hsa_signal_t signal_fwd, hsa_signal_t signal_rev) {
  hsa_status_t status;
  // Obtain time taken for forward copy
  hsa_amd_profiling_async_copy_time_t async_time_fwd {0, 0};
  if (HSA_STATUS_SUCCESS !=
     (status =
       hsa_amd_profiling_get_async_copy_time(signal_fwd, &async_time_fwd)))
    print_hsa_status(__FILE__, __LINE__, __func__,
                   "hsa_amd_profiling_get_async_copy_time(forward)",
                   status);
  if (bidirectional == false) {
    RVSHSATRACE_
    return(async_time_fwd.end - async_time_fwd.start);
  }
  RVSHSATRACE_

  hsa_amd_profiling_async_copy_time_t async_time_rev {0, 0};
  if (HSA_STATUS_SUCCESS !=
     (status =
        hsa_amd_profiling_get_async_copy_time(signal_rev, &async_time_rev)))
    print_hsa_status(__FILE__, __LINE__, __func__,
                   "hsa_amd_profiling_get_async_copy_time(backward)",
                   status);
  double start = std::min(async_time_fwd.start, async_time_rev.start);
  double end = std::max(async_time_fwd.end, async_time_rev.end);
  RVSHSATRACE_
  return(end - start);
}

/**
 * @brief Allocate buffers in source and destination memory pools
 *
 * @param SrcAgent source agent index in agent_list vector
 * @param DstAgent destination agent index in agent_list vector
 * @param Size size of data to transfer
 * @param pSrcPool [out] ptr to source memory pool
 * @param SrcBuff  [out] ptr to source buffer
 * @param pDstPool [out] ptr to destination memory pool
 * @param DstBuff  [out] ptr to destination buffer
 * @return 0 - if successfull, non-zero otherwise
 *
 * */
int rvs::hsa::Allocate(int SrcAgent, int DstAgent, size_t Size,
                     hsa_amd_memory_pool_t* pSrcPool, void** SrcBuff,
                     hsa_amd_memory_pool_t* pDstPool, void** DstBuff) {
  hsa_status_t status;
  void* srcbuff = nullptr;
  void* dstbuff = nullptr;

  // iterate over src pools
  for (size_t i = 0; i < agent_list[SrcAgent].mem_pool_list.size(); i++) {
    RVSHSATRACE_
    // size too small, continue
    if (Size > agent_list[SrcAgent].max_size_list[i]) {
      RVSHSATRACE_
      continue;
    }

    RVSHSATRACE_
    // try allocating source buffer
    if (HSA_STATUS_SUCCESS != (status = hsa_amd_memory_pool_allocate(
                agent_list[SrcAgent].mem_pool_list[i], Size, 0, &srcbuff))) {
      print_hsa_status(__FILE__, __LINE__, __func__,
                   "hsa_amd_memory_pool_allocate()",
                   status);
      continue;
    }

    RVSHSATRACE_
    // iterate over dst pools
    for (size_t j = 0; j < agent_list[DstAgent].mem_pool_list.size(); j++) {
      RVSHSATRACE_
      // size too small, continue
      if (Size > agent_list[DstAgent].max_size_list[j]) {
        RVSHSATRACE_
        continue;
      }

      RVSHSATRACE_
      // check if src agent has access to this dst agent's pool
      hsa_amd_memory_pool_access_t access =
        HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
      if (agent_list[SrcAgent].agent_device_type == "CPU") {
        RVSHSATRACE_
        if (HSA_STATUS_SUCCESS != (status = hsa_amd_agent_memory_pool_get_info(
        agent_list[DstAgent].agent,
        agent_list[SrcAgent].mem_pool_list[j],
        HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS,
        &access)))
          print_hsa_status(__FILE__, __LINE__, __func__,
                   "hsa_amd_agent_memory_pool_get_info()",
                   status);
      } else {
        RVSHSATRACE_
        if (HSA_STATUS_SUCCESS != (status = hsa_amd_agent_memory_pool_get_info(
        agent_list[SrcAgent].agent,
        agent_list[DstAgent].mem_pool_list[j],
        HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS,
        &access)))
          print_hsa_status(__FILE__, __LINE__, __func__,
                   "hsa_amd_agent_memory_pool_get_info()",
                   status);
      }

      if (access == HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) {
        RVSHSATRACE_
        continue;
      }
      RVSHSATRACE_
      // try allocating destination buffer
      if (HSA_STATUS_SUCCESS != (status = hsa_amd_memory_pool_allocate(
        agent_list[DstAgent].mem_pool_list[j], Size, 0, &dstbuff))) {
        print_hsa_status(__FILE__, __LINE__, __func__,
                   "hsa_amd_memory_pool_allocate()",
                   status);
        continue;
      }

      // destination buffer allocated,
      // give access to agents

      RVSHSATRACE_

      // determine which one is a cpu and allow access on the other agent
      if (agent_list[SrcAgent].agent_device_type == "CPU") {
        RVSHSATRACE_
        status = hsa_amd_agents_allow_access(1,
                                            &agent_list[DstAgent].agent,
                                            NULL,
                                            srcbuff);
      } else {
        RVSHSATRACE_
        status = hsa_amd_agents_allow_access(1,
                                            &agent_list[SrcAgent].agent,
                                            NULL,
                                            dstbuff);
      }
      if (status != HSA_STATUS_SUCCESS) {
        RVSHSATRACE_
        print_hsa_status(__FILE__, __LINE__, __func__,
                "hsa_amd_agents_allow_access()",
                status);
        // do cleanup
        hsa_amd_memory_pool_free(dstbuff);
        dstbuff = nullptr;
        continue;
      }

      RVSHSATRACE_
      // all OK, set output parameters:
      *pSrcPool = agent_list[SrcAgent].mem_pool_list[i];
      *pDstPool = agent_list[DstAgent].mem_pool_list[j];
      *SrcBuff = srcbuff;
      *DstBuff = dstbuff;

      return 0;
    }  // end of dst agent pool loop

    RVSHSATRACE_
    // suitable destination buffer not foud, deallocate src buff and exit
    hsa_amd_memory_pool_free(srcbuff);
  }  // end of src agent pool loop

  RVSHSATRACE_
  return -1;
}

/**
 * @brief Allocate buffers in source and destination memory pools
 *
 * @param SrcNode source NUMA node
 * @param DstNode destination NUMA node
 * @param Size size of data to transfer
 * @param bidirectional 'true' for bidirectional transfer
 * @param Duration [out] duration of transfer in seconds
 * @return 0 - if successfull, non-zero otherwise
 *
 * */
int rvs::hsa::SendTraffic(uint32_t SrcNode, uint32_t DstNode,
                              size_t Size, bool bidirectional,
                              double* Duration) {
  hsa_status_t status;
  int sts;

  int32_t src_ix_fwd;
  int32_t dst_ix_fwd;
  hsa_amd_memory_pool_t src_pool_fwd;
  hsa_amd_memory_pool_t dst_pool_fwd;
  void* src_ptr_fwd = nullptr;
  void* dst_ptr_fwd = nullptr;
  hsa_signal_t signal_fwd;

  int32_t src_ix_rev;
  int32_t dst_ix_rev;
  hsa_amd_memory_pool_t src_pool_rev;
  hsa_amd_memory_pool_t dst_pool_rev;
  void* src_ptr_rev = nullptr;
  void* dst_ptr_rev = nullptr;
  hsa_signal_t signal_rev;

  RVSHSATRACE_

  // given NUMA nodes, find agent indexes
  src_ix_fwd = FindAgent(SrcNode);
  dst_ix_fwd = FindAgent(DstNode);
  src_ix_rev = dst_ix_fwd;
  dst_ix_rev = src_ix_fwd;

    if (src_ix_fwd < 0 || dst_ix_fwd < 0) {
    RVSHSATRACE_
    return -1;
  }

  // allocate buffers and grant permissions for forward transfer
  sts = Allocate(src_ix_fwd, dst_ix_fwd, Size,
           &src_pool_fwd, &src_ptr_fwd,
           &dst_pool_fwd, &dst_ptr_fwd);
  if (sts) {
    RVSHSATRACE_
    return -1;
  }

  // Create a signal to wait on copy operation
  if (HSA_STATUS_SUCCESS !=
     (status = hsa_signal_create(1, 0, NULL, &signal_fwd))) {
    print_hsa_status(__FILE__, __LINE__, __func__,
              "hsa_signal_create()",
              status);
      hsa_amd_memory_pool_free(src_ptr_fwd);
      hsa_amd_memory_pool_free(dst_ptr_fwd);
      RVSHSATRACE_
      return -1;
  }

  if (bidirectional) {
    RVSHSATRACE_
//     src_ix_rev = dst_ix_fwd;
//     dst_ix_rev = src_ix_fwd;

    // allocate buffers and grant permissions for reverse transfer
    sts = Allocate(src_ix_rev, dst_ix_rev, Size,
            &src_pool_rev, &src_ptr_rev,
            &dst_pool_rev, &dst_ptr_rev);

    if (sts) {
      RVSHSATRACE_
      hsa_amd_memory_pool_free(src_ptr_fwd);
      hsa_amd_memory_pool_free(dst_ptr_fwd);
      return -1;
    }

    // Create a signal to wait on for reverse copy operation
    if (HSA_STATUS_SUCCESS !=
       (status = hsa_signal_create(1, 0, NULL, &signal_rev))) {
      print_hsa_status(__FILE__, __LINE__, __func__,
              "hsa_signal_create()",
              status);
      hsa_amd_memory_pool_free(src_ptr_fwd);
      hsa_amd_memory_pool_free(dst_ptr_fwd);
      hsa_amd_memory_pool_free(src_ptr_rev);
      hsa_amd_memory_pool_free(dst_ptr_rev);
      hsa_signal_destroy(signal_fwd);
      return -1;
    }
  }

  // initiate forward transfer
  hsa_signal_store_relaxed(signal_fwd, 1);
  if (HSA_STATUS_SUCCESS !=
     (status = hsa_amd_memory_async_copy(
                dst_ptr_fwd, agent_list[dst_ix_fwd].agent,
                src_ptr_fwd, agent_list[src_ix_fwd].agent,
                Size,
                0, NULL, signal_fwd)))
    print_hsa_status(__FILE__, __LINE__, __func__,
              "hsa_amd_memory_async_copy()",
              status);
  if (bidirectional) {
    RVSHSATRACE_
    // initiate reverse transfer
    hsa_signal_store_relaxed(signal_rev, 1);
    if (HSA_STATUS_SUCCESS != (status = hsa_amd_memory_async_copy(
        dst_ptr_rev, agent_list[dst_ix_rev].agent,
        src_ptr_rev, agent_list[src_ix_rev].agent, Size,
        0, NULL, signal_rev)))
      print_hsa_status(__FILE__, __LINE__, __func__,
              "hsa_amd_memory_async_copy()",
              status);
  }

  // wait for transfer to complete
  RVSHSATRACE_
  while (hsa_signal_wait_acquire(signal_fwd, HSA_SIGNAL_CONDITION_LT,
    1, uint64_t(-1), HSA_WAIT_STATE_ACTIVE)) {}

  // if bidirectional, also wait for reverse transfer to complete
  if (bidirectional == true) {
    RVSHSATRACE_
    while (hsa_signal_wait_acquire(signal_rev, HSA_SIGNAL_CONDITION_LT,
    1, uint64_t(-1), HSA_WAIT_STATE_ACTIVE)) {}
  }

  RVSHSATRACE_
  // get transfer duration
  *Duration = GetCopyTime(bidirectional, signal_fwd, signal_rev)/1000000000;

  hsa_amd_memory_pool_free(src_ptr_fwd);
  hsa_amd_memory_pool_free(dst_ptr_fwd);
  hsa_signal_destroy(signal_fwd);

  if (bidirectional) {
    RVSHSATRACE_
    hsa_amd_memory_pool_free(src_ptr_rev);
    hsa_amd_memory_pool_free(dst_ptr_rev);
    hsa_signal_destroy(signal_rev);
  }
  RVSHSATRACE_

  return 0;
}


/**
 * @brief Get peer status between Src and Dst nodes
 *
 * @param SrcNode source NUMA node
 * @param DstNode destination NUMA node
 * @return 0 - no access, 1 - Src can acces Dst, 2 - both have access
 *
 * */
int rvs::hsa::GetPeerStatus(uint32_t SrcNode, uint32_t DstNode) {
  int32_t srcix;
  int32_t dstix;
  std::string msg;

  RVSHSATRACE_
  // given NUMA nodes, find agent indexes
  srcix = FindAgent(SrcNode);
  dstix = FindAgent(DstNode);
  if (srcix < 0 || dstix < 0) {
    RVSHSATRACE_
    return 0;
  }

  int peer_status = GetPeerStatusAgent(agent_list[srcix], agent_list[dstix]);

  msg = "Src: " + std::to_string(SrcNode) + "  Dst: " + std::to_string(DstNode)
      + "  access: " + std::to_string(peer_status);
  rvs::lp::Log(msg, rvs::logdebug);

  return peer_status;
}

/**
 * @brief Get peer status between Src and Dst agents
 *
 * @param SrcAgent source agent
 * @param DstAgent destination agent
 * @return 0 - no access, 1 - Src can acces Dst, 2 - both have access
 *
 * */
int rvs::hsa::GetPeerStatusAgent(const AgentInformation&  SrcAgent,
                                 const AgentInformation&  DstAgent) {
  hsa_amd_memory_pool_access_t access_fwd;
  hsa_amd_memory_pool_access_t access_bck;
  hsa_status_t status;
  int cur_access_rights;
  int res_access_rights;
  std::string msg;

  res_access_rights = 0;
  for (size_t i = 0; i < SrcAgent.mem_pool_list.size(); i++) {
    RVSHSATRACE_
    for (size_t j = 0; j < DstAgent.mem_pool_list.size(); j++) {
      RVSHSATRACE_
      // check if Src can access Dst
      if (HSA_STATUS_SUCCESS != (status = hsa_amd_agent_memory_pool_get_info(
        SrcAgent.agent,
        DstAgent.mem_pool_list[j],
        HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &access_fwd))) {
        print_hsa_status(__FILE__, __LINE__, __func__,
              "GetPeerStatus(SRC->DST)",
              status);
        return 0;
      }
      RVSHSATRACE_
      // also check if Dst can access Src
      if (HSA_STATUS_SUCCESS != (status = hsa_amd_agent_memory_pool_get_info(
        DstAgent.agent, SrcAgent.mem_pool_list[i],
        HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &access_bck))) {
        print_hsa_status(__FILE__, __LINE__, __func__,
              "GetPeerStatus(DST->SRC)",
              status);
        return 0;
      }

      RVSHSATRACE_

      if (access_fwd == HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED &&
          access_bck == HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) {
        RVSHSATRACE_
        cur_access_rights = 0;
      }

      // Access between the two agents is Unidirectional
      if ((access_fwd == HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) ||
          (access_bck == HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED)) {
        if ((SrcAgent.agent_device_type == "GPU") &&
            (DstAgent.agent_device_type == "GPU")) {
          RVSHSATRACE_
          cur_access_rights = 0;
        } else {
          RVSHSATRACE_
          cur_access_rights = 1;
        }
      }

      if (access_fwd != HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED &&
          access_bck != HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) {
        RVSHSATRACE_
        cur_access_rights = 2;
      }

      // in case of multiple memory pools return the best access rights
      if (cur_access_rights > res_access_rights) {
        RVSHSATRACE_
        res_access_rights = cur_access_rights;
      }
      RVSHSATRACE_
    }
    RVSHSATRACE_
  }

  RVSHSATRACE_
  return res_access_rights;
}

/**
 * @brief Get link information between Src and Dst nodes
 *
 * @param SrcNode source node
 * @param DstNode destination node
 * @param pDistance ptr to NUMA distance
 * @param pInfoarr ptr to list of hop infos
 * @return 0 - OK, non-zero otherwise
 *
 * */
int rvs::hsa::GetLinkInfo(uint32_t SrcNode, uint32_t DstNode,
                  uint32_t* pDistance, std::vector<linkinfo_t>* pInfoarr) {
  int32_t srcix;
  int32_t dstix;
  hsa_status_t sts;

  RVSHSATRACE_
  // given NUMA nodes, find agent indexes
  srcix = FindAgent(SrcNode);
  dstix = FindAgent(DstNode);
  if (srcix < 0 || dstix < 0) {
    RVSHSATRACE_
    return -1;
  }

  RVSHSATRACE_

  *pDistance = NO_CONN;
  pInfoarr->clear();
  hsa_agent_t& srcagent = agent_list[srcix].agent;

  // Agent has no pools so no need to look for numa distance
  if (agent_list[dstix].mem_pool_list.size() == 0) {
    RVSHSATRACE_
    return 0;
  }

  uint32_t hops = 0;
  hsa_amd_memory_pool_t& dstpool = agent_list[dstix].mem_pool_list[0];
  sts = hsa_amd_agent_memory_pool_get_info(srcagent, dstpool,
                   HSA_AMD_AGENT_MEMORY_POOL_INFO_NUM_LINK_HOPS, &hops);
  print_hsa_status(__FILE__, __LINE__, __func__,
                  "[RVSHSA] HSA_AMD_AGENT_MEMORY_POOL_INFO_NUM_LINK_HOPS", sts);
  if (hops < 1) {
    RVSHSATRACE_
    return 0;
  }

  RVSHSATRACE_
  hsa_amd_memory_pool_link_info_t *link_info;
  uint32_t link_info_sz = hops * sizeof(hsa_amd_memory_pool_link_info_t);
  link_info =
    static_cast<hsa_amd_memory_pool_link_info_t*>(malloc(link_info_sz));
  memset(link_info, 0, (hops * sizeof(hsa_amd_memory_pool_link_info_t)));

  sts = hsa_amd_agent_memory_pool_get_info(srcagent, dstpool,
                 HSA_AMD_AGENT_MEMORY_POOL_INFO_LINK_INFO, link_info);
  print_hsa_status(__FILE__, __LINE__, __func__,
                   "[RVSHSA] HSA_AMD_AGENT_MEMORY_POOL_INFO_LINK_INFO", sts);
  *pDistance = 0;
  pInfoarr->clear();
  for (uint32_t hopIdx = 0; hopIdx < hops; hopIdx++) {
    RVSHSATRACE_
    linkinfo_t rvslinkinfo;
    *pDistance += (link_info[hopIdx]).numa_distance;
    rvslinkinfo.distance = (link_info[hopIdx]).numa_distance;
    rvslinkinfo.etype = (link_info[hopIdx]).link_type;
    switch (rvslinkinfo.etype) {
      case HSA_AMD_LINK_INFO_TYPE_HYPERTRANSPORT:
        rvslinkinfo.strtype = "HyperTransport";
        break;
      case HSA_AMD_LINK_INFO_TYPE_QPI:
        rvslinkinfo.strtype = "QPI";
        break;
      case HSA_AMD_LINK_INFO_TYPE_PCIE:
        rvslinkinfo.strtype = "PCIe";
        break;
      case HSA_AMD_LINK_INFO_TYPE_INFINBAND:
        rvslinkinfo.strtype = "InfiniBand";
        break;
      case HSA_AMD_LINK_INFO_TYPE_XGMI:
        rvslinkinfo.strtype = "xGMI";
        break;
      default:
        RVSHSATRACE_
        rvslinkinfo.strtype = "unknown-"
                            + std::to_string(rvslinkinfo.etype);
    }
    pInfoarr->push_back(rvslinkinfo);
  }
  free(link_info);

  RVSHSATRACE_
  return 0;
}
