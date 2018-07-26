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
#include <cstring>

#include "rvs_module.h"
#include "worker.h"
#include "pci_caps.h"
#include "gpu_util.h"
#include "rvsloglp.h"
#include "rvstimer.h"

//  static Worker* pworker;  //FIXME

//! Default constructor
action::action() {
}

//! Default destructor
action::~action() {
  property.clear();
}

/**
 * @brief Implements action functionality
 *
 * Functionality:
 *
 * - If "do_gpu_list" property is set, it lists all AMD GPUs present in the system and exits
 * - If "monitor" property is set to "true", it creates Worker thread and initiates monitoring and exits
 * - If "monitor" property is not set or is not set to "true", it stops the Worker thread and exits
 *
 * @return 0 - success. non-zero otherwise
 *
 * */
int action::run(void) {
  hsa_agent_t src_agent, dst_agent;
  hsa_amd_memory_pool_t src_buff, dst_buff;
  size_t src_max_size, dst_max_size;
//   bool bidirectional;
  string log_msg;
  
  log("[PQT] in run()", rvs::logdebug);
  
  // get all the agents
  GetAgents();
  
  rvs::lp::Log("[PQT] gpu_list.size() = " + std::to_string(gpu_list.size()), rvs::logdebug);
  for (uint32_t i = 0; i < gpu_list.size(); i++) {
    for (uint32_t j = 0; j < gpu_list.size(); j++) {
      if (i == j) { continue; };
      for (uint32_t n = 0; n < gpu_list[i].mem_pool_list.size(); n++) {      
        for (uint32_t m = 0; m < gpu_list[j].mem_pool_list.size(); m++) {
          src_agent    = gpu_list[i].agent;
          dst_agent    = gpu_list[j].agent;
          src_buff     = gpu_list[i].mem_pool_list[n];
          dst_buff     = gpu_list[j].mem_pool_list[m];
          src_max_size = gpu_list[i].max_size_list[n];
          dst_max_size = gpu_list[j].max_size_list[m];
          // print
          log_msg = "[PQT] src = " + gpu_list[i].agent_name + " / " + gpu_list[i].agent_device_type + " and dst = " + gpu_list[j].agent_name + " / " + gpu_list[j].agent_device_type;
          log(log_msg.c_str(), rvs::logdebug);
          // send p2p traffic (unidirectional)
          send_p2p_traffic(src_agent, dst_agent, src_buff, dst_buff, false, src_max_size, dst_max_size, false);
          // send p2p traffic (bidirectional)
//           send_p2p_traffic(src_agent, dst_agent, src_buff, dst_buff, true, src_max_size, dst_max_size, false);          
        }
      }
    }
  } 
  
  rvs::lp::Log("[PQT] Test DONE", rvs::logdebug);
  
  return 0;
}

/**
 * @brief Lists AMD GPUs
 *
 * Functionality:
 *
 * Lists all AMD GPUs present in the system.
 *
 * @return 0 - success. non-zero otherwise
 *
 * */
int action::do_gpu_list() {
  log("[PQT] in do_gpu_list()", rvs::logdebug);

  return 0;
}

// TODO add info
void action::print_hsa_status(string message, hsa_status_t st) {
  string log_msg = message;
//   // skip successfull messages
//   if (st == HSA_STATUS_SUCCESS) {
//     return;
//   }
  switch (st) {
    case HSA_STATUS_SUCCESS : {
      log_msg = message + " The function has been executed successfully.";
      break;
    };
    case HSA_STATUS_INFO_BREAK : {
      log_msg = message + " A traversal over a list of elements has been interrupted by the application before completing.";
      break;
    };
    case HSA_STATUS_ERROR : {
      log_msg = message + " A generic error has occurred.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_ARGUMENT : {
      log_msg = message + " One of the actual arguments does not meet a precondition stated in the documentation of the corresponding formal argument.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_QUEUE_CREATION : {
      log_msg = message + " The requested queue creation is not valid.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_ALLOCATION : {
      log_msg = message + " The requested allocation is not valid.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_AGENT : {
      log_msg = message + " The agent is invalid.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_REGION : {
      log_msg = message + " The memory region is invalid.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_SIGNAL : {
      log_msg = message + " The signal is invalid.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_QUEUE : {
      log_msg = message + " The queue is invalid.";
      break;
    };
    case HSA_STATUS_ERROR_OUT_OF_RESOURCES : {
      log_msg = message + " The HSA runtime failed to allocate the necessary resources. This error may also occur when the HSA runtime needs to spawn threads or create internal OS-specific events.";
      break;
    };    
    case HSA_STATUS_ERROR_INVALID_PACKET_FORMAT : {
      log_msg = message + " The AQL packet is malformed.";
      break;
    };
    case HSA_STATUS_ERROR_RESOURCE_FREE : {
      log_msg = message + " An error has been detected while releasing a resource.";
      break;
    };
    case HSA_STATUS_ERROR_NOT_INITIALIZED : {
      log_msg = message + " An API other than ::hsa_init has been invoked while the reference count of the HSA runtime is 0.";
      break;
    };
    case HSA_STATUS_ERROR_REFCOUNT_OVERFLOW : {
      log_msg = message + " The maximum reference count for the object has been reached.";
      break;
    };
    case HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS : {
      log_msg = message + " The arguments passed to a functions are not compatible.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_INDEX : {
      log_msg = message + " The index is invalid.";
      break;
    };    
    case HSA_STATUS_ERROR_INVALID_ISA : {
      log_msg = message + " The instruction set architecture is invalid.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_ISA_NAME : {
      log_msg = message + " The instruction set architecture name is invalid.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_CODE_OBJECT : {
      log_msg = message + " The code object is invalid.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_EXECUTABLE : {
      log_msg = message + " The executable is invalid.";
      break;
    };
    case HSA_STATUS_ERROR_FROZEN_EXECUTABLE : {
      log_msg = message + " The executable is frozen.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_SYMBOL_NAME : {
      log_msg = message + " There is no symbol with the given name.";
      break;
    };    
    case HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED : {
      log_msg = message + " The variable is already defined.";
      break;
    };
    case HSA_STATUS_ERROR_VARIABLE_UNDEFINED : {
      log_msg = message + " The variable is undefined.";
      break;
    };
    case HSA_STATUS_ERROR_EXCEPTION : {
      log_msg = message + " An HSAIL operation resulted on a hardware exception.";
      break;
    };
    default : {
      log_msg = message + " Unknown error.";
      break;      
    }
  };
  log(log_msg.c_str(), rvs::logdebug);
}


/**
 * @brief Fetch all hsa_agents
 *
 * Functionality:
 *
 * Fetch all CPUs and GPUs present in the system.
 *
 * @return void
 *
 * */
void action::GetAgents() {
  hsa_status_t status;
  string log_msg;
  
  // Initialize Roc Runtime
  log("[PQT] Before hsa_init ...", rvs::logdebug);
  status = hsa_init();
  log("[PQT] After hsa_init ...", rvs::logdebug);
  print_hsa_status("[PQT] GetAgents - hsa_init()", status);
  
  // Initialize profiling
  log("[PQT] Before hsa_amd_profiling_async_copy_enable ...", rvs::logdebug);
  status = hsa_amd_profiling_async_copy_enable(true);
  log("[PQT] Before hsa_amd_profiling_async_copy_enable ...", rvs::logdebug);    
  print_hsa_status("[PQT] GetAgents - hsa_amd_profiling_async_copy_enable()", status);  
  
  // Populate the lists of agents
  log("[PQT] Before hsa_iterate_agents ...", rvs::logdebug);
  status = hsa_iterate_agents(ProcessAgent, &agent_list);
  log("[PQT] After hsa_iterate_agents ...", rvs::logdebug);
  print_hsa_status("[PQT] GetAgents - hsa_iterate_agents()", status);
    
  for (uint32_t i = 0; i < agent_list.size(); i++) {
    log("[PQT] ===================================================================================================================", rvs::logdebug);
    log_msg = "[PQT] GetAgents - agent with name = "  + agent_list[i].agent_name + " and device_type = " + agent_list[i].agent_device_type;
    log(log_msg.c_str(), rvs::logdebug);
    log("[PQT] ===================================================================================================================", rvs::logdebug);
    
    // Populate the list of memory pools
    log("[PQT] Before hsa_amd_agent_iterate_memory_pools ...", rvs::logdebug);
    status = hsa_amd_agent_iterate_memory_pools(agent_list[i].agent, ProcessMemPool, &agent_list[i]);
    log("[PQT] After hsa_amd_agent_iterate_memory_pools ...", rvs::logdebug);
    print_hsa_status("[PQT] GetAgents - hsa_amd_agent_iterate_memory_pools()", status);
   
    // separate the lists
    if (agent_list[i].agent_device_type == "CPU") {
      cpu_list.push_back(agent_list[i]);
    } else if (agent_list[i].agent_device_type == "GPU") {
      gpu_list.push_back(agent_list[i]);
    }
  }

  // Initialize the list of buffer sizes to use in copy/read/write operations
  // For All Copy operations use only one buffer size
  if (size_list.size() == 0) {
    uint32_t size_len = sizeof(DEFAULT_SIZE_LIST)/sizeof(uint32_t);
    for (uint32_t idx = 0; idx < size_len; idx++) {
      size_list.push_back(DEFAULT_SIZE_LIST[idx]);
    }
  } else {
    uint32_t size_len = size_list.size();
    for (uint32_t idx = 0; idx < size_len; idx++) {
      size_list[idx] = size_list[idx] * 1024 * 1024;
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
hsa_status_t action::ProcessAgent(hsa_agent_t agent, void* data) {
  hsa_status_t status;
  char agent_name[64];
  hsa_device_type_t device_type;
  string log_msg, log_agent_name;
  AgentInformation agent_info;
  
  // get agent list
  vector<AgentInformation>* agent_l = reinterpret_cast<vector<AgentInformation>*>(data);
  
  log("[PQT] Called ProcessAgent() ...", rvs::logdebug);
  
  // Get the name of the agent
  status = hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, agent_name);
  print_hsa_status("[PQT] ProcessAgent - hsa_agent_get_info()", status);

  // Get device type
  status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
  print_hsa_status("[PQT] ProcessAgent - hsa_agent_get_info()", status);
  
  log_agent_name = agent_name;
  log_msg = "[PQT] Found agent with name = "  + log_agent_name + " and device_type = ";
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
  };
  log(log_msg.c_str(), rvs::logdebug);
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
hsa_status_t action::ProcessMemPool(hsa_amd_memory_pool_t pool, void* data) {
  hsa_status_t status;

  // get current agents memory pools
  AgentInformation* agent_info = reinterpret_cast<AgentInformation*>(data);

  // Query pools' segment, report only pools from global segment
  hsa_amd_segment_t segment;
  status = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
  print_hsa_status("[PQT] ProcessMemPool - hsa_amd_memory_pool_get_info(HSA_AMD_MEMORY_POOL_INFO_SEGMENT)", status);
  if (HSA_AMD_SEGMENT_GLOBAL != segment) {
    return HSA_STATUS_SUCCESS;
  }

  // Determine if allocation is allowed in this pool
  // Report only pools that allow an alloction by user
  bool alloc = false;
  status = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &alloc);
  print_hsa_status("[PQT] ProcessMemPool - hsa_amd_memory_pool_get_info(HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED)", status);
  if (alloc != true) {
    return HSA_STATUS_SUCCESS;
  }

  // Query the max allocatable size
  size_t max_size = 0;
  status = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SIZE, &max_size);
  print_hsa_status("[PQT] ProcessMemPool - hsa_amd_memory_pool_get_info(HSA_AMD_MEMORY_POOL_INFO_SIZE)", status);
  agent_info->max_size_list.push_back(max_size);

  // Determine if the pools is accessible to all agents
  bool access_to_all = false;
  status = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL, &access_to_all);
  print_hsa_status("[PQT] ProcessMemPool - hsa_amd_memory_pool_get_info(HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL)", status);

  // Determine type of access to owner agent
  hsa_amd_memory_pool_access_t owner_access;
  hsa_agent_t agent = agent_info->agent;
  status = hsa_amd_agent_memory_pool_get_info(agent, pool, HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &owner_access);
  print_hsa_status("[PQT] ProcessMemPool - hsa_amd_memory_pool_get_info(HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS)", status);

  // Determine if the pool is fine-grained or coarse-grained
  uint32_t flag = 0;
  status = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flag);
  print_hsa_status("[PQT] ProcessMemPool - hsa_amd_memory_pool_get_info(HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS)", status);
  bool is_kernarg = (HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT & flag);
//   bool is_fine_grained = (HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED & flag);

  // Update the pool handle for system memory if kernarg is true
  log("[PQT] *******************************************************************************************************************", rvs::logdebug);  
  if (is_kernarg) {
    agent_info->sys_pool = pool;
    log("[PQT] Found system memory region", rvs::logdebug);
  } else if (owner_access != HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) {
    agent_info->mem_pool_list.push_back(pool);
    log("[PQT] Found regular memory region", rvs::logdebug);
  }
  log("[PQT] *******************************************************************************************************************", rvs::logdebug);

  return HSA_STATUS_SUCCESS;
}


// TODO info
double action::GetCopyTime(bool bidirectional, hsa_signal_t signal_fwd, hsa_signal_t signal_rev) {
  hsa_status_t status;
  // Obtain time taken for forward copy
  hsa_amd_profiling_async_copy_time_t async_time_fwd = {0};
  status = hsa_amd_profiling_get_async_copy_time(signal_fwd, &async_time_fwd);
  print_hsa_status("[PQT] GetCopyTime - hsa_amd_profiling_get_async_copy_time(forward)", status);
  if (bidirectional == false) {
    return(async_time_fwd.end - async_time_fwd.start);
  }

  hsa_amd_profiling_async_copy_time_t async_time_rev = {0};
  status = hsa_amd_profiling_get_async_copy_time(signal_rev, &async_time_rev);
  print_hsa_status("[PQT] GetCopyTime - hsa_amd_profiling_get_async_copy_time(backward)", status);
  double start = min(async_time_fwd.start, async_time_rev.start);
  double end = max(async_time_fwd.end, async_time_rev.end);
  return(end - start);
}


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
void action::send_p2p_traffic(hsa_agent_t src_agent, hsa_agent_t dst_agent, hsa_amd_memory_pool_t src_buff, hsa_amd_memory_pool_t dst_buff, bool bidirectional, size_t src_max_size, size_t dst_max_size, bool validate) {
  hsa_status_t status;
  void* src_pool_pointer;
  void* dst_pool_pointer;
  string log_msg;
  hsa_signal_t signal_fwd, signal_rev;;
  char s_buff[256];
  uint64_t total_size = 0;
  double total_time = 0;  
  double curr_time;
  double bandwidth;
  
  // Create a timer object and reset signals
  PerfTimer timer;
  uint32_t index = timer.CreateTimer();

  log("[PQT] +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++", rvs::logdebug);
  log("[PQT] send_p2p_traffic called ... ", rvs::logdebug);
  log("[PQT] +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++", rvs::logdebug);

  snprintf(s_buff, sizeof(s_buff), "%lX", src_agent.handle);
  rvs::lp::Log(std::string("src_agent = ") + s_buff, rvs::logdebug);

  snprintf(s_buff, sizeof(s_buff), "%lX", dst_agent.handle);
  rvs::lp::Log(std::string("dst_agent = ") + s_buff, rvs::logdebug);

  // Initialize size of buffer to equal the largest element of allocation
  uint32_t size_len = size_list.size();
//   uint32_t max_size = size_list.back();

  // Iterate through the differnt buffer sizes to
  // compute the bandwidth as determined by copy
  for (uint32_t idx = 0; idx < size_len; idx++) {
    
    // This should not be happening
    uint32_t curr_size = size_list[idx];
    if (curr_size > src_max_size || curr_size > dst_max_size) {
      break;
    }  

    // print current size
    log("[PQT] -----------------------------------------------------------------------------------", rvs::logdebug);
    log_msg = "[PQT] send_p2p_traffic - curr_size = " + std::to_string(curr_size) + " Bytes";
    log(log_msg.c_str(), rvs::logdebug);
    log("[PQT] -----------------------------------------------------------------------------------", rvs::logdebug);  

    // Allocate buffers in src and dst pools
    status = hsa_amd_memory_pool_allocate(src_buff, curr_size, 0, (void**)&src_pool_pointer);
    print_hsa_status("[PQT] send_p2p_traffic - hsa_amd_memory_pool_allocate(SRC)", status);
    snprintf(s_buff, sizeof(s_buff), "%p", src_pool_pointer);
    rvs::lp::Log(std::string("src_pool_pointer = ") + s_buff, rvs::logdebug);

    status = hsa_amd_memory_pool_allocate(dst_buff, curr_size, 0, (void**)&dst_pool_pointer);
    print_hsa_status("[PQT] send_p2p_traffic - hsa_amd_memory_pool_allocate(DST)", status);
    snprintf(s_buff, sizeof(s_buff), "%p", dst_pool_pointer);
    rvs::lp::Log(std::string("dst_pool_pointer = ") + s_buff, rvs::logdebug);

    // Initialize buffers if validate is used
    if (validate == true) {
      std::memset(src_pool_pointer, 0xA5, curr_size);
      std::memset(dst_pool_pointer, 0x5A, curr_size);
    }

    // Create a signal to wait on copy operation
    // hsa_signal_create(hsa_signal_value_t initial_value, uint32_t num_consumers, const hsa_agent_t *consumers, hsa_signal_t *signal)
    status = hsa_signal_create(1, 0, NULL, &signal_fwd);
    print_hsa_status("[PQT] send_p2p_traffic - hsa_signal_create()", status);
    
    // get agent access
    status = hsa_amd_agents_allow_access(1, &src_agent, NULL, dst_pool_pointer);
    print_hsa_status("[PQT] send_p2p_traffic - hsa_amd_agents_allow_access(SRC)", status);

    status = hsa_amd_agents_allow_access(1, &dst_agent, NULL, src_pool_pointer);
    print_hsa_status("[PQT] send_p2p_traffic - hsa_amd_agents_allow_access(DST)", status);
    // store signal
    hsa_signal_store_relaxed(signal_fwd, 1);

    // TODO need to check if the combination is unidirectonial or bidirectional
    
    // Determine if accessibility to dst pool for src agent is not denied
    hsa_amd_memory_pool_access_t access;
    status = hsa_amd_agent_memory_pool_get_info(src_agent, dst_buff, HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &access);
    print_hsa_status("[PQT] send_p2p_traffic - hsa_amd_agent_memory_pool_get_info(SRC->DST)", status);

    if (access == HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) {
      log_msg = "[PQT] send_p2p_traffic - HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED for SRC -> DST), so skip it ...";
      log(log_msg.c_str(), rvs::logdebug);
      return;
    }
    
    // Start the timer and launch forward copy operation
    total_size += curr_size;
    timer.StartTimer(index); 
    
    // Copy from src into dst buffer
    // hsa_amd_memory_async_copy(void* dst, hsa_agent_t dst_agent, const void* src, hsa_agent_t src_agent, size_t size, uint32_t num_dep_signals, const hsa_signal_t* dep_signals, hsa_signal_t completion_signal)
    status = hsa_amd_memory_async_copy(dst_pool_pointer, dst_agent, src_pool_pointer, src_agent, curr_size, 0, NULL, signal_fwd);
    print_hsa_status("[PQT] send_p2p_traffic - hsa_amd_memory_async_copy(SRC -> DST)", status);

    // Wait for the forward copy operation to complete
    log_msg = "[PQT] send_p2p_traffic - hsa_signal_wait_acquire(SRC -> DST) before ...";
    log(log_msg.c_str(), rvs::logdebug);
    while (hsa_signal_wait_acquire(signal_fwd, HSA_SIGNAL_CONDITION_LT, 1, uint64_t(-1), HSA_WAIT_STATE_ACTIVE));
    log_msg = "[PQT] send_p2p_traffic - hsa_signal_wait_acquire(SRC -> DST) after ...";
    log(log_msg.c_str(), rvs::logdebug);
    
    // Stop the timer object
    timer.StopTimer(index);

    // Push the time taken for copy into a vector of copy times
//     curr_time = timer.ReadTimer(index);
    log_msg = "[PQT] send_p2p_traffic - timer = " + std::to_string(timer.ReadTimer(index));
    log(log_msg.c_str(), rvs::logdebug);    
    
    curr_time = GetCopyTime(bidirectional, signal_fwd, signal_rev)/1000000000;
    total_time += curr_time;
    log_msg = "[PQT] send_p2p_traffic - total_time = " + std::to_string(total_time);
    log(log_msg.c_str(), rvs::logdebug);
    
    // convert to GB/s
    bandwidth = (curr_size / curr_time);
    bandwidth /= (1024*1024*1024);
    log_msg = "[PQT] send_p2p_traffic - PARTIAL curr_size = " + std::to_string(curr_size) + " Bytes and curr_time = " + std::to_string(curr_time) + " bandwidth = " + std::to_string(bandwidth) + " GBytes/s";
    log(log_msg.c_str(), rvs::logdebug);    
    
    // Compare output equals input
    if (validate == true && memcmp(src_pool_pointer, dst_pool_pointer, curr_size) != 0) {
      log("After copy is finished memory content is not the same", rvs::logerror);
    }
    
  }
  
  // convert to GB/s
  bandwidth = (total_size / total_time);
  bandwidth /= (1024*1024*1024);
  log_msg = "[PQT] send_p2p_traffic - END total_size = " + std::to_string(total_size) + " Bytes and total_time = " + std::to_string(total_time) + " bandwidth = " + std::to_string(bandwidth) + " GBytes/s";
  log(log_msg.c_str(), rvs::logdebug);
  
}



