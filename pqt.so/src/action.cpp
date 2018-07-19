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

#include "rvs_module.h"
#include "worker.h"

extern "C"
{
#include <pci/pci.h>
#include <linux/pci.h>
}
#include <iostream>
#include <algorithm>

#include "pci_caps.h"
#include "gpu_util.h"
#include "rvs_module.h"
#include "rvsloglp.h"


using namespace std;

static Worker* pworker;

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
  log("[PQT] in run()", rvs::logdebug);
  
  // get all the agents
  GetAgents();
  
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
  switch (st) {
    case HSA_STATUS_SUCCESS : {
      log_msg = message + "[PQT] HSA_STATUS: The function has been executed successfully.";
      break;
    };
    case HSA_STATUS_INFO_BREAK : {
      log_msg = message + "[PQT] HSA_STATUS: A traversal over a list of elements has been interrupted by the application before completing.";
      break;
    };
    case HSA_STATUS_ERROR : {
      log_msg = message + "[PQT] HSA_STATUS: A generic error has occurred.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_ARGUMENT : {
      log_msg = message + "[PQT] HSA_STATUS: One of the actual arguments does not meet a precondition stated in the documentation of the corresponding formal argument.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_QUEUE_CREATION : {
      log_msg = message + "[PQT] HSA_STATUS: The requested queue creation is not valid.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_ALLOCATION : {
      log_msg = message + "[PQT] HSA_STATUS: The requested allocation is not valid.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_AGENT : {
      log_msg = message + "[PQT] HSA_STATUS: The agent is invalid.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_REGION : {
      log_msg = message + "[PQT] HSA_STATUS: The memory region is invalid.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_SIGNAL : {
      log_msg = message + "[PQT] HSA_STATUS: The signal is invalid.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_QUEUE : {
      log_msg = message + "[PQT] HSA_STATUS: The queue is invalid.";
      break;
    };
    case HSA_STATUS_ERROR_OUT_OF_RESOURCES : {
      log_msg = message + "[PQT] HSA_STATUS: The HSA runtime failed to allocate the necessary resources. This error may also occur when the HSA runtime needs to spawn threads or create internal OS-specific events.";
      break;
    };    
    case HSA_STATUS_ERROR_INVALID_PACKET_FORMAT : {
      log_msg = message + "[PQT] HSA_STATUS: The AQL packet is malformed.";
      break;
    };
    case HSA_STATUS_ERROR_RESOURCE_FREE : {
      log_msg = message + "[PQT] HSA_STATUS: An error has been detected while releasing a resource.";
      break;
    };
    case HSA_STATUS_ERROR_NOT_INITIALIZED : {
      log_msg = message + "[PQT] HSA_STATUS: An API other than ::hsa_init has been invoked while the reference count of the HSA runtime is 0.";
      break;
    };
    case HSA_STATUS_ERROR_REFCOUNT_OVERFLOW : {
      log_msg = message + "[PQT] HSA_STATUS: The maximum reference count for the object has been reached.";
      break;
    };
    case HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS : {
      log_msg = message + "[PQT] HSA_STATUS: The arguments passed to a functions are not compatible.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_INDEX : {
      log_msg = message + "[PQT] HSA_STATUS: The index is invalid.";
      break;
    };    
    case HSA_STATUS_ERROR_INVALID_ISA : {
      log_msg = message + "[PQT] HSA_STATUS: The instruction set architecture is invalid.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_ISA_NAME : {
      log_msg = message + "[PQT] HSA_STATUS: The instruction set architecture name is invalid.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_CODE_OBJECT : {
      log_msg = message + "[PQT] HSA_STATUS: The code object is invalid.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_EXECUTABLE : {
      log_msg = message + "[PQT] HSA_STATUS: The executable is invalid.";
      break;
    };
    case HSA_STATUS_ERROR_FROZEN_EXECUTABLE : {
      log_msg = message + "[PQT] HSA_STATUS: The executable is frozen.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_SYMBOL_NAME : {
      log_msg = message + "[PQT] HSA_STATUS: There is no symbol with the given name.";
      break;
    };    
    case HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED : {
      log_msg = message + "[PQT] HSA_STATUS: The variable is already defined.";
      break;
    };
    case HSA_STATUS_ERROR_VARIABLE_UNDEFINED : {
      log_msg = message + "[PQT] HSA_STATUS: The variable is undefined.";
      break;
    };
    case HSA_STATUS_ERROR_EXCEPTION : {
      log_msg = message + "[PQT] HSA_STATUS: An HSAIL operation resulted on a hardware exception.";
      break;
    };
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
  print_hsa_status("[PQT] GetAgents - hsa_init()\n", status);
  
  // Populate the lists of agents
  log("[PQT] Before hsa_iterate_agents ...", rvs::logdebug);
  status = hsa_iterate_agents(ProcessAgent, &agent_list);
  log("[PQT] After hsa_iterate_agents ...", rvs::logdebug);
  print_hsa_status("[PQT] GetAgents - hsa_iterate_agents()\n", status);
    
  for (int i = 0; i < agent_list.size(); i++) {
    log_msg = "[PQT] GetAgents - agent with name = "  + agent_list[i].agent_name + " and device_type = " + agent_list[i].agent_device_type;
    log(log_msg.c_str(), rvs::logdebug);
    
//     // Populate the list of memory pools
//     log("[PQT] Before hsa_amd_agent_iterate_memory_pools ...", rvs::logdebug);
//     status = hsa_amd_agent_iterate_memory_pools(agent_list[i].agent, ProcessMemPool, &agent_list);
//     log("[PQT] After hsa_amd_agent_iterate_memory_pools ...", rvs::logdebug);
//     print_hsa_status("[PQT] GetAgents - hsa_amd_agent_iterate_memory_pools()\n", status);
    
    
    // separate the lists
    if (agent_list[i].agent_device_type == "CPU") {
      cpu_list.push_back(agent_list[i]);
    } else if (agent_list[i].agent_device_type == "GPU") {
      gpu_list.push_back(agent_list[i]);
    }
  }
 
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
  print_hsa_status("[PQT] ProcessAgent - hsa_agent_get_info()\n", status);

  // Get device type
  status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
  print_hsa_status("[PQT] ProcessAgent - hsa_agent_get_info()\n", status);
  
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
