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
#ifndef INCLUDE_RVSHSA_H_
#define INCLUDE_RVSHSA_H_

#include <stdlib.h>
#include <assert.h>
#include <algorithm>
#include <unistd.h>
#include <cctype>
#include <sstream>
#include <limits>

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

using std::string;
using std::vector;


// TODO add info
class rvshsa {
 public:
  rvshsa();
  virtual ~rvshsa();

  // agent structure
  struct AgentInformation {
    // global agent information
    hsa_agent_t                   agent;
    string                        agent_name;
    string                        agent_device_type;
    // System region
    hsa_amd_memory_pool_t         sys_pool;
    // Memory region
    vector<hsa_amd_memory_pool_t> mem_pool_list;
    vector<size_t>                max_size_list;
  };


  // The values are in megabytes at allocation time
  const uint32_t DEFAULT_SIZE_LIST[20] = {  1 * 1024,
                                            2 * 1024,
                                            4 * 1024,
                                            8 * 1024,
                                            16 * 1024,
                                            32 * 1024,
                                            64 * 1024,
                                            128 * 1024,
                                            256 * 1024,
                                            512 * 1024,
                                            1 * 1024 * 1024,
                                            2 * 1024 * 1024,
                                            4 * 1024 * 1024,
                                            8 * 1024 * 1024,
                                            16 * 1024 * 1024,
                                            32 * 1024 * 1024,
                                            64 * 1024 * 1024,
                                            128 * 1024 * 1024,
                                            256 * 1024 * 1024,
                                            512 * 1024 * 1024 };  
  
  // List of sizes to use in copy and read/write transactions
  // Size is specified in terms of Megabytes
  vector<uint32_t> size_list;
                                   
  // TODO add info
  vector<AgentInformation> agent_list;
  vector<AgentInformation> gpu_list;
  vector<AgentInformation> cpu_list;

  // TODO add info
  // Get all agents
  void InitAgents();
  
  // TODO add info
  // Process one agent and put it in the list
  static hsa_status_t ProcessAgent(hsa_agent_t agent, void* data);
  
  // TODO add info
  // Process one agent and put it in the list
  static hsa_status_t ProcessMemPool(hsa_amd_memory_pool_t pool, void* data);
  
  // TODO add info
  static void print_hsa_status(string message, hsa_status_t st);

  // TODO add info
  double send_traffic(hsa_agent_t src_agent, hsa_agent_t dst_agent, hsa_amd_memory_pool_t src_buff, hsa_amd_memory_pool_t dst_buff, bool bidirectional, size_t curr_size);
  
  // TODO add info
  double GetCopyTime(bool bidirectional, hsa_signal_t signal_fwd, hsa_signal_t signal_rev);

};

#endif  // INCLUDE_RVSHSA_H_