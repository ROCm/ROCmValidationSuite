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
#include <unistd.h>
#include <assert.h>

#include <algorithm>
#include <cctype>
#include <sstream>
#include <limits>
#include <string>
#include <vector>

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

using std::string;
using std::vector;

namespace rvs {

// TODO(dmatichdl) add info
class hsa {
 public:
  hsa();
  virtual ~hsa();

  // agent structure
  struct AgentInformation {
    // global agent information
    hsa_agent_t                   agent;
    string                        agent_name;
    string                        agent_device_type;
    uint32_t                      node;
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

  // TODO(dmatichdl) add info
  vector<AgentInformation> agent_list;
  vector<AgentInformation> gpu_list;
  vector<AgentInformation> cpu_list;

 public:
  static void Init();
  static void Terminate();
  static rvs::hsa* Get();

  const int FindAgent(uint32_t Node);

  // TODO(mlucinhdl) add info
  int SendTraffic(uint32_t SrcNode, uint32_t DstNode,
                  size_t   Size,    bool     bidirectional,
                  double*  Duration
                 );

  // TODO(dmatichdl) add info
  double send_traffic(hsa_agent_t src_agent, hsa_agent_t dst_agent,
                      hsa_amd_memory_pool_t src_buff,
                      hsa_amd_memory_pool_t dst_buff,
                      bool bidirectional, size_t curr_size);


 protected:
  // TODO(dmatichdl) add info
  // Get all agents
  void InitAgents();

  int Allocate(int SrcAgent, int DstAgent, size_t Size,
                     hsa_amd_memory_pool_t* pSrcPool, void** SrcBuff,
                     hsa_amd_memory_pool_t* pDstPool, void** DstBuff);

  // TODO(dmatichdl) add info
  // Process one agent and put it in the list
  static hsa_status_t ProcessAgent(hsa_agent_t agent, void* data);

  // TODO(dmatichdl) add info
  // Process one agent and put it in the list
  static hsa_status_t ProcessMemPool(hsa_amd_memory_pool_t pool, void* data);

  // TODO(dmatichdl) add info
  static void print_hsa_status(string message, hsa_status_t st);
  // TODO(dmatichdl) add info
  static void print_hsa_status(const std::string& file, int line, const std::string& function, hsa_status_t st);

  // TODO(dmatichdl) add info
  double GetCopyTime(bool bidirectional,
                     hsa_signal_t signal_fwd, hsa_signal_t signal_rev);
 protected:
   static rvs::hsa* pDsc;
};

}  // namespace rvs
#endif  // INCLUDE_RVSHSA_H_
