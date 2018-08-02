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

/**
 * @class hsa
 * @ingroup RVS
 *
 * @brief Wrapper class for HSA functionality needed for rvs tests
 *
 */
class hsa {
 public:
  //! Default constructor
  hsa();
  //! Default destructor
  virtual ~hsa();

/**
 * @class AgentInformation
 * @ingroup RVS
 *
 * @brief Utility class used to store HSA agent information
 *
 */
  struct AgentInformation {
    //! HSA agent handle
    hsa_agent_t                   agent;
    //! agent name
    string                        agent_name;
    //! device type, can be "GPU" or "CPU"
    string                        agent_device_type;
    //! NUMA node this agent belongs to
    uint32_t                      node;
    //! system memory pool
    hsa_amd_memory_pool_t         sys_pool;
    //! vector of memory pool HSA handles as reported during mem pool enumeration
    vector<hsa_amd_memory_pool_t> mem_pool_list;
    //! vecor of mem pools max sizes (index alligned with mem_pool_list)
    vector<size_t>                max_size_list;
  };


  //! list of test transfer sizes
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

  //! same as DEFAULT_SIZE_LIST but as std::vector
  vector<uint32_t> size_list;

  //! array of all found HSA agents
  vector<AgentInformation> agent_list;
  //! array of HSA GPU agents
  vector<AgentInformation> gpu_list;
  //! array of HSA CPU agents
  vector<AgentInformation> cpu_list;

 public:
  static void Init();
  static void Terminate();
  static rvs::hsa* Get();

  const int FindAgent(uint32_t Node);

  int SendTraffic(uint32_t SrcNode, uint32_t DstNode,
                  size_t   Size,    bool     bidirectional,
                  double*  Duration);

//   double send_traffic(hsa_agent_t src_agent, hsa_agent_t dst_agent,
//                       hsa_amd_memory_pool_t src_buff,
//                       hsa_amd_memory_pool_t dst_buff,
//                       bool bidirectional, size_t curr_size);

 protected:
  void InitAgents();

  int Allocate(int SrcAgent, int DstAgent, size_t Size,
                     hsa_amd_memory_pool_t* pSrcPool, void** SrcBuff,
                     hsa_amd_memory_pool_t* pDstPool, void** DstBuff);

  static hsa_status_t ProcessAgent(hsa_agent_t agent, void* data);
  static hsa_status_t ProcessMemPool(hsa_amd_memory_pool_t pool, void* data);
  static void print_hsa_status(string message, hsa_status_t st);
  static void print_hsa_status(const std::string& file, int line,
                               const std::string& function, hsa_status_t st);
  double GetCopyTime(bool bidirectional,
                     hsa_signal_t signal_fwd, hsa_signal_t signal_rev);

 protected:
  //! pointer to RVS HSA singleton
  static rvs::hsa* pDsc;
};

}  // namespace rvs
#endif  // INCLUDE_RVSHSA_H_
