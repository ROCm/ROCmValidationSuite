

#ifndef ACTION_H_
#define ACTION_H_

#include <stdlib.h>
#include <assert.h>
#include <algorithm>
#include <unistd.h>
#include <cctype>
#include <sstream>
#include <limits>

#include "rvsactionbase.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

using std::string;
using std::vector;

/**
 * @class action
 * @ingroup PQT
 *
 * @brief PQT action implementation class
 *
 * Derives from rvs::actionbase and implements actual action functionality
 * in its run() method.
 *
 */
class action : public rvs::actionbase
{
public:
  action();
  virtual ~action();

  virtual int run(void);
  
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
  void GetAgents();
  
  // TODO add info
  // Process one agent and put it in the list
  static hsa_status_t ProcessAgent(hsa_agent_t agent, void* data);
  
  // TODO add info
  // Process one agent and put it in the list
  static hsa_status_t ProcessMemPool(hsa_amd_memory_pool_t pool, void* data);
  
  // TODO add info
  static void print_hsa_status(string message, hsa_status_t st);

  // TODO add info
  void send_p2p_traffic(hsa_agent_t src_agent, hsa_agent_t dst_agent, hsa_amd_memory_pool_t src_buff, hsa_amd_memory_pool_t dst_buff, bool bidirectional, size_t src_max_size, size_t dst_max_size);
  
protected:
  int do_gpu_list(void);
};

#endif /* ACTION_H_ */
