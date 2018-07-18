

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
  
  // TODO add info
  vector<hsa_agent_t> agent_list;
  vector<hsa_agent_t> gpu_list;
  vector<hsa_agent_t> cpu_list;

  // TODO add info
//   hsa_status_t status;

  // TODO add info
//   string log_msg;
  
  // TODO add info
  // Get all agents
  void GetAgents();
  
  // TODO add info
  // Process one agent and put it in CPU or GPU list
  static hsa_status_t ProcessAgent(hsa_agent_t agent, void* data);

  // TODO add info
  static void print_hsa_status(string message, hsa_status_t st);
  
protected:
  int do_gpu_list(void);
};

#endif /* ACTION_H_ */
