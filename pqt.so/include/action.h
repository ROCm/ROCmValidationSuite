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
#ifndef PQT_SO_INCLUDE_ACTION_H_
#define PQT_SO_INCLUDE_ACTION_H_

#include <unistd.h>
#include <stdlib.h>
#include <assert.h>

#include <algorithm>
#include <cctype>
#include <sstream>
#include <limits>
#include <string>
#include <vector>

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

#include "include/rvsactionbase.h"

class pqtworker;

/**
 * @class pqt_action
 * @ingroup PQT
 *
 * @brief PQT action implementation class
 *
 * Derives from rvs::actionbase and implements actual action functionality
 * in its run() method.
 *
 */
class pqt_action : public rvs::actionbase {
 public:
  pqt_action();
  virtual ~pqt_action();

  virtual int run(void);

 protected:
  bool get_all_pqt_config_keys(void);
  bool get_all_common_config_keys(void);

  // PQT specific config keys
  bool property_get_peers(int *error);
  void property_get_test_bandwidth(int *error);
//  void property_get_log_interval(int *error);
  void property_get_bidirectional(int *error);

  //! 'true' if "all" is found under "peer" key for this action
  bool      prop_peer_device_all_selected;
  //! array of peer GPU IDs to be used in data trasfers
  std::vector<std::string> prop_peers;
  //! deviceid of peer GPUs
  uint32_t  prop_peer_deviceid;
  //! 'true' if bandwidth test is to be executed for verified peers
  bool prop_test_bandwidth;
  //! 'true' if bidirectional data transfer is required
  bool prop_bidirectional;
  //! list of test block sizes
  std::vector<uint32_t> block_size;
  //! set to 'true' if the default block sizes are to be used
  bool b_block_size_all;
  //! test block size for back-to-back transfers
  uint32_t b2b_block_size;
  //! link type
  int link_type;

 protected:
  int is_peer(uint16_t Src, uint16_t Dst);
  int create_threads();
  int destroy_threads();

  int run_single();
  int run_parallel();

  int print_running_average();
  int print_running_average(pqtworker* pWorker);

  int print_final_average();

  //! 'true' for the duration of test
  bool brun;

  //! bjson field indicates if the json flag is set
  bool bjson;

 private:
  void do_running_average(void);
  void do_final_average(void);

  std::vector<pqtworker*> test_array;
};

#endif  // PQT_SO_INCLUDE_ACTION_H_
