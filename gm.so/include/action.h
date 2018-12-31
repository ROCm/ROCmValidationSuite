/*******************************************************************************
 *
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

#ifndef GM_SO_INCLUDE_ACTION_H_
#define GM_SO_INCLUDE_ACTION_H_

#include <string>
#include <map>

#include "include/worker.h"
#include "include/rvsactionbase.h"

using std::string;

/**
 * @class gm_action
 * @ingroup GM
 *
 * @brief GM action implementation class
 *
 * Derives from rvs::actionbase and implements actual action functionality
 * in its run() method.
 *
 */

class gm_action : public rvs::actionbase {
 public:
    gm_action();
    virtual ~gm_action();

    virtual int run(void);

 protected:
/**
 * @brief gets the number of ROCm compatible AMD GPUs
 * @return run number of GPUs
 */
  int get_num_amd_gpu_devices(void);
  bool get_all_common_config_keys(void);
  bool get_all_gm_config_keys(void);
  int get_bounds(const char* pMetric);

 protected:
  //! 'true' if JSON logging is required
  bool     bjson;
  //! true if test has to be aborted on bounds violation
  bool     prop_terminate;
  //! true if forced termination is required
  bool     prop_force;
  //! configuration 'sample_interval'' key
  uint64_t sample_interval;

 protected:
  //! device_irq and metric bounds
  std::map<std::string, Worker::Metric_bound> property_bounds;

 private:
  //! JSON roor node helper var
  void* json_root_node;
};

#endif  // GM_SO_INCLUDE_ACTION_H_
