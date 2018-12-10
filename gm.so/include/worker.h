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
#ifndef GM_SO_INCLUDE_WORKER_H_
#define GM_SO_INCLUDE_WORKER_H_

#include <string>
#include <map>

#include "include/rvsthreadbase.h"


/**
 * @class Worker
 * @ingroup GM
 *
 * @brief Monitoring implementation class
 *
 * Derives from rvs::ThreadBase and implements actual monitoring functionality
 * in its run() method.
 *
 */

class Worker : public rvs::ThreadBase {
 public:
  //! monitored metric and its bound values
  struct Metric_bound {
    //! true if metric observed
    bool mon_metric;
    //! true if bounds checked
    bool check_bounds;
    //! bound max_val
    uint32_t max_val;
    //! bound min_val
    uint32_t min_val;
  };
  //! number of violations for metrics
  struct Metric_violation {
    //! gpu_id
    int32_t gpu_id;
    //! number of temperature violation
    int temp_violation;
    //! number of clock violation
    int clock_violation;
    //! number of mem_clock violation
    int mem_clock_violation;
    //! number of fan violation
    int fan_violation;
    //! number of power violation
    int power_violation;
  };
  //! current metric values
  struct Metric_value {
    //! gpu_id
    int32_t gpu_id;
    //! current temperature value
    uint32_t temp;
    //! current clock value
    uint32_t clock;
    //! current mem_clock value
    uint32_t mem_clock;
    //! current fan value
    uint32_t fan;
    //! current power value
    uint32_t power;
  };
  //! average metric values
  struct Metric_avg {
    //! gpu_id
    int32_t gpu_id;
    //! average temperature
    uint32_t av_temp;
    //! average clock
    uint32_t av_clock;
    //! average mem_clock
    uint32_t av_mem_clock;
    //! average fan
    uint32_t av_fan;
    //! average power
    float av_power;
  };

 public:
  Worker();
  virtual ~Worker();

  void stop(void);
  //! Sets initiating action name
  void set_name(const std::string& name) { action_name = name; }
  //! sets stopping action name
  void set_stop_name(const std::string& name) { stop_action_name = name; }
  //! Sets device indices for filtering
  void set_dv_ind(const std::map<uint32_t, int32_t>& DvInd) {
    dv_ind = DvInd;
  }
  //! Sets JSON flag
  void json(const bool flag) { bjson = flag; }
  //! Returns initiating action name
//  const std::string& get_name(void) { return action_name; }
  //! sets sample interval
  void set_sample_int(int interval) { sample_interval = interval; }
  //! sets log interval
  void set_log_int(int interval) { log_interval = interval; }
  //! sets terminate key
  void set_terminate(bool term_true) { term = term_true; }
  //! sets force key
  void set_force(bool flag) { force = flag; }
  //! sets true/false for metric
  void set_metr_mon(std::string metr_name, bool metr_true);
  //! sets bound values for metric
  void set_bound(const std::map<std::string, Metric_bound>& Bound) {
    bounds = Bound;
  }
  //! gets irq of device
  const std::string get_irq(const std::string path);
  //! gets power of device
  int get_power(const std::string path);
  //! prints captured metric values
  void do_metric_values(void);

 protected:
  virtual void run(void);

 protected:
  //! Name of the action which initiated monitoring
  std::string  action_name;
  //! Name of the action which stops monitoring
  std::string  stop_action_name;
  //! sample interval
  int sample_interval;
  //! log interval;
  int log_interval;
  //! terminate key
  bool term;
  //! force key
  bool force;
  //! TRUE if JSON output is required
  bool bjson;
  //! Loops while TRUE
  bool brun;
  //! list of rocm_smi_lib device indices to monitor
  std::map<uint32_t, int32_t> dv_ind;
  //! number of times of get metric
  int count;
  //! dv_ind and metric bounds
  std::map<std::string, Metric_bound> bounds;
  //! dv_ind and metrics violation
  std::map<uint32_t, Metric_violation> met_violation;
  //! dv_ind and current metric values
  std::map<uint32_t, Metric_value> met_value;
  //! dv_ind and current metric values
  std::map<uint32_t, Metric_avg> met_avg;
};

#endif  // GM_SO_INCLUDE_WORKER_H_
