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
#include <vector>
#include <memory>
#include <map>

#include "rvsthreadbase.h"
#include "rocm_smi/rocm_smi.h"


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
  Worker();
  virtual ~Worker();

  void stop(void);
  //! Sets initiating action name
  void set_name(const std::string& name) { action_name = name; }
  //! sets stopping action name
  void set_stop_name(const std::string& name) { stop_action_name = name; }
  //! Sets device id for filtering
  void set_deviceid(const int id) { device_id = id; }
  //! Sets GPU IDs for filtering
  void set_gpuids(const std::vector<uint16_t>& GpuIds);
  //! Sets GPU IDs for filtering (string used in messages)
  //! @param Devices List of devices to monitor
  void set_strgpuids(const std::string& Devices) { strgpuids = Devices; }
  //! Sets JSON flag
  void json(const bool flag) { bjson = flag; }
  //! Returns initiating action name
  const std::string& get_name(void) { return action_name; }
  //! sets sample interval
  void set_sample_int(int interval) { sample_interval = interval; }
  //! sets log interval
  void set_log_int(int interval) { log_interval = interval; }
  //! sets terminate key
  void set_terminate(bool term_true) { term = term_true; }
  //! sets true/false for metric
  void set_metr_mon(std::string metr_name, bool metr_true);
  //! sets bound values for metric
  void set_bound(std::string metr_name, bool met_bound, int metr_max,
                 int metr_min);
  //! gets irq of device
  const std::string get_irq(const std::string path);
  //! gets power of device
  int get_power(const std::string path);
  //! prints captured metric values
  void do_metric_values(void);

 protected:
  virtual void run(void);

 protected:
  //! TRUE if JSON output is required
  bool bjson;
  //! Loops while TRUE
  bool brun;
  //! device id to filter for. 0 if no filtering.
  int device_id;
  //! GPU id filtering flag
  bool bfiltergpu;
  //! list of GPU devices to monitor
  std::vector<uint16_t> gpuids;
  //! list of GPU devices to monitor (string used in messages)
  std::string strgpuids;
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
  //! number of times of get metric
  int count;
//! gpu_id and average metrics values
struct Dev_metrics {
    //! gpu_id
    uint32_t gpu_id;
    //! average temperature
    uint32_t av_temp;
    //! average clock
    uint32_t av_clock;
    //! average mem_clock
    uint32_t av_mem_clock;
    //! average fan
    uint32_t av_fan;
    //! average power
    uint32_t av_power;
};
//! monitored metric and its bound values
struct Metric_bound {
    //! true if metric observed
    bool mon_metric;
    //! true if bounds checked
    bool check_bounds;
    //! bound max_val
    int max_val;
    //! bound min_val
    int min_val;
};
//! number of violations for metrics
struct Metric_violation {
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
    //! current temperature value
    int temp;
    //! current clock value
    int clock;
    //! current mem_clock value
    int mem_clock;
    //! current fan value
    int fan;
    //! current power value
    int power;
};

  //! device irq, gpu_id and average metric
  std::map<std::string, Dev_metrics> irq_gpu_ids;
  //! device_irq and metric bounds
  std::map<std::string, Metric_bound> bounds;
  //! device_irq and metrics violation
  std::map<std::string, Metric_violation> met_violation;
  //! device_irq and current metric values
  std::map<std::string, Metric_value> met_value;
};

#endif  // GM_SO_INCLUDE_WORKER_H_
