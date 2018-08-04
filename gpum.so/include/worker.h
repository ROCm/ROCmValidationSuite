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
#ifndef GPUM_SO_INCLUDE_WORKER_H_
#define GPUM_SO_INCLUDE_WORKER_H_

#include <string>
#include <vector>
#include <memory>
#include <map>
#include "rocm_smi/rocm_smi.h"

#include "rvsthreadbase.h"


/**
 * @class Worker
 * @ingroup GPUM
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
  void set_gpuids(const std::vector<int>& GpuIds);
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

 protected:
  virtual void run(void);

 protected:
  //! TRUE if JSON output is required
  bool bjson;
  std::shared_ptr<amd::smi::Device> dev;
  //! Loops while TRUE
  bool brun;
  //! device id to filter for. 0 if no filtering.
  int device_id;
  //! GPU id filtering flag
  bool bfiltergpu;
  //! list of GPU devices to monitor
  std::vector<int> gpuids;
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

struct Dev_metrics {
    uint32_t gpu_id;
    uint32_t av_temp;
    uint32_t av_clock;
    uint32_t av_mem_clock;
    uint32_t av_fan;
    uint32_t av_power;
};

struct Metric_bound {
    bool mon_metric;
    bool check_bounds;
    int max_val;
    int min_val;
};

struct Metric_violation {
    int temp_violation;
    int clock_violation;
    int mem_clock_violation;
    int fan_violation;
    int power_violation;
};

  std::map<std::string, Dev_metrics> irq_gpu_ids;
  std::map<std::string, Metric_bound> bounds;
  std::map<std::string, Metric_violation> met_violation;
};

#endif  // GPUM_SO_INCLUDE_WORKER_H_
