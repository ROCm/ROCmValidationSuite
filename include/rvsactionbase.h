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
#ifndef INCLUDE_RVSACTIONBASE_H_
#define INCLUDE_RVSACTIONBASE_H_

#include <map>
#include <string>
#include <vector>

#define YAML_DEVICE_PROPERTY_ERROR      "Error while parsing <device> property"
#define YAML_DEVICEID_PROPERTY_ERROR    "Error while parsing <deviceid> "\
                                        "property"
#define YAML_TARGET_STRESS_PROP_ERROR   "Error while parsing <target_stress> "\
                                        "property"
#define YAML_DEVICE_PROP_DELIMITER      " "
#define YAML_REGULAR_EXPRESSION_ERROR   "Regular expression error"

#define KFD_QUERYING_ERROR              "An error occurred while querying "\
                                        "the GPU properties"

#define RVS_CONF_NAME_KEY               "name"
#define RVS_CONF_DEVICE_KEY             "device"
#define RVS_CONF_PARALLEL_KEY           "parallel"
#define RVS_CONF_COUNT_KEY              "count"
#define RVS_CONF_WAIT_KEY               "wait"
#define RVS_CONF_DURATION_KEY           "duration"
#define RVS_CONF_DEVICEID_KEY           "deviceid"
#define RVS_JSON_LOG_GPU_ID_KEY         "gpu_id"
#define RVS_CONF_SAMPLE_INTERVAL_KEY    "sample_interval"
#define RVS_CONF_LOG_INTERVAL_KEY       "log_interval"
#define RVS_CONF_TERMINATE_KEY          "terminate"

namespace rvs {
/**
 * @class actionbase
 * @ingroup RVS
 * @brief Base class for all module level actions
 *
 */

class actionbase {
 public:
  virtual ~actionbase();

 protected:
  actionbase();
  void sleep(const unsigned int ms);

 public:
  virtual int     property_set(const char*, const char*);

  //! Virtual action function. To be implemented in every derived class.
  virtual int     run(void) = 0;
  bool has_property(const std::string& key, std::string* pval);
  bool has_property(const std::string& key);
  int  property_get_deviceid(int *error);
  bool property_get_device(int *error);

  void property_get_action_name(int *error);
  void property_get_run_parallel(int *error);
  void property_get_run_count(int *error);
  void property_get_run_wait(int *error);
  void property_get_run_duration(int *error);
  int  property_get_sample_interval(int *error);
  int  property_get_log_interval(int *error);
  bool property_get_terminate(int* error);
  void property_get_uint_list(const std::string& key,
                                   const std::string& delimiter,
                                   std::vector<uint32_t>* pval,
                                   bool* pball,
                                   int *error);

 protected:
/**
 *  @brief Collection of properties
 *
 * Properties represent:
 *  - content of corresponding "action" tag in YAML .conf file
 *  - command line arguments given when invoking rvs
 *  - other parameters given for specific module actions (see module action for help)
 */

  //! name of the action
  std::string action_name;
  //! TRUE if the GST action will run on all selected devices in parallel
  bool gst_runs_parallel;
  //! number of GST stress test iterations to run
  uint64_t gst_run_count;
  //! stress test run delay
  uint64_t gst_run_wait_ms;
  //! stress test run duration
  uint64_t gst_run_duration_ms;

  //! data from config file
  std::map<std::string, std::string>  property;

  //! List of all gpu_id in the action's "device" property in .config file
  std::vector<std::string> device_prop_gpu_id_list;
};
}  // namespace rvs
#endif  // INCLUDE_RVSACTIONBASE_H_
