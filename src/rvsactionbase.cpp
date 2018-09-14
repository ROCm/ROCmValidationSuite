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
#include "rvsactionbase.h"

#include <unistd.h>
#include <chrono>
#include <utility>
#include <map>
#include <string>
#include <vector>

#include "rvs_util.h"

/**
 * @brief Default constructor.
 *
 * */
rvs::actionbase::actionbase() {
}

/**
 * @brief Default destructor.
 *
 * */
rvs::actionbase::~actionbase() {
}

/**
 * @brief Sets action property
 *
 * @param pKey Property key
 * @param pVal Property value
 * @return 0 - success. non-zero otherwise
 *
 * */
int rvs::actionbase::property_set(const char* pKey, const char* pVal) {
  property.insert(std::pair<string, string>(pKey, pVal));
  return 0;
}

/**
 * @brief Pauses current thread for the given time period
 *
 * @param ms Sleep time in milliseconds.
 * @return (void)
 *
 * */
void rvs::actionbase::sleep(const unsigned int ms) {
  ::usleep(1000*ms);
}

/**
 * @brief Checks if property is set.
 *
 * Returns value if propety is set.
 *
 * @param key Property key
 * @param pval Property value. Not changed if propery is not set.
 * @return TRUE - property set, FALSE - othewise
 *
 * */
bool rvs::actionbase::has_property(const std::string& key, std::string* pval) {
  auto it = property.find(key);
  if (it != property.end()) {
    *pval = it->second;
    return true;
  }
  return false;
}

/**
 * @brief Checks if property is set.
 *
 * @param key Property key
 * @return TRUE - property set, FALSE - othewise
 *
 * */
bool rvs::actionbase::has_property(const std::string& key) {
  string val;
  return has_property(key, &val);
}

/**
 * @brief Gets uint16_t list from the module's properties collection
 * @param key jey name
 * @param delimiter delimiter in YAML file
 * @param pval ptr to reulting list
 * @param pball ptr to flag to be set to 'true' when "all" is detected
 * @param error ptr to error: 0 - OK, 1 - syntax error, 2 - not found
 */
void rvs::actionbase::property_get_uint_list(const std::string& key,
                                   const std::string& delimiter,
                                   std::vector<uint32_t>* pval,
                                   bool* pball,
                                   int *error) {
  bool        bfound = false;
  std::string strval;

  // init with 'no error'
  *error = 0;

  // fetch key value if any
  bfound = has_property(key, &strval);

  // key not found - return
  if (!bfound) {
    *error = 2;
    return;
  }

  // found and is "all" - set flag and return
  if (strval == "all") {
    *pball = true;
    pval->clear();
    return;
  } else {
    *pball = false;
  }

  // parse key value into std::vector<std::string>
  auto strarray = str_split(strval, delimiter);

  // convert str arary into uint16_t array
  int sts = rvs_util_strarr_to_uintarr(strarray, pval);

  if (sts < 0) {
    *error = 1;
    pval->clear();
  }
}


/**
 * gets the deviceid from the module's properties collection
 * @param error pointer to a memory location where the error code will be stored
 * @return deviceid value if valid, -1 otherwise
 */
int rvs::actionbase::property_get_deviceid(int *error) {
    auto it = property.find(RVS_CONF_DEVICEID_KEY);
    int deviceid = -1;
    *error = 0;  // init with 'no error'

    if (it != property.end()) {
        if (it->second != "") {
            if (is_positive_integer(it->second)) {
                deviceid = std::stoi(it->second);
            } else {
                *error = 1;  // we have something but it's not a number
            }
        } else {
            *error = 1;  // we have an empty string
        }
    }
    return deviceid;
}

/**
 * gets the gpu_id list from the module's properties collection
 * @param error pointer to a memory location where the error code will be stored
 * @return true if "all" is selected, false otherwise
 */
bool rvs::actionbase::property_get_device(int *error) {
    *error = 0;  // init with 'no error'
    auto it = property.find(RVS_CONF_DEVICE_KEY);
    if (it != property.end()) {
        if (it->second == "all") {
            return true;
        } else {
            // split the list of gpu_id
            device_prop_gpu_id_list = str_split(it->second,
                    YAML_DEVICE_PROP_DELIMITER);

            if (device_prop_gpu_id_list.empty()) {
                *error = 1;  // list of gpu_id cannot be empty
            } else {
                for (vector<string>::iterator it_gpu_id =
                        device_prop_gpu_id_list.begin();
                        it_gpu_id != device_prop_gpu_id_list.end(); ++it_gpu_id)
                    if (!is_positive_integer(*it_gpu_id)) {
                        *error = 1;
                        break;
                    }
            }
            return false;
        }

    } else {
        *error = 1;
        // when error is set, it doesn't really matter whether the method
        // returns true or false
        return false;
    }
}

/**
 * @brief gets the action name from the module's properties collection
 */
void rvs::actionbase::property_get_action_name(int *error) {
  action_name = "[]";
  auto it = property.find(RVS_CONF_NAME_KEY);
  if (it != property.end()) {
    action_name = it->second;
    *error = 0;
  } else {
    *error = 2;
  }
}

/**
 * @brief reads the module's properties collection to see whether the GST should
 * run the stress test in parallel
 */
void rvs::actionbase::property_get_run_parallel(int *error) {
  gst_runs_parallel = false;
  auto it = property.find(RVS_CONF_PARALLEL_KEY);
  if (it != property.end()) {
    if (it->second == "true") {
      gst_runs_parallel = true;
      *error = 0;
    } else if (it->second == "false") {
      *error = 0;
    } else {
      *error = 1;
    }
  } else {
    *error = 2;
  }
}

/**
 * @brief reads the run count from the module's properties collection
 */
void rvs::actionbase::property_get_run_count(int *error) {
  gst_run_count = 1;
  auto it = property.find(RVS_CONF_COUNT_KEY);
  if (it != property.end()) {
    if (is_positive_integer(it->second)) {
      gst_run_count = std::stoi(it->second);
      *error = 0;
    } else {
      *error = 1;
    }
  } else {
    *error = 2;
  }
}

/**
 * @brief reads the module's properties collection to check how much to delay
 * each stress test session
 */
void rvs::actionbase::property_get_run_wait(int *error) {
  gst_run_wait_ms = 0;
  auto it = property.find(RVS_CONF_WAIT_KEY);
  if (it != property.end()) {
    if (is_positive_integer(it->second)) {
      gst_run_wait_ms = std::stoul(it->second);
      *error = 0;
    } else {
      *error = 1;
    }
  } else {
    *error = 2;
  }
}

/**
 * @brief reads the total run duration from the module's properties collection
 */
void rvs::actionbase::property_get_run_duration(int *error) {
  gst_run_duration_ms = 0;
  auto it = property.find(RVS_CONF_DURATION_KEY);
  if (it != property.end()) {
    if (is_positive_integer(it->second)) {
      gst_run_duration_ms = std::stoul(it->second);
      *error = 0;
    } else {
      *error = 1;
    }
  } else {
    *error = 2;
  }
}

/**
 * @brief reads the sample interval from the module's properties collection
 */
int rvs::actionbase::property_get_sample_interval(int *error) {
  int sample_int = -1;
  auto it = property.find(RVS_CONF_SAMPLE_INTERVAL_KEY);
  if (it != property.end()) {
    if (is_positive_integer(it->second)) {
      sample_int = std::stoi(it->second);
      property.erase(it);
      *error = 0;
    } else {
      *error = 1;
    }
  } else {
    *error = 2;
  }
  return sample_int;
}

/**
 * @brief reads the log interval from the module's properties collection
 */
int rvs::actionbase::property_get_log_interval(int *error) {
  int log_int = -1;
  auto it = property.find(RVS_CONF_LOG_INTERVAL_KEY);
  if (it != property.end()) {
    if (is_positive_integer(it->second)) {
      log_int = std::stoul(it->second);
      property.erase(it);
      *error = 0;
    } else {
      *error = 1;
    }
  } else {
    *error = 2;
  }
  return log_int;
}

/**
 * @brief reads terminate from the module's properties collection
 */
bool rvs::actionbase::property_get_terminate(int *error) {
  bool term = -1;
  auto it = property.find(RVS_CONF_TERMINATE_KEY);
  if (it != property.end()) {
    if (it->second == "true") {
      term = true;
      property.erase(it);
    } else if (it->second == "false") {
      term = false;
      property.erase(it);
    } else {
      *error = 0;
    }
  }

  return term;
}

