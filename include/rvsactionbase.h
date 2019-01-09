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

#include "include/rvs_util.h"

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
  int property_get_device();

  /**
  * @brief Gets uint16_t list from the module's properties collection
  * @param key jey name
  * @param delimiter delimiter in YAML file
  * @param pval ptr to reulting list
  * @param pball ptr to flag to be set to 'true' when "all" is detected
  * @return 0 - OK
  * @return 1 - syntax error in 'device' configuration key
  * @return 2 - missing 'device' key
  */
  template <typename T>
  int property_get_uint_list(const std::string& key,
                                   const std::string& delimiter,
                                   std::vector<T>* pval,
                                   bool* pball) {
    std::string strval;

    // fetch key value if any
    if (!has_property(key, &strval)) {
      return 2;
    }

    // found and is "all" - set flag and return
    if (strval == "all") {
      *pball = true;
      pval->clear();
      return 0;
    } else {
      *pball = false;
    }

    // parse key value into std::vector<std::string>
    auto strarray = str_split(strval, delimiter);

    // convert str arary into uint16_t array
    int sts = rvs_util_strarr_to_uintarr<T>(strarray, pval);

    if (sts < 0) {
      pval->clear();
      return 1;
    }

    return 0;
  }


/**
 * @brief reads key integer type from the module's properties collection
 * returns 1 for invalid key, 2 for missing key 
*/    
  template <typename T>
  int property_get_int(const std::string& prop_name, T* key) {
    std::string val;
    int error = 0;  // init with 'no error'
    if (has_property(prop_name, &val)) {
       error = rvs_util_parse<T>(val, key);
    } else {
      error = 2;
    }
    return error;
  }

  /**
   * @brief reads key integer type from the module's properties collection
   * returns 1 for invalid key
   * takes default value if key is missing 
   */
  template <typename T>
  int property_get_int
  (const std::string& prop_name, T* key, T def_value) {
    std::string val;
    int error = 0;  // init with 'no error'
    if (has_property(prop_name, &val)) {
      error = rvs_util_parse<T>(val, key);
    } else {
      *key = def_value;
    }
    return error;
  }

  int property_get(const std::string& prop_name, bool* pVal);

  int property_get(const std::string& prop_name, std::string* pVal);

  int property_get(const std::string& prop_name, float* pVal);

  /**
   * @brief reads key value from the module's properties collection
   * returns 1 for invalid key
   * takes default value if key is missing
   */
  template <typename T>
  int property_get(const std::string& prop_name, T* pVal, T Default) {
    int sts = property_get(prop_name, pVal);
    if (sts == 2) {
      *pVal = Default;
      return 0;
    }

    return sts;
  }

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
  //! device_id - non-zero if filtering of device id is required
  uint16_t property_device_id;
  //! array of GPU IDs listed in config 'device' key
  std::vector<uint16_t> property_device;
  //! 'true' when all devices are selected ('device: all')
  bool property_device_all;
  //! TRUE if the action will run on all selected devices in parallel
  bool property_parallel;
  //! number of stress test iterations to run
  uint64_t property_count;
  //! stress test run delay
  uint64_t property_wait;
  //! stress test run duration
  uint64_t property_duration;
  //! logging interval
  uint64_t property_log_interval;

  //! data from config file
  std::map<std::string, std::string> property;

//   //! List of all gpu_id in the action's "device" property in .config file
//   std::vector<std::string> device_prop_gpu_id_list;

  //! logging level
  int property_log_level;
};

}  // namespace rvs
#endif  // INCLUDE_RVSACTIONBASE_H_
