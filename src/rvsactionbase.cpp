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
#include <iostream>

#include "rvsloglp.h"
#include "rvs_key_def.h"
#include "rvs_util.h"

using std::cout;
using std::endl;
using std::string;
/**
 * @brief Default constructor.
 *
 * */
rvs::actionbase::actionbase() {
  property_log_level = 2;
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
  property.insert(property.cend(), std::pair<string, string>(pKey, pVal));
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
 * @return TRUE - property set, FALSE - otherwise
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
 * @return TRUE - property set, FALSE - otherwise
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
 * gets the gpu_id list from the module's properties collection
 * @param error pointer to a memory location where the error code will be stored
 * @return true if "all" is selected, false otherwise
 */
bool rvs::actionbase::property_get_device(int *error) {
  std::string val;
  *error = 0;
  if (!has_property(RVS_CONF_DEVICE_KEY, &val)) {
    RVSTRACE_
    *error = 2;
    return false;
  }

  if (val == "all") {
    return true;
  }
  // split the list of gpu_id
  device_prop_gpu_id_list = str_split(val, YAML_DEVICE_PROP_DELIMITER);

  if (device_prop_gpu_id_list.empty()) {
    RVSTRACE_
    *error = 2;  // list of gpu_id cannot be empty
    return false;
  }

  for (auto it_gpu_id = device_prop_gpu_id_list.begin();
       it_gpu_id != device_prop_gpu_id_list.end(); ++it_gpu_id) {
    RVSTRACE_
    if (!is_positive_integer(*it_gpu_id)) {
      RVSTRACE_
      *error = 1;
      return false;
    }
  }
  RVSTRACE_
  return false;
}

/**
 * @brief Reads boolean property value from properties collection
 */
int rvs::actionbase::property_get(const std::string& prop_name,
                                       bool* pVal) {
  std::string sval;
  if (!has_property(prop_name, &sval)) {
    return 2;
  }
  return rvs_util_parse(sval, pVal);
}

/**
 * @brief Reads boolean property value from properties collection.
 * Assigns the default value if not found.
 */
int rvs::actionbase::property_get(const std::string& prop_name,
                                       bool* pVal, bool bDef) {
  int sts = property_get(prop_name, pVal);
  if (sts == 2) {
    *pVal = bDef;
    return 0;
  }

  return sts;
}

/**
 * @brief Reads string property value from properties collection
 */
int rvs::actionbase::property_get(const std::string& prop_name,
                                       std::string* pVal) {
  if (!has_property(prop_name, pVal)) {
    return 2;
  }
  return 0;
}

/**
 * @brief Reads string property value from properties collection.
 * Assigns the default value if not found.
 */
int rvs::actionbase::property_get(
  const std::string& prop_name,
  std::string* pVal,
  const std::string& bDefault) {
  int sts = property_get(prop_name, pVal);
  if (sts == 2) {
    *pVal = bDefault;
    return 0;
  }

  return sts;
}

