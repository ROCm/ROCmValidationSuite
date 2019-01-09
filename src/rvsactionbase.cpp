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
#include "include/rvsactionbase.h"

#include <unistd.h>
#include <chrono>
#include <utility>
#include <regex>
#include <map>
#include <string>
#include <vector>
#include <iostream>

#include "include/rvsloglp.h"
#include "include/rvs_key_def.h"
#include "include/rvs_util.h"

#define FLOATING_POINT_REGEX            "^[0-9]*\\.?[0-9]+$"

using std::cout;
using std::endl;
using std::string;
/**
 * @brief Default constructor.
 *
 * */
rvs::actionbase::actionbase() {
  property_log_level = 2;
  property_device_all = true;
  property_device_id = 0u;
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
 * gets the gpu_id list from the module's properties collection
 * @return 0 - OK
 * @return 1 - syntax error in 'device' configuration key
 * @return 2 - missing 'device' key
 */
int rvs::actionbase::property_get_device() {
  return property_get_uint_list<uint16_t>(
    RVS_CONF_DEVICE_KEY,
    YAML_DEVICE_PROP_DELIMITER,
    &property_device,
    &property_device_all);
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
 * @brief Reads float property value from properties collection
 */
int rvs::actionbase::property_get(const std::string& prop_name,
                                       float* pVal) {
  std::string sval;
  if (!has_property(prop_name, &sval)) {
    return 2;
  }
  try {
    *pVal = std::stof(sval);
  } catch (...) {
      return 1;  // something went wrong with the regex
  }
  return 0;
}
