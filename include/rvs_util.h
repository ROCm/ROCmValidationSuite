/********************************************************************************
 *
 * Copyright (c) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef INCLUDE_RVS_UTIL_H_
#define INCLUDE_RVS_UTIL_H_

#include <vector>
#include <string>
#include <iostream>
#include <map>
#include "include/rvsloglp.h"
#include "include/gpu_util.h"

using std::map;

struct action_descriptor{
  std::string action_name;
  std::string module_name;
  uint16_t gpu_id;
};
extern bool is_positive_integer(const std::string& str_val);

extern std::vector<std::string> str_split(const std::string& str_val,
        const std::string& delimiter);

/**
 * Convert array of strings into array of signed integers of type T
 * @param sArr input string
 * @param iArr tokens' delimiter
 * @return -1 if error, number of elements in the array otherwise
 */
template <typename T>
int rvs_util_strarr_to_intarr(const std::vector<std::string>& sArr,
                              std::vector<T>* piArr) {
  piArr->clear();

  for (auto it = sArr.begin(); it != sArr.end(); ++it) {
    try {
      if (is_positive_integer(*it)) {
        piArr->push_back(std::stoi(*it));
      }
    }
    catch(...) {
    }
  }

  if (sArr.size() != piArr->size())
    return -1;

  return piArr->size();
}


/**
 * Convert array of strings into array of unsigned integers of type T
 * @param sArr input string
 * @param iArr tokens' delimiter
 * @return -1 if error, number of elements in the array otherwise
 */
template <typename T>
int rvs_util_strarr_to_uintarr(const std::vector<std::string>& sArr,
                              std::vector<T>* piArr) {
  piArr->clear();

  for (auto it = sArr.begin(); it != sArr.end(); ++it) {
    try {
      if (is_positive_integer(*it)) {
        piArr->push_back(std::stoul(*it));
      }
    }
    catch(...) {
    }
  }

  if (sArr.size() != piArr->size())
    return -1;

  return piArr->size();
}


extern int rvs_util_parse(const std::string& buff, bool* pval);

/**
 * @brief turns string value into right type of integer, else returns error
 */

template <typename T>
int rvs_util_parse(const std::string& buff,
                                    T* pval) {
  int error;
  if (buff.empty()) {  // method empty
    error = 2;
  } else {
    if (is_positive_integer(buff)) {
      try {
        *pval = std::stoul(buff);
        error = 0;
      } catch(...) {
        error = 1;  // we have an empty string
      }
    } else {
      error = 1;
    }
  }
  return error;
}


void *json_node_create(std::string module_name, std::string action_name,
                     int log_level);
bool fetch_gpu_list(int hip_num_gpu_devices, map<int, uint16_t>& gpus_device_index,
    const std::vector<uint16_t>& property_device, const int& property_device_id,
    bool property_device_all, const std::vector<uint16_t>& property_device_index,
    bool property_device_index_all, bool mcm_check = false);
void getBDF(int idx ,unsigned int& domain,unsigned int& bus,unsigned int& device,unsigned int& function);
int display_gpu_info(void);
void *json_list_create(std::string lname, int log_level);

template <typename... KVPairs>
void log_to_json(action_descriptor desc, int log_level, KVPairs...  key_values ) {
        std::vector<std::string> kvlist{key_values...};
    if  (kvlist.size() == 0 || kvlist.size() %2 != 0){
            return;
    }
    void *json_node = json_node_create(desc.module_name,
        desc.action_name.c_str(), log_level);
    if (json_node) {

      rvs::lp::AddString(json_node, "gpu_id", std::to_string(desc.gpu_id));

      uint16_t gpu_index = 0;
      rvs::gpulist::gpu2gpuindex(desc.gpu_id, &gpu_index);
      rvs::lp::AddString(json_node, "gpu_index", std::to_string(gpu_index));

      for (int i =0; i< kvlist.size()-1; i +=2){
          rvs::lp::AddString(json_node, kvlist[i], kvlist[i+1]);
      }

      rvs::lp::LogRecordFlush(json_node, log_level);
    }
}
void json_add_primary_fields(std::string moduleName, std::string action_name);
void cleanup_logs();
#endif  // INCLUDE_RVS_UTIL_H_
