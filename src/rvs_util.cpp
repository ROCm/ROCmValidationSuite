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
#include "include/rvs_util.h"
#include <vector>
#include <string>
#include <regex>
#include <iomanip>
#include <algorithm>
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include "amd_smi/amdsmi.h"


/**
 * splits a std::string based on a given delimiter
 * @param str_val input std::string
 * @param delimiter tokens' delimiter
 * @return std::vector containing all tokens
 */

std::vector<std::string> str_split(const std::string& str_val,
                                   const std::string& delimiter) {
    std::vector<std::string> str_tokens;
    int prev_pos = 0, cur_pos = 0;
    do {
        cur_pos = str_val.find(delimiter, prev_pos);
        if (cur_pos == std::string::npos)
            cur_pos = str_val.length();
        std::string token = str_val.substr(prev_pos, cur_pos - prev_pos);
        if (!token.empty())
            str_tokens.push_back(token);
        prev_pos = cur_pos + delimiter.length();
    } while (cur_pos < str_val.length() && prev_pos < str_val.length());
    return str_tokens;
}


/**
 * checks if input std::string is a positive integer number
 * @param str_val the input std::string
 * @return true if std::string is a positive integer number, false otherwise
 */
bool is_positive_integer(const std::string& str_val) {
    return !str_val.empty()
            && std::find_if(str_val.begin(), str_val.end(),
                    [](char c) {return !std::isdigit(c);}) == str_val.end();
}

int rvs_util_parse(const std::string& buff, bool* pval) {
  if (buff.empty()) {  // method empty
    return 2;  // not found
  }

  if (buff == "true") {
    *pval = true;
    return 0;  // OK - true
  }

  if (buff == "false") {
    *pval = false;
    return 0;  // OK - false
  }

  return 1;  // syntax error
}

void *json_node_create(std::string module_name, std::string action_name,
                     int log_level){
        unsigned int sec;
        unsigned int usec;

        rvs::lp::get_ticks(&sec, &usec);
        void *json_node = rvs::lp::LogRecordCreate(module_name.c_str(),
                            action_name.c_str(), log_level, sec, usec, true);
        return json_node;
}



void *json_list_create(std::string lname, int log_level){

        void *json_node = rvs::lp::JsonNamedListCreate(lname.c_str() ,log_level);
        return json_node;
}


/**
 * summary: Fetches gpu id to index map for valid set of GPUs as per config.
 * Note: mcm_check is needed to output MCM specific messages while we iterate
 * through list, because iet power readings vary on this.
 * @out: map of gpu id to index in gpus_device_index and returns true if found 
 */
bool fetch_gpu_list(int hip_num_gpu_devices, map<int, uint16_t>& gpus_device_index,
    const std::vector<uint16_t>& property_device, const int& property_device_id,
    bool property_device_all, const std::vector<uint16_t>& property_device_index,
    bool property_device_index_all, bool mcm_check) {

  bool amd_gpus_found = false;
  bool mcm_die = false;
  bool amd_mcm_gpu_found = false;

  for (int i = 0; i < hip_num_gpu_devices; i++) {

    // get GPU device properties
    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, i);

    // compute device location_id (needed in order to identify this device
    // in the gpus_id/gpus_device_id list
    unsigned int pDom, pBus, pDev, pFun;
    getBDF(i, pDom, pBus, pDev, pFun);
    unsigned int dev_location_id =
      ((((unsigned int) (pBus)) << 8) | (((unsigned int)(pDev)) << 3)  | ((unsigned int)(pFun)));
    uint16_t dev_domain = pDom;
    uint16_t devId;
    uint16_t gpuIdx;
    uint16_t gpu_id;

    if (rvs::gpulist::domlocation2gpu(dev_domain, dev_location_id, &gpu_id)) {
      continue;
    }
    if (rvs::gpulist::gpu2device(gpu_id, &devId)){
      continue;
    }
    if (rvs::gpulist::gpu2gpuindex(gpu_id, &gpuIdx)){
      continue;
    }

    // filter by device id if needed
    if (property_device_id > 0 && property_device_id != devId)
      continue;

    // check if this GPU is part of the select ones as per config
    // (device = "all" or the gpu_id is in the device: <gpu id> list)
    bool cur_gpu_selected = false;

    if (property_device_all || property_device_index_all) {
      cur_gpu_selected = true;
    } else {
      // search for this gpu in the list
      // provided under the <device> property
      auto it_gpu_id = find(property_device.begin(),
          property_device.end(),
          gpu_id);

      if (it_gpu_id != property_device.end())
        cur_gpu_selected = true;

      // search for this gpu in the list
      // provided under the <device_index> property
      auto it_gpu_idx = find(property_device_index.begin(),
          property_device_index.end(),
          gpuIdx);

      if (it_gpu_idx != property_device_index.end())
        cur_gpu_selected = true;
    }

    // if mcm check enabled, print message if device is MCM
    //  MCM is only for mi200, so check only if gfx90a
    if (mcm_check && (std::string{props.gcnArchName}.find("gfx90a") != std::string::npos) ){
      std::stringstream msg_stream;
      mcm_die =  gpu_check_if_mcm_die(i);
      if (mcm_die) {
        msg_stream.str("");
        msg_stream << "GPU ID : " << std::setw(5) << gpu_id << " - " << "Device : " << std::setw(5) << devId <<
          " - " << "GPU is a die/chiplet in Multi-Chip Module (MCM) GPU";
        rvs::lp::Log(msg_stream.str(), rvs::logresults);

        amd_mcm_gpu_found = true;
      }
    }
    // excluding secondary die from test list, drops power reading substantially
    if (cur_gpu_selected) { // need not exclude secondary die, just log it out.
      gpus_device_index.insert
        (std::pair<int, uint16_t>(i, gpu_id));
      amd_gpus_found = true;
    }
    mcm_die = false;
  }

  if (amd_mcm_gpu_found && mcm_check) {
    std::stringstream msg_stream;
    msg_stream.str("");
    msg_stream << "Note: The system has Multi-Chip Module (MCM) GPU/s." << "\n"
      << "In MCM GPU, primary GPU die shows total socket (primary + secondary) power information." << "\n"
      << "Secondary GPU die does not have any power information associated with it independently."<< "\n"
      << "So, expect power reading from Secondary GPU die as 0."<< "\n";
    rvs::lp::Log(msg_stream.str(), rvs::logresults);
  }
  return amd_gpus_found;
}

void getBDF(int idx ,unsigned int& domain,unsigned int& bus,unsigned int& device,unsigned int& function){
    char pciString[256] = {0};
    auto hipRet = hipDeviceGetPCIBusId(pciString, 256, idx);
    std::string msg;
    if(hipRet != hipSuccess){
      msg = "For GPU:" + std::to_string(idx) + ", failed to get PCI Bus id";
      rvs::lp::Log(msg, rvs::logresults);
      return;
    }
    if (sscanf(pciString, "%04x:%02x:%02x.%01x", reinterpret_cast<unsigned int*>(&domain),
       reinterpret_cast<unsigned int*>(&bus),
       reinterpret_cast<unsigned int*>(&device),
       reinterpret_cast<unsigned int*>(&function)) != 4) {
         msg = std::string("parsing incomplete for BDF id: ") + pciString ;
         rvs::lp::Log(msg, rvs::logresults);
    }
}

int display_gpu_info (void) {

  struct device_info {
    std::string bus;
    std::string name;
    int32_t node_id;
    int32_t gpu_id;
    int32_t device_id;
    uint64_t bdfId;// this is pcie location id to uniquely identify device
  };

  char buf[256];
  int hip_num_gpu_devices;
  std::string errmsg = " No supported GPUs available.";
  std::vector<device_info> gpu_info_list;

  hipGetDeviceCount(&hip_num_gpu_devices);
  if( hip_num_gpu_devices == 0){
    std::cout << std::endl << errmsg << std::endl;
    return 0;
  }
  for (int i = 0; i < hip_num_gpu_devices; i++) {
    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, i);
    unsigned int pDom, pBus, pDev, pFun;
    getBDF(i , pDom, pBus, pDev, pFun);
    // compute device location_id (needed in order to identify this device
    // in the gpus_id/gpus_device_id list
    unsigned int dev_location_id =
        ((((unsigned int) (pBus)) << 8) | (((unsigned int)(pDev)) << 3) | ((unsigned int)(pFun)));
    uint16_t dev_domain = pDom;  
    uint16_t node_id;
    if (rvs::gpulist::domlocation2node(dev_domain, dev_location_id, &node_id)) {
      continue;
    }
    uint16_t gpu_id;
    if (rvs::gpulist::domlocation2gpu(dev_domain, dev_location_id, &gpu_id)) {
      continue;
    }
    uint16_t dev_id;
    if (rvs::gpulist::gpu2device(gpu_id, &dev_id)){
      continue;
    }

    uint64_t completBDFId = 0;
    completBDFId |= (static_cast<uint64_t>(pDom) & 0xFFFF) << 24;
    completBDFId |= (static_cast<uint64_t>(pBus) & 0xFF) << 16;
    completBDFId |= (static_cast<uint64_t>(pDev) & 0x1F) << 3;
    completBDFId |= (static_cast<uint64_t>(pFun) & 0x7);

    hipDeviceGetPCIBusId(buf, 256, i);
    device_info info;
    info.bus       = buf;
    info.name      = props.name;
    info.node_id   = node_id;
    info.gpu_id    = gpu_id;
    info.device_id = dev_id;
    info.bdfId = completBDFId;
    gpu_info_list.push_back(info);
  }

  std::sort(gpu_info_list.begin(), gpu_info_list.end(),
           [](const struct device_info& a, const struct device_info& b) {
             return a.bdfId < b.bdfId; });
  if (!gpu_info_list.empty()) {
    std::cout << "Supported GPUs available:\n";
    for (const auto& info : gpu_info_list) {
      std::cout << info.bus  << " - GPU[" << std::setw(2) << info.node_id
      << " - " << std::setw(5) << info.gpu_id << "] " << info.name
      << " (Device " << info.device_id << ")\n";
    }
  } else {
      std::cout << std::endl << errmsg << std::endl;
  }
  return 0;
}


void json_add_primary_fields(std::string moduleName, std::string action_name){
  if (rvs::lp::JsonActionStartNodeCreate(moduleName.c_str(), action_name.c_str())){
    rvs::lp::Err("json start create failed", moduleName, action_name);
    return;
  }
}

void cleanup_logs(){
  rvs::lp::JsonEndNodeCreate();
}

