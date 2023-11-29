/********************************************************************************
 *
 * Copyright (c) 2018-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef INCLUDE_GPU_UTIL_H_
#define INCLUDE_GPU_UTIL_H_

#include <stdint.h>
#include <vector>
#include <string>
#include <map>

#define KFD_SYS_PATH_NODES              "/sys/class/kfd/kfd/topology/nodes"
#define KFD_PATH_MAX_LENGTH             256

extern int  gpu_num_subdirs(const char* dirpath, const char* prefix);
extern void gpu_get_all_location_id(std::vector<uint16_t>* pgpus_location_id);
extern void gpu_get_all_gpu_id(std::vector<uint16_t>* pgpus_id);
extern void gpu_get_all_device_id(std::vector<uint16_t>* pgpus_device_id);
extern void gpu_get_all_node_id(std::vector<uint16_t>* pgpus_node_id);
extern void gpu_get_all_domain_id(std::vector<uint16_t>* pgpus_domain_id,
                std::map<std::pair<uint16_t, uint16_t> , uint16_t>& pgpus_dom_loc_map); 
extern bool gpu_check_if_mcm_die (int idx);
extern int gpu_hip_to_smi_index(int hip_index, uint32_t* smi_index);

namespace rvs {

  ::std::string bdf2string(uint32_t BDF);

/**
 * @class gpulist
 *
 * @brief GPU cross-indexing utility class
 *
 * Used to quickly get GPU ID from location ID and vs. versa
 *
 */
class gpulist {
 public:
  static int Initialize();

  static int location2gpu(const uint16_t LocationID, uint16_t* pGpuID);
  static int gpu2location(const uint16_t GpuID, uint16_t* pLocationID);
  static int node2gpu(const uint16_t NodeID, uint16_t* pGpuID);
  static int location2device(const uint16_t LocationID, uint16_t* pDeviceID);
  static int gpu2device(const uint16_t GpuID, uint16_t* pDeviceID);
  static int location2node(const uint16_t LocationID, uint16_t* pNodeID);
  static int gpu2node(const uint16_t GpuID, uint16_t* pNodeID);
  static int gpu2domain(const uint16_t GpuID, uint16_t* pDomain);
  static int domlocation2node(const uint16_t domainID, const uint16_t LocationID,
                                    uint16_t* pNodeID);
  static int domlocation2gpu(const uint16_t domainID, const uint16_t LocationID,
                                    uint16_t* pGPUID);
 protected:
  //! Array of GPU location IDs
  static std::vector<uint16_t> location_id;
  //! Array of GPU IDs
  static std::vector<uint16_t> gpu_id;
  //! Array of device IDs
  static std::vector<uint16_t> device_id;
  //! Array of node IDs
  static std::vector<uint16_t> node_id;
  //! Array of domain IDs
  static std::vector<uint16_t> domain_id;
  static std::map<std::pair<uint16_t, uint16_t>	, uint16_t> domain_loc_map;
};


}  // namespace rvs

#endif  // INCLUDE_GPU_UTIL_H_
