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
#ifndef INCLUDE_GPU_UTIL_H_
#define INCLUDE_GPU_UTIL_H_

#include <stdint.h>
#include <vector>

#define KFD_SYS_PATH_NODES              "/sys/class/kfd/kfd/topology/nodes"
#define KFD_PATH_MAX_LENGTH             256

extern int  gpu_num_subdirs(const char* dirpath, const char* prefix);
extern void gpu_get_all_location_id(std::vector<uint16_t>& gpus_location_id);
extern void gpu_get_all_gpu_id(std::vector<uint16_t>& gpus_id);
extern void gpu_get_all_device_id(std::vector<uint16_t>& gpus_device_id);
extern void gpu_get_all_node_id(std::vector<uint16_t>& gpus_node_id);

namespace rvs {

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
  static int32_t GetLocation(const uint32_t GpuID);
  static int32_t GetGpuId(const uint32_t LocationID);
  static int32_t GetGpuIdFromNodeId(const uint32_t NodeID);
  static int32_t GetDeviceIdFromLocationId(const uint32_t LocationID);
  static int32_t GetDeviceIdFromGpuId(const uint32_t GpuID);
  static int32_t GetNodeIdFromGpuId(const uint32_t GpuID);
  static int32_t GetNodeIdFromLocationId(const uint32_t LocationID);

 protected:
  //! Array of GPU location IDs
  static std::vector<uint16_t> location_id;
  //! Array of GPU IDs
  static std::vector<uint16_t> gpu_id;
  //! Array of device IDs
  static std::vector<uint16_t> device_id;
  //! Array of node IDs
  static std::vector<uint16_t> node_id;
};

}  // namespace rvs

#endif  // INCLUDE_GPU_UTIL_H_
