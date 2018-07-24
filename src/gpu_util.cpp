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
#include "gpu_util.h"

#include <stdlib.h>
#include <dirent.h>
#include <string.h>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <algorithm>

std::vector<uint16_t> rvs::gpulist::location_id;
std::vector<uint16_t> rvs::gpulist::gpu_id;
std::vector<uint16_t> rvs::gpulist::device_id;
std::vector<uint16_t> rvs::gpulist::node_id;

using std::vector;
using std::string;
using std::ifstream;

int gpu_num_subdirs(const char* dirpath, const char* prefix) {
  int count = 0;
  DIR *dirp;
  struct dirent *dir;
  int prefix_len = strlen(prefix);

  dirp = opendir(dirpath);
  if (dirp) {
    while ((dir = readdir(dirp)) != 0) {
      if ((strcmp(dir->d_name, ".") == 0) ||
        (strcmp(dir->d_name, "..") == 0))
        continue;
      if (prefix_len &&
        strncmp(dir->d_name, prefix, prefix_len))
        continue;
      count++;
    }
    closedir(dirp);
  }
  return count;
}

/**
 * gets all GPUS location_id
 * @param gpus_location_id the vector that will store all the GPU location_id
 * @return
 */
void gpu_get_all_location_id(std::vector<uint16_t>& gpus_location_id) {
  ifstream f_id, f_prop;
  char path[KFD_PATH_MAX_LENGTH];

  std::string prop_name;
  int gpu_id;
  uint32_t prop_val;


  // Discover the number of nodes: Inside nodes folder there are only folders
  // that represent the node number
  int num_nodes = gpu_num_subdirs(KFD_SYS_PATH_NODES, "");

  // get all GPUs device id
  for (int node_id = 0; node_id < num_nodes; node_id++) {
    snprintf(path, KFD_PATH_MAX_LENGTH, "%s/%d/gpu_id", KFD_SYS_PATH_NODES,
           node_id);
    f_id.open(path);
    snprintf(path, KFD_PATH_MAX_LENGTH, "%s/%d/properties",
             KFD_SYS_PATH_NODES, node_id);
    f_prop.open(path);

    f_id >> gpu_id;

    if (gpu_id != 0) {
      while (f_prop >> prop_name) {
        if (prop_name == "location_id") {
          f_prop >> prop_val;
          gpus_location_id.push_back(prop_val);
          break;
        }
      }
    }

    f_id.close();
    f_prop.close();
  }
}

/**
 * gets all GPUS gpu_id
 * @param gpus_id the vector that will store all the GPU gpu_id
 * @return
 */
void gpu_get_all_gpu_id(std::vector<uint16_t>& gpus_id) {
  ifstream f_id, f_prop;
  char path[KFD_PATH_MAX_LENGTH];

  int gpu_id;

  // Discover the number of nodes: Inside nodes folder there are only folders
  // that represent the node number
  int num_nodes = gpu_num_subdirs(KFD_SYS_PATH_NODES, "");

  // get all GPUs device id
  for (int node_id = 0; node_id < num_nodes; node_id++) {
    snprintf(path, KFD_PATH_MAX_LENGTH, "%s/%d/gpu_id", KFD_SYS_PATH_NODES,
             node_id);
    f_id.open(path);

    f_id >> gpu_id;

    if (gpu_id != 0)
      gpus_id.push_back(gpu_id);

    f_id.close();
  }
}

/**
 * gets all GPUS device_id
 * @param gpus_device_id the vector that will store all the GPU location_id
 * @return
 */
void gpu_get_all_device_id(std::vector<uint16_t>& gpus_device_id) {
  ifstream f_id, f_prop;
  char path[KFD_PATH_MAX_LENGTH];

  std::string prop_name;
  int gpu_id;
  uint32_t prop_val;

  // Discover the number of nodes: Inside nodes folder there are only folders
  // that represent the node number
  int num_nodes = gpu_num_subdirs(KFD_SYS_PATH_NODES, "");

  // get all GPUs device id
  for (int node_id = 0; node_id < num_nodes; node_id++) {
    snprintf(path, KFD_PATH_MAX_LENGTH, "%s/%d/gpu_id", KFD_SYS_PATH_NODES,
             node_id);
    f_id.open(path);
    snprintf(path, KFD_PATH_MAX_LENGTH, "%s/%d/properties",
             KFD_SYS_PATH_NODES, node_id);
    f_prop.open(path);

    f_id >> gpu_id;

    if (gpu_id != 0) {
      while (f_prop >> prop_name) {
        if (prop_name == "device_id") {
          f_prop >> prop_val;
          gpus_device_id.push_back(prop_val);
          break;
        }
      }
    }

    f_id.close();
    f_prop.close();
  }
}

/**
 * gets all GPUS nodes
 * @param gpus_node_id the vector that will store all the GPU nodes
 * @return
 */
void gpu_get_all_node_id(std::vector<uint16_t>& gpus_node_id) {
  ifstream f_id;
  char path[KFD_PATH_MAX_LENGTH];
  int gpu_id;


  // Discover the number of nodes: Inside nodes folder there are only folders
  // that represent the node number
  int num_nodes = gpu_num_subdirs(KFD_SYS_PATH_NODES, "");

  // get all GPUs device id
  for (int node_id = 0; node_id < num_nodes; node_id++) {
    snprintf(path, KFD_PATH_MAX_LENGTH, "%s/%d/gpu_id", KFD_SYS_PATH_NODES,
           node_id);
    f_id.open(path);
    f_id >> gpu_id;

    if (gpu_id != 0) {
      gpus_node_id.push_back(node_id);
    }
    f_id.close();
  }
}

/**
 * @brief Initialize gpulist helper class
 * @return 0 if successful, -1 otherwise
 *}
 * */
int rvs::gpulist::Initialize() {
  gpu_get_all_location_id(location_id);
  gpu_get_all_gpu_id(gpu_id);
  gpu_get_all_device_id(device_id);
  gpu_get_all_node_id(node_id);
  return 0;
}



/**
 * @brief Given Gpu ID return Location ID
 * @param GpuID Gpu ID
 * @return Location ID if found, -1 otherwise
 *}
 * */
int32_t rvs::gpulist::GetLocation(const uint32_t GpuID) {
  const auto it = std::find(gpu_id.cbegin(),
                            gpu_id.cend(), GpuID);
  if (it != gpu_id.cend()) {
    size_t pos = std::distance(gpu_id.cbegin(), it);
    return location_id[pos];
  }
  return -1;
}


/**
 * @brief Given Location ID return GPU ID
 * @param LocationID Location ID of a GPU
 * @return Gpu ID if found, -1 otherwise
 *}
 * */
int32_t rvs::gpulist::GetGpuId(const uint32_t LocationID) {
  const auto it = std::find(location_id.cbegin(),
                            location_id.cend(), LocationID);
  if (it != location_id.cend()) {
    size_t pos = std::distance(location_id.cbegin(), it);
    return gpu_id[pos];
  }
  return -1;
}

/**
 * @brief Given Location ID return GPU device ID
 * @param LocationID Location ID of a GPU
 * @return Device ID if found, -1 otherwise
 *}
 * */
int32_t rvs::gpulist::GetDeviceIdFromLocationId(const uint32_t LocationID) {
  const auto it = std::find(location_id.cbegin(),
                            location_id.cend(), LocationID);
  if (it != location_id.cend()) {
    size_t pos = std::distance(location_id.cbegin(), it);
    return device_id[pos];
  }
  return -1;
}

/**
 * @brief Given Gpu ID return GPU device ID
 * @param GpuID Gpu ID of a GPU
 * @return Device ID if found, -1 otherwise
 *}
 * */
int32_t rvs::gpulist::GetDeviceIdFromGpuId(const uint32_t GpuID) {
  const auto it = std::find(gpu_id.cbegin(),
                            gpu_id.cend(), GpuID);
  if (it != gpu_id.cend()) {
    size_t pos = std::distance(gpu_id.cbegin(), it);
    return device_id[pos];
  }

  return -1;
}

/**
 * @brief Given Gpu ID return GPU node ID
 * @param GpuID Gpu ID of a GPU
 * @return Node ID if found, -1 otherwise
 *}
 * */
int32_t rvs::gpulist::GetNodeIdFromGpuId(const uint32_t GpuID) {
  const auto it = std::find(gpu_id.cbegin(),
                            gpu_id.cend(), GpuID);
  if (it != gpu_id.cend()) {
    size_t pos = std::distance(gpu_id.cbegin(), it);
    return node_id[pos];
  }

  return -1;
}

/**
 * @brief Given Location ID return GPU device ID
 * @param LocationID Location ID of a GPU
 * @return Device ID if found, -1 otherwise
 *}
 * */
int32_t rvs::gpulist::GetNodeIdFromLocationId(const uint32_t LocationID) {
  const auto it = std::find(location_id.cbegin(),
                            location_id.cend(), LocationID);
  if (it != location_id.cend()) {
    size_t pos = std::distance(location_id.cbegin(), it);
    return node_id[pos];
  }
  return -1;
}
