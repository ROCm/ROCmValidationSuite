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
#include "include/gpu_util.h"

#include <stdlib.h>
#include <dirent.h>
#include <string.h>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>

std::vector<uint16_t> rvs::gpulist::location_id;
std::vector<uint16_t> rvs::gpulist::gpu_id;
std::vector<uint16_t> rvs::gpulist::device_id;
std::vector<uint16_t> rvs::gpulist::node_id;
std::vector<uint16_t> rvs::gpulist::domain_id;
std::map<std::pair<uint16_t, uint16_t> , uint16_t> rvs::gpulist::domain_loc_map;
using std::vector;
using std::string;
using std::ifstream;

/* No of GPU devices with MCM GPU */
#define MAX_NUM_MCM_GPU 4

/* Unique Device Ids of MCM GPUS */
static const uint16_t mcm_gpu_device_id[MAX_NUM_MCM_GPU] = {
	/* Aldebaran */
	0x7408,
	0x740C,
	0x740F,
	0x7410};

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
 * @param pgpus_location_id ptr to vector that will store all the GPU location_id
 * @return
 */
void gpu_get_all_location_id(std::vector<uint16_t>* pgpus_location_id) {
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
          (*pgpus_location_id).push_back(prop_val);
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
 * @param pgpus_id ptr to vector that will store all the GPU gpu_id
 * @return
 */
void gpu_get_all_gpu_id(std::vector<uint16_t>* pgpus_id) {
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
      (*pgpus_id).push_back(gpu_id);

    f_id.close();
  }
}

/**
 * gets all GPUS device_id
 * @param pgpus_device_id ptr to vector that will store all the GPU location_id
 * @return
 */
void gpu_get_all_device_id(std::vector<uint16_t>* pgpus_device_id) {
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
          (*pgpus_device_id).push_back(prop_val);
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
 * @param pgpus_node_id ptr to vector that will store all the GPU nodes
 * @return
 */
void gpu_get_all_node_id(std::vector<uint16_t>* pgpus_node_id) {
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
      (*pgpus_node_id).push_back(node_id);
    }
    f_id.close();
  }
}

/**
 * gets all GPUS' domain id which along BDF forms whole address
 * @param pgpus_domain_id ptr to vector that will store all the GPU domain_id
 * @return
 */
void gpu_get_all_domain_id(std::vector<uint16_t>* pgpus_domain_id, 
		std::map<std::pair<uint16_t, uint16_t> , uint16_t>& pgpus_dom_loc_map) {
  ifstream f_id, f_prop;
  char path[KFD_PATH_MAX_LENGTH];

  std::string prop_name;
  int gpu_id;
  uint32_t domain_val;
  uint32_t loc_val;

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
        if (prop_name == "domain") {
          f_prop >> domain_val;
          (*pgpus_domain_id).push_back(domain_val);
          continue;
        }
	else if(prop_name == "location_id"){
            f_prop >> loc_val;
            continue;
	}
      }
      pgpus_dom_loc_map[std::make_pair(domain_val, loc_val)] = gpu_id;
    }

    f_id.close();
    f_prop.close();
  }
}

/**
 * @brief Check if the GPU is die (chiplet) in Multi-Chip Module (MCM) GPU.
 * @param device_id GPU Device ID
 * @return true if GPU is die in MCM GPU, false if GPU is single die GPU.
 **/
bool gpu_check_if_mcm_die (uint16_t device_id) {

  uint16_t i = 0;
  bool mcm_die = false;
  
  for (i  = 0; i < MAX_NUM_MCM_GPU; i++) {
    if(mcm_gpu_device_id[i] == device_id) {
      mcm_die = true;
      break;
    }
  }
  return mcm_die;
}

/**
 * @brief Initialize gpulist helper class
 * @return 0 if successful, -1 otherwise
 **/
int rvs::gpulist::Initialize() {
  gpu_get_all_location_id(&location_id);
  gpu_get_all_gpu_id(&gpu_id);
  gpu_get_all_device_id(&device_id);
  gpu_get_all_node_id(&node_id);
  gpu_get_all_domain_id(&domain_id, domain_loc_map);

  return 0;
}


/**
 * @brief Given Gpu ID return Location ID
 * @param GpuID Gpu ID
 * @param pLocationID Location ID of the GPU
 * @return 0 if found, -1 otherwise
 **/
int rvs::gpulist::gpu2location(const uint16_t GpuID,
                               uint16_t* pLocationID) {
  const auto it = std::find(gpu_id.cbegin(),
                            gpu_id.cend(), GpuID);
  if (it == gpu_id.cend()) {
    return -1;
  }
  size_t pos = std::distance(gpu_id.cbegin(), it);
  *pLocationID = location_id[pos];
  return 0;
}


/**
 * @brief Given Location ID return GPU ID
 * @param LocationID Location ID of a GPU
 * @param pGpuID GPU ID of the GPU on Location ID
 * @return 0 if found, -1 otherwise
 **/
int rvs::gpulist::location2gpu(const uint16_t LocationID, uint16_t* pGpuID) {
  const auto it = std::find(location_id.cbegin(),
                            location_id.cend(), LocationID);
  if (it == location_id.cend()) {
    return -1;
  }
  size_t pos = std::distance(location_id.cbegin(), it);
  *pGpuID = gpu_id[pos];
  return 0;
}


/**
 * @brief Given Node ID return GPU ID
 * @param NodeID Location ID of a GPU
 * @param pGpuID device ID of the GPU
 * @return 0 if found, -1 otherwise
 **/
int rvs::gpulist::node2gpu(const uint16_t NodeID, uint16_t* pGpuID) {
  const auto it = std::find(node_id.cbegin(),
                            node_id.cend(), NodeID);
  if (it == node_id.cend()) {
    return -1;
  }
  size_t pos = std::distance(node_id.cbegin(), it);
  *pGpuID = gpu_id[pos];
  return 0;
}


/**
 * @brief Given Location ID return GPU device ID
 * @param LocationID Location ID of a GPU
 * @param pDeviceID device ID of the GPU
 * @return 0 if found, -1 otherwise
 **/
int rvs::gpulist::location2device(const uint16_t LocationID,
                                  uint16_t* pDeviceID) {
  const auto it = std::find(location_id.cbegin(),
                            location_id.cend(), LocationID);
  if (it == location_id.cend()) {
    return -1;
  }
  size_t pos = std::distance(location_id.cbegin(), it);
  *pDeviceID = device_id[pos];
  return 0;
}


/**
 * @brief Given Gpu ID return GPU device ID
 * @param GpuID Gpu ID of a GPU
 * @param pDeviceID device ID of the GPU
 * @return 0 if found, -1 otherwise
 **/
int rvs::gpulist::gpu2device(const uint16_t GpuID, uint16_t* pDeviceID) {
  const auto it = std::find(gpu_id.cbegin(),
                            gpu_id.cend(), GpuID);
  if (it == gpu_id.cend()) {
    return -1;
  }

  size_t pos = std::distance(gpu_id.cbegin(), it);
  *pDeviceID = device_id[pos];
  return 0;
}


/**
 * @brief Given Gpu ID return GPU HSA Node ID
 * @param GpuID Gpu ID of a GPU
 * @param pNodeID Node ID of the GPU
 * @return 0 if found, -1 otherwise
 **/
int rvs::gpulist::gpu2node(const uint16_t GpuID, uint16_t* pNodeID) {
  const auto it = std::find(gpu_id.cbegin(),
                            gpu_id.cend(), GpuID);
  if (it == gpu_id.cend()) {
    return -1;
  }

  size_t pos = std::distance(gpu_id.cbegin(), it);
  *pNodeID = node_id[pos];
  return 0;
}


/**
 * @brief Given Location ID return GPU node ID
 * @param LocationID Location ID of a GPU
 * @param pNodeID Node ID of the GPU
 * @return 0 if found, -1 otherwise
 **/
int rvs::gpulist::location2node(const uint16_t LocationID,
                                    uint16_t* pNodeID) {
  const auto it = std::find(location_id.cbegin(),
                            location_id.cend(), LocationID);
  if (it == location_id.cend()) {
    return -1;
  }

  size_t pos = std::distance(location_id.cbegin(), it);
  *pNodeID = node_id[pos];
  return 0;
}

/**
 * @brief Given domain id and Location ID return GPU node ID
 * @param LocationID Location ID of a GPU
 * @param pNodeID Node ID of the GPU
 * @return 0 if found, -1 otherwise
 **/
int rvs::gpulist::domlocation2node(const uint16_t domainID, const uint16_t LocationID,
                                    uint16_t* pNodeID) {
  auto it = domain_loc_map.find(std::make_pair(domainID, LocationID));
  if (it == domain_loc_map.end()) {
    return -1;
  }

  return gpu2node(it->second, pNodeID);
}

/**
 * @brief Given domain id and Location ID return GPU node ID
 * @param LocationID Location ID of a GPU
 * @param pNodeID Node ID of the GPU
 * @return 0 if found, -1 otherwise
 **/
int rvs::gpulist::domlocation2gpu(const uint16_t domainID, const uint16_t LocationID,
                                    uint16_t* pGPUID) {
  auto it = domain_loc_map.find(std::make_pair(domainID, LocationID));
  if (it == domain_loc_map.end()) {
    return -1;
  }
  *pGPUID = it->second;
  return 0;
}


/**
 * @brief Given Gpu ID return GPU domain ID
 * @param GpuID Gpu ID of a GPU
 * @param pDomain domain ID of the GPU
 * @return 0 if found, -1 otherwise
 **/
int rvs::gpulist::gpu2domain(const uint16_t GpuID, uint16_t* pDomain) {
  const auto it = std::find(gpu_id.cbegin(),
                            gpu_id.cend(), GpuID);
  if (it == gpu_id.cend()) {
    return -1;
  }
  size_t pos = std::distance(gpu_id.cbegin(), it);
  std::cout << "For GPU " << GpuID << " domain is " << domain_id[pos] << std::endl;
  *pDomain = domain_id[pos];
  return 0;
}

std::string rvs::bdf2string(uint32_t BDF) {
  char buff[32];
  snprintf(buff, sizeof(buff), "%02X:%02X.%d",
           BDF>>8, (BDF & 0xFF), 0);
  return buff;
}
