/********************************************************************************
 *
 * Copyright (c) 2018-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <unordered_map>
#include <utility>
#include "amd_smi/amdsmi.h"

#define KFD_SYS_PATH_NODES              "/sys/class/kfd/kfd/topology/nodes"
#define KFD_PATH_MAX_LENGTH             256

extern int  gpu_num_subdirs(const char* dirpath, const char* prefix);
extern void gpu_get_all_location_id(std::vector<uint16_t>* pgpus_location_id);
extern void gpu_get_all_gpu_id(std::vector<uint16_t>* pgpus_id);
extern void gpu_get_all_gpu_idx(std::vector<uint16_t>* pgpus_idx);
extern void gpu_get_all_device_id(std::vector<uint16_t>* pgpus_device_id);
extern void gpu_get_all_node_id(std::vector<uint16_t>* pgpus_node_id);
extern void gpu_get_all_domain_id(std::vector<uint16_t>* pgpus_domain_id,
                std::map<std::pair<uint16_t, uint16_t> , uint16_t>& pgpus_dom_loc_map); 
extern bool gpu_check_if_mcm_die (int idx);
extern int gpu_hip_to_smi_hdl(int hip_index, amdsmi_processor_handle* smi_index);
extern void gpu_get_all_pci_bdf(std::vector<std::string>& ppci_bdf);
extern bool gpu_check_if_gpu_indexes (const std::vector <uint16_t> &idx);
extern std::string gpu_get_platform_name (void);

namespace rvs {

/**
 * @struct GpuInfo
 * @brief Unified structure containing all GPU identifiers and properties
 * 
 * This structure to replace the parallel arrays with a single cohesive data structure
 * that keeps all GPU-related information together, improving maintainability.
 * Older arrays if not updated in lockstep results in wrong behaviour
 */
struct GpuInfo {
    uint16_t location_id;   //GPU Location ID(location_id) from KFD topology
    uint16_t gpu_id;        //GPU ID from KFD
    uint16_t gpu_idx;       // GPU index in system
    uint16_t device_id;     // PCI Device ID
    uint16_t node_id;       // HSA Node ID
    uint16_t domain_id;     // PCI Domain ID
    std::string pci_bdf;    // PCI Bus:Device.Function string
    
    GpuInfo() 
        : location_id(0), gpu_id(0), gpu_idx(0), device_id(0),
          node_id(0), domain_id(0), pci_bdf("") {}
    
    GpuInfo(uint16_t loc_id, uint16_t g_id, uint16_t idx, 
            uint16_t dev_id, uint16_t n_id, uint16_t dom_id, 
            const std::string& bdf)
        : location_id(loc_id), gpu_id(g_id), gpu_idx(idx),
          device_id(dev_id), node_id(n_id), domain_id(dom_id),
          pci_bdf(bdf) {}

    bool is_valid() const {
        return gpu_id != 0;
    }
    
    std::pair<uint16_t, uint16_t> get_domain_location_pair() const {
        return std::make_pair(domain_id, location_id);
    }
};
/**
 * Error codes for GPU lookup operations
 */
enum class GpuLookupError {
    SUCCESS = 0,              // Operation successful
    NOT_FOUND = -1,           // GPU identifier not found
    NULL_POINTER = -2,        // Null pointer passed as argument
    UNINITIALIZED = -3,       // GPU list not initialized
    INDEX_OUT_OF_BOUNDS = -4, // Internal index corruption
    INVALID_ARGUMENT = -5     // Invalid argument value
};

/**
 * Convert error code to human-readable string
 */
inline const char* error_to_string(GpuLookupError err) {
    switch (err) {
        case GpuLookupError::SUCCESS: 
            return "Success";
        case GpuLookupError::NOT_FOUND: 
            return "GPU identifier not found in system";
        case GpuLookupError::NULL_POINTER: 
            return "Null pointer passed as argument";
        case GpuLookupError::UNINITIALIZED: 
            return "GPU list not initialized - call Initialize() first";
        case GpuLookupError::INDEX_OUT_OF_BOUNDS: 
            return "Internal index corruption detected";
        case GpuLookupError::INVALID_ARGUMENT: 
            return "Invalid argument value";
        default: 
            return "Unknown error";
    }
}


/**
 * @class gpulist
 *
 * @brief GPU cross-indexing utility class
 *
 * Used to quickly get GPU ID from location ID and vs. versa
 *
 * This class maintains a unified list of GPU information and provides
 * fast lookups 
 */
class gpulist {
 public:
  static int Initialize();
  static int Shutdown();
  static int location2gpu(const uint16_t LocationID, uint16_t* pGpuID);
  static int gpu2location(const uint16_t GpuID, uint16_t* pLocationID);
  static int node2gpu(const uint16_t NodeID, uint16_t* pGpuID);
  static int location2device(const uint16_t LocationID, uint16_t* pDeviceID);
  static int gpu2device(const uint16_t GpuID, uint16_t* pDeviceID);
  static int gpu2gpuindex(const uint16_t GpuID, uint16_t* pGpuIdx);
  static int location2node(const uint16_t LocationID, uint16_t* pNodeID);
  static int gpu2node(const uint16_t GpuID, uint16_t* pNodeID);
  static int gpu2domain(const uint16_t GpuID, uint16_t* pDomain);
  static int domlocation2node(const uint16_t domainID, const uint16_t LocationID,
                                    uint16_t* pNodeID);
  static int domlocation2gpu(const uint16_t domainID, const uint16_t LocationID,
                                    uint16_t* pGPUID);
  static int node2bdf(const uint16_t NodeID, std::string& pPciBDF);
  static std::string gpu_get_platform_name (void);
  static const GpuInfo* get_gpu_info_by_gpu_id(uint16_t gpu_id);
    
  static const GpuInfo* get_gpu_info_by_location(uint16_t location_id);
    
  static const GpuInfo* get_gpu_info_by_node(uint16_t node_id);
    
  //Get all GPU information
  static const std::vector<GpuInfo>& get_all_gpu_info();
  static bool is_valid_gpu_id(uint16_t gpu_id);
    
  static size_t get_gpu_count();
    
  static void clear();
    
 protected:
  /// Master list of all GPU information
  static std::vector<GpuInfo> gpu_info_list;
    
  static std::unordered_map<uint16_t, size_t> gpu_id_to_index;//get index in master list
  static std::unordered_map<uint16_t, size_t> location_id_to_index;
  static std::unordered_map<uint16_t, size_t> node_id_to_index;
    
  static std::unordered_map<uint16_t, size_t> device_id_to_index;
 
  struct PairHash {
      template <class T1, class T2>
      std::size_t operator()(const std::pair<T1, T2>& p) const {
          auto h1 = std::hash<T1>{}(p.first);
          auto h2 = std::hash<T2>{}(p.second);
          return h1 ^ (h2 << 1);
      }
  };
    
  static std::unordered_map<std::pair<uint16_t, uint16_t>, size_t, PairHash> 
        domain_location_to_index;
    
  //! Array of GPU location IDs
  static std::vector<uint16_t> location_id;
  //! Array of GPU IDs
  static std::vector<uint16_t> gpu_id;
  //! Array of GPU Indexes
  static std::vector<uint16_t> gpu_idx;
  //! Array of device IDs
  static std::vector<uint16_t> device_id;
  //! Array of node IDs
  static std::vector<uint16_t> node_id;
  //! Array of domain IDs
  static std::vector<uint16_t> domain_id;
  //! Array of PCI BDFs
  static std::vector<std::string> pci_bdf;
  static std::map<std::pair<uint16_t, uint16_t>	, uint16_t> domain_loc_map;

private:
  // checkers for sanity
  static bool validate_output_pointer(const void* ptr, const char* func_name);
  
  static bool check_initialized(const char* func_name);
    
  static bool validate_index(size_t index, const char* func_name);
};

}  // namespace rvs

#endif  // INCLUDE_GPU_UTIL_H_
