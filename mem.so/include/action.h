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
#ifndef MEM_SO_INCLUDE_ACTION_H_
#define MEM_SO_INCLUDE_ACTION_H_

#ifdef __cplusplus
extern "C" {
#endif
#include <pci/pci.h>
#ifdef __cplusplus
}
#endif

#include <vector>
#include <string>
#include <mutex>
#include <map>

#include "include/rvsactionbase.h"

using std::vector;
using std::string;
using std::map;

#define MODULE_NAME                     "mem"
#define MODULE_NAME_CAPS                "MEM"

#if 1
#define RVS_CONF_MAPPED_MEM             "mapped_memory"
#define RVS_CONF_MEM_PATTERN            "mem_pattern"
#define RVS_CONF_MEM_STRESS             "stress"
#define RVS_CONF_NUM_BLOCKS             "mem_blocks"
#define RVS_CONF_NUM_ITER               "num_iter"
#define RVS_CONF_PATTERN                "pattern"
#define RVS_CONF_NUM_PASSES             "num_passes"
#define RVS_CONF_THRDS_PER_BLK          "thrds_per_blk"


#define MEM_DEFAULT_NUM_BLOCKS          256
#define MEM_DEFAULT_THRDS_BLK           128
#define MEM_DEFAULT_NUM_ITERATIONS      1
#define MEM_DEFAULT_NUM_PASSES          1
#define MEM_DEFAULT_CUDA_MEMTEST        1
#define MEM_DEFAULT_MAPPED_MEM          false 
#define MEM_DEFAULT_STRESS              false


#define MEM_NO_COMPATIBLE_GPUS          "No AMD compatible GPU found!"
#define FLOATING_POINT_REGEX            "^[0-9]*\\.?[0-9]+$"
#define JSON_CREATE_NODE_ERROR          "JSON cannot create node"
#endif



/**
 * @class mem_action
 * @ingroup MEM
 *
 * @brief MEM action implementation class
 *
 * Derives from rvs::actionbase and implements actual action functionality
 * in its run() method.
 *
 */
class mem_action: public rvs::actionbase {
 public:
    mem_action();

    virtual ~mem_action();

    virtual int run(void);

    std::string mem_ops_type;

 protected:
    //! TRUE if JSON output is required
    bool bjson;
    //! Memorry mapped
    bool mem_mapped;
    //! maximum number of blocks
    uint64_t max_num_blocks;
    //! pattern
    uint64_t pattern;
    //! Num of iterations
    uint64_t num_iterations;
    //! Num of passes
    uint64_t num_passes;
    //! stress
    bool stress;
    // Mapped memory
    bool useMappedMemory;
    // memory blocks
    uint64_t numofMemblocks;
    //threads per block
    uint64_t threadsPerBlock;
    
    // exclude tests list
    vector<uint32_t> exclude_list;
    // configuration properties getters
    bool get_all_mem_config_keys(void);
  /**
  * @brief reads all common configuration keys from
  * the module's properties collection
  * @return true if no fatal error occured, false otherwise
  */
    bool get_all_common_config_keys(void);

  /**
  * @brief gets the number of ROCm compatible AMD GPUs
  * @return run number of GPUs
  */
  int get_num_amd_gpu_devices(void);
  int get_all_selected_gpus(void);
  int set_mem_mapped(void);

  bool do_mem_stress_test(map<int, uint16_t> mem_gpus_device_index);
};

#endif  // MEM_SO_INCLUDE_ACTION_H_
