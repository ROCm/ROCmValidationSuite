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

#define MODULE_NAME                     "babel"
#define MODULE_NAME_CAPS                "BABEL"

#define RVS_CONF_ARRAY_SIZE             "array_size"
#define RVS_CONF_NUM_ITER               "num_iter"
#define RVS_CONF_TEST_TYPE              "test_type"
#define RVS_CONF_MEM_MIBIBYTE           "mibibytes"
#define RVS_CONF_OP_CSV                 "o/p_csv"
#define RVS_CONF_DWORDS_PER_LANE        "dwords_per_lane"
#define RVS_CONF_CHUNKS_PER_BLOCK       "chunks_per_block"
#define RVS_CONF_TB_SIZE                "tb_size"
#define RVS_CONF_READ                   "read"
#define RVS_CONF_WRITE                  "write"
#define RVS_CONF_COPY                   "copy"
#define RVS_CONF_ADD                    "add"
#define RVS_CONF_MUL                    "mul"
#define RVS_CONF_DOT                    "dot"
#define RVS_CONF_TRIAD                  "triad"

#define MEM_DEFAULT_ARRAY_SIZE          33554432   // 32 MB
#define MEM_DEFAULT_NUM_ITER            100
#define MEM_DEFAULT_TEST_TYPE           1
#define MEM_DEFAULT_MEM_MIBIBYTE        false
#define MEM_DEFAULT_OP_CSV              false
#define MEM_DEFAULT_DWORDS_PER_LANE     4
#define MEM_DEFAULT_CHUNKS_PER_BLOCK    2
#define MEM_DEFAULT_TB_SIZE             1024
#define MEM_DEFAULT_TEST_ENABLE         false

#define MEM_NO_COMPATIBLE_GPUS          "No AMD compatible GPU found!"
#define FLOATING_POINT_REGEX            "^[0-9]*\\.?[0-9]+$"
#define JSON_CREATE_NODE_ERROR          "JSON cannot create node"

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
    //! Memory in bytes
    bool mibibytes;
    //! output in csv 
    bool output_csv;
    //! read test enable/disable
    bool read;
    //! write test enable/disable
    bool write;
    //! copy test enable/disable
    bool copy;
    //! add test enable/disable
    bool add;
    //! mul test enable/disable
    bool mul;
    //! dot test enable/disable
    bool dot;
    //! triad test enable/disable
    bool triad;
    //! test type
    int  test_type;
    //! number of iterations
    uint64_t num_iterations;
    //! number of iterations
    uint64_t array_size;
    //! number of dwords per lane
    uint16_t dwords_per_lane;
    //! number of chunks per block
    uint16_t chunks_per_block;
    //! thread block size
    uint16_t tb_size;

    // configuration properties getters
    bool get_all_mem_config_keys(void);

  /**
  * @brief gets the number of ROCm compatible AMD GPUs
  * @return run number of GPUs
  */
  int get_num_amd_gpu_devices(void);
  int get_all_selected_gpus(void);
  bool do_mem_stress_test(map<int, uint16_t> mem_gpus_device_index);
};

#endif  // MEM_SO_INCLUDE_ACTION_H_
