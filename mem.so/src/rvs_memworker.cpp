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

#include <unistd.h>
#include <string>
#include <memory>
#include <iostream>

#include "hip/hip_runtime.h"

#include "include/rvs_memworker.h"
#include "include/rvs_memtest.h"
#include "include/rvsloglp.h"


#define MODULE_NAME                             "mem"

#define MEM_MEM_ALLOC_ERROR                     "memory allocation error!"
#define MEM_BLAS_ERROR                          "memory/blas error!"
#define MEM_BLAS_MEMCPY_ERROR                   "HostToDevice mem copy error!"

#define MEM_NUM_SAVE_BLOCKS                     16

#define MEM_START_MSG                           "start"
#define MEM_PASS_KEY                            "pass"
#define MEM_RAMP_EXCEEDED_MSG                   "ramp time exceeded"
#define MEM_TARGET_ACHIEVED_MSG                 "target achieved"
#define MEM_STRESS_VIOLATION_MSG                "stress violation"

using std::string;

bool MemWorker::bjson = false;

MemWorker::MemWorker() {}
MemWorker::~MemWorker() {}



/**
 * @brief performs the stress test on the given GPU
 */
void MemWorker::run() {
    unsigned int    tot_num_blocks;
    unsigned long   totmem;
    hipDeviceProp_t props;
    char*           ptr = NULL;
    string          err_description;
    string          msg;
    size_t          free;
    size_t          total;
    int             error;
    int             deviceId;
   
    //Initializations
    error = 0;

    // log MEM stress test - start message
    msg = "[" + action_name + "] " + MODULE_NAME + " " +
            std::to_string(gpu_id) + " "  +
            " Starting the Memory stress test "; 
    rvs::lp::Log(msg, rvs::loginfo);

    deviceId  = get_gpu_device_index();

    HIP_CHECK(hipGetDeviceProperties(&props, deviceId));

    totmem = props.totalGlobalMem;

    msg = "[" + action_name + "] " + MODULE_NAME + " " +
            std::to_string(gpu_id) + " " + "Toal Global Memory" + " " +
            std::to_string(totmem); 
    rvs::lp::Log(msg, rvs::logtrace);

    //need to leave a little headroom or later calls will fail
    tot_num_blocks = totmem/BLOCKSIZE - MEM_NUM_SAVE_BLOCKS;

    if (max_num_blocks != 0){
	       tot_num_blocks = MIN(max_num_blocks + MEM_NUM_SAVE_BLOCKS, tot_num_blocks);
    }

    HIP_CHECK(hipSetDevice(deviceId));

    hipDeviceSynchronize();

    HIP_CHECK(hipMemGetInfo(&free, &total));

    msg = "[" + action_name + "] " + MODULE_NAME + " " +
            std::to_string(gpu_id) + " " + "Toal Memory from hipMemGetInfo " + " " +
            std::to_string(total) + " " + " Free Memory from hipMemGetInfo " + " " + 
            std::to_string(free);
    rvs::lp::Log(msg, rvs::logtrace);

    allocate_small_mem();

    tot_num_blocks = MIN(tot_num_blocks, free/BLOCKSIZE - MEM_NUM_SAVE_BLOCKS);

    msg = "[" + action_name + "] " + MODULE_NAME + " " +
            std::to_string(gpu_id) + " " + "Toal Num of blocks " + " " +
            std::to_string(tot_num_blocks); 

    rvs::lp::Log(msg, rvs::logtrace);

    do{
        tot_num_blocks -= MEM_NUM_SAVE_BLOCKS ; //magic number 16 MB

        if (tot_num_blocks <= 0){
            msg = "[" + action_name + "] " + MODULE_NAME + " " +
                           std::to_string(gpu_id) + " " + " Total Number of blocks is zero, cant allocate memory" + " " +
                           std::to_string(tot_num_blocks); 

            rvs::lp::Log(msg, rvs::logtrace);
            return; 

        }

        if(useMappedMemory)
        {
            //create HIP mapped memory
            hipHostMalloc((void**)&mappedHostPtr, tot_num_blocks* BLOCKSIZE, hipHostMallocWriteCombined | hipHostMallocMapped);

            hipHostGetDevicePointer(&mappedHostPtr, &ptr, 0);

        }
        else
        {
             HIP_CHECK(hipMalloc((void**)&ptr, tot_num_blocks* BLOCKSIZE));
        }

    }while(hipGetLastError() != hipSuccess);

    std::lock_guard<std::mutex> lck(mtx_mem_test);

    msg = "[" + action_name + "] " + MODULE_NAME + " " +
                  std::to_string(gpu_id) + " " + "Starting running tests " + " " + 
                  "Total Num of blocks " + std::to_string(tot_num_blocks);

    rvs::lp::Log(msg, rvs::logtrace);

    run_tests(ptr, tot_num_blocks);

    free_small_mem();
}



