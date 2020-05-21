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
#include <sys/time.h>
#include <mutex>

#include "hip/hip_runtime.h"
#include "include/rvs_memworker.h"
#include "include/rvs_memtest.h"
#include "include/rvsloglp.h"

using std::string;

extern unsigned int    *ptCntOfError;
extern unsigned long   *ptFailedAdress;
extern unsigned long   *ptExpectedValue;
extern unsigned long   *ptCurrentValue;
extern unsigned long   *ptValueOfStartAddr;
bool MemWorker::bjson = false;
extern rvs_memdata   memdata;
 


MemWorker::MemWorker() {}
MemWorker::~MemWorker() {}

rvs_memtest_t rvs_memtests[]={
    {test0, (char*)"Test0 [Walking 1 bit]",			1},
    {test1, (char*)"Test1 [Own address test]",			1},
    {test2, (char*)"Test2 [Moving inversions, ones&zeros]",	1},
    {test3, (char*)"Test3 [Moving inversions, 8 bit pat]",	1},
    {test4, (char*)"Test4 [Moving inversions, random pattern]",1},
    {test5, (char*)"Test5 [Block move, 64 moves]",		1},
    {test6, (char*)"Test6 [Moving inversions, 32 bit pat]",	1},
    {test7, (char*)"Test7 [Random number sequence]",		1},
    {test8, (char*)"Test8 [Modulo 20, random pattern]",	1},
    {test9, (char*)"Test9 [Bit fade test]",			0},
    {test10, (char*)"Test10 [Memory stress test]",		1},
};


void MemWorker::allocate_small_mem(void)
{
    //Initialize memory
    HIP_CHECK(hipMalloc((void**)&ptCntOfError, sizeof(unsigned int) )); 
    HIP_CHECK(hipMemset(ptCntOfError, 0, sizeof(unsigned int) )); 

    HIP_CHECK(hipMalloc((void**)&ptFailedAdress, sizeof(unsigned long) * MAX_ERR_RECORD_COUNT));
    HIP_CHECK(hipMemset(ptFailedAdress, 0, sizeof(unsigned long) * MAX_ERR_RECORD_COUNT));

    HIP_CHECK(hipMalloc((void**)&ptExpectedValue, sizeof(unsigned long) * MAX_ERR_RECORD_COUNT));
    HIP_CHECK(hipMemset(ptExpectedValue, 0, sizeof(unsigned long) * MAX_ERR_RECORD_COUNT));

    HIP_CHECK(hipMalloc((void**)&ptCurrentValue, sizeof(unsigned long) * MAX_ERR_RECORD_COUNT));
    HIP_CHECK(hipMemset(ptCurrentValue, 0, sizeof(unsigned long) * MAX_ERR_RECORD_COUNT));

    HIP_CHECK(hipMalloc((void**)&ptValueOfStartAddr, sizeof(unsigned long) * MAX_ERR_RECORD_COUNT));
    HIP_CHECK(hipMemset(ptValueOfStartAddr, 0, sizeof(unsigned long) * MAX_ERR_RECORD_COUNT));
}

void MemWorker::free_small_mem(void)
{
    //Initialize memory
    hipFree((void*)&ptCntOfError);

    hipFree((void*)ptFailedAdress);

    hipFree((void*)ptExpectedValue);

    hipFree((void*)ptCurrentValue);

    hipFree((void*)ptValueOfStartAddr);
}

void MemWorker::Initialization(void)
{
    memdata.threadsPerBlock = get_threads_per_block();
    memdata.blocks = get_num_mem_blocks();
    memdata.num_passes = get_num_passes();
    memdata.global_pattern = 0;
    memdata.global_pattern_long = 0;
    memdata.action_name = action_name;
    memdata.gpu_idx = gpu_id;
    memdata.num_iterations = num_iterations;
}
 
void MemWorker::run_tests(char* ptr, unsigned int tot_num_blocks)
{
    struct timeval  t0, t1;
    unsigned int pass = 0;
    unsigned int i;
    std::string msg;

    Initialization();

    for (i = 0; i < DIM(rvs_memtests); i++){
          gettimeofday(&t0, NULL);
          rvs_memtests[i].func(ptr, tot_num_blocks);
          gettimeofday(&t1, NULL);
          msg = "[" + action_name + "] " + MODULE_NAME + " " +
                   std::to_string(gpu_id) + " To run memtest time taken: " + std::to_string(TDIFF(t1, t0)) + " seconds with " + std::to_string(i) + " passes \n";
          rvs::lp::Log(msg, rvs::loginfo);
     }//for

     msg = "[" + action_name + "] " + MODULE_NAME + " " +
                   std::to_string(gpu_id) + " " + " Memory tests : " + std::to_string(i) + " tests complete \n";
     rvs::lp::Log(msg, rvs::loginfo);
}


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
            std::to_string(gpu_id) + " "  + " Starting the Memory stress test "; 
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


         msg = "[" + action_name + "] " + MODULE_NAME + " " +
                             std::to_string(gpu_id) + " " + "Use mapped memory  " + " " +
                             std::to_string(useMappedMemory) + " Block Size: " +  std::to_string(BLOCKSIZE); 

         rvs::lp::Log(msg, rvs::loginfo);

         unsigned int alloc_size =  tot_num_blocks* BLOCKSIZE;

         if(useMappedMemory == true) {

           msg = "[" + action_name + "] " + MODULE_NAME + " " +
                             std::to_string(gpu_id) + " " + "Memory to be allocated: " + std::to_string(alloc_size); 

           rvs::lp::Log(msg, rvs::loginfo);

            //create HIP mapped memory
            HIP_CHECK(hipHostMalloc((void**)&mappedHostPtr, alloc_size, hipHostMallocWriteCombined | hipHostMallocMapped));

            HIP_CHECK(hipHostGetDevicePointer((void**)&ptr, mappedHostPtr, 0));

        }
        else
        {

             msg = "[" + action_name + "] " + MODULE_NAME + " " +
                             std::to_string(gpu_id) + " " + "Memory to be allocated: " + std::to_string(alloc_size); 

             rvs::lp::Log(msg, rvs::loginfo);

             HIP_CHECK(hipMalloc((void**)&ptr, alloc_size));
        }

    }while(hipGetLastError() != hipSuccess);


    msg = "[" + action_name + "] " + MODULE_NAME + " " + std::to_string(gpu_id) + " " + "Starting running tests " + " " + 
                  "Total Num of blocks " + std::to_string(tot_num_blocks);

    rvs::lp::Log(msg, rvs::logtrace);

    run_tests(ptr, tot_num_blocks);

    free_small_mem();
}



