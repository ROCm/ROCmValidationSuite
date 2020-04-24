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


#define MODULE_NAME                             "mem"
#define MEM_MEM_ALLOC_ERROR                     "memory allocation error!"
#define MEM_BLAS_ERROR                          "memory/blas error!"
#define MEM_BLAS_MEMCPY_ERROR                   "HostToDevice mem copy error!"
#define MAX_ERR_RECORD_COUNT                    10
#define MEM_NUM_SAVE_BLOCKS                     16

#define MEM_START_MSG                           "start"
#define MEM_PASS_KEY                            "pass"
#define MEM_RAMP_EXCEEDED_MSG                   "ramp time exceeded"
#define MEM_TARGET_ACHIEVED_MSG                 "target achieved"
#define MEM_STRESS_VIOLATION_MSG                "stress violation"

using std::string;

extern unsigned int    blocks;
extern uint64_t        threadsPerBlock;
extern bool            useMappedMemory;
extern void*           mappedHostPtr;

extern unsigned int    *ptCntOfError;
extern unsigned long   *ptFailedAdress;
extern unsigned long   *ptExpectedValue;
extern unsigned long   *ptCurrentValue;
extern unsigned long   *ptValueOfStartAddr;
 

bool MemWorker::bjson = false;

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

void MemWorker::list_tests_info(void)
{
    size_t i;

    for (i = 0;i < DIM(rvs_memtests); i++){
	          printf("%s %s\n", rvs_memtests[i].desc, rvs_memtests[i].enabled?"":" ==disabled by default==");
    }

    return;
}


void MemWorker::usage(char** argv)
{

    char example_usage[] =
	      "run on default setting:       ./rvs_memtest\n"
	      "run on stress test only:      ./rvs_memtest --stress\n";

    printf("Usage:%s [options]\n", argv[0]);
    printf("options:\n");
    printf("--mappedMem                 run all checks with rvs mapped memory instead of native device memory\n");
    printf("--silent                    Do not print out progress message (default)\n");
    printf("--device <idx>              Designate one device for test\n");
    printf("--interactive               Progress info will be printed in the same line\n");
    printf("--disable_all               Disable all tests\n");
    printf("--enable_test <test_idx>    Enable the test <test_idx>\n");
    printf("--disable_test <test_idx>   Disable the test <test_idx>\n");
    printf("--max_num_blocks <n>        Set the maximum of blocks of memory to test\n");
    printf("                            1 block = 1 MB in here\n");
    printf("--exit_on_error             When finding error, print error message and exit\n");
    printf("--monitor_temp <interval>   Monitoring temperature, the temperature will be updated every <interval> seconds\n");
    printf("                            This feature is experimental\n");
    printf("--emails <a@b,c@d,...>      Setting email notification\n");
    printf("--report_interval <n>       Setting the interval in seconds between email notifications(default 1800)\n");
    printf("--pattern <pattern>         Manually set test pattern for test4/test8/test10\n");
    printf("--list_tests                List all test descriptions\n");
    printf("--num_iterations <n>        Set the number of iterations (only effective on test0 and test10)\n");
    printf("--num_passes <n>            Set the number of test passes (this affects all tests)\n");
    printf("--verbose <n>               Setting verbose level\n");
    printf("                              0 -- Print out test start and end message only (default)\n");
    printf("                              1 -- Print out pattern messages in test\n");
    printf("                              2 -- Print out progress messages\n");
    printf("--stress                    Stress test. Equivalent to --disable_all --enable_test 10 --exit_on_error\n");
    printf("--help                      Print this message\n");
    printf("\nExample usage:\n\n");
    printf("%s\n", example_usage);

    exit(ERR_GENERAL);
}

void MemWorker::run_tests(char* ptr, unsigned int tot_num_blocks)
{
    struct timeval  t0, t1;
    unsigned int pass = 0;
    unsigned int i;

    blocks = 512;
    threadsPerBlock = 256;
    num_iterations = 1;

    for(int n = 0; n < num_iterations; n++) {
        for (i = 0;i < DIM(rvs_memtests); i++){
            gettimeofday(&t0, NULL);
            rvs_memtests[i].func(ptr, tot_num_blocks);
            gettimeofday(&t1, NULL);
            hipDeviceReset();
            std::cout << "\n To run memtest time taken : " << TDIFF(t1, t0) << " seconds with " << num_iterations << " passes\n";
        }//for
        std::cout << "\n Memory tests :: " << std::dec << i << " tests complete \n ";
    }

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


    msg = "[" + action_name + "] " + MODULE_NAME + " " +
                  std::to_string(gpu_id) + " " + "Starting running tests " + " " + 
                  "Total Num of blocks " + std::to_string(tot_num_blocks);

    rvs::lp::Log(msg, rvs::logtrace);

    run_tests(ptr, tot_num_blocks);

    free_small_mem();
}



