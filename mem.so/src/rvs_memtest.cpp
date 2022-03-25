/*
 * Illinois Open Source License
 *
 * University of Illinois/NCSA
 * Open Source License
 *
 * Copyright � 2009,    University of Illinois.  All rights reserved.
 *
 * Developed by:
 *
 * Innovative Systems Lab
 * National Center for Supercomputing Applications
 * http://www.ncsa.uiuc.edu/AboutUs/Directorates/ISL.html
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal with
 * the Software without restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
 * Software, and to permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * * Redistributions of source code must retain the above copyright notice, this list
 * of conditions and the following disclaimers.
 *
 * * Redistributions in binary form must reproduce the above copyright notice, this list
 * of conditions and the following disclaimers in the documentation and/or other materials
 * provided with the distribution.
 *
 * * Neither the names of the Innovative Systems Lab, the National Center for Supercomputing
 * Applications, nor the names of its contributors may be used to endorse or promote products
 * derived from this Software without specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 * PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
 * OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 */


#include <iostream>
#include <pthread.h>
#include <thread>
#include <chrono>
#include <cstdio>
#include <sys/time.h>
#include <unistd.h>
#include <sstream>
#include <mutex>



#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"


#include "include/rvs_key_def.h"
#include "include/rvs_util.h"
#include "include/rvsactionbase.h"
#include "include/rvsloglp.h"
#include "include/action.h"
#include "include/rvs_memworker.h"
#include "include/gpu_util.h"
#include "include/rvs_memkernel.h"
#include "include/rvs_memtest.h"

unsigned int     blocks = 512;
unsigned int     threadsPerBlock = 256;

static __thread  unsigned long* ptFailedAdress;
static __thread  unsigned long* ptExpectedValue;
static __thread  unsigned long* ptCurrentValue;
static __thread  unsigned long* ptValueOfSecondRead;
static __thread  unsigned int*  ptCntOfError;

rvs_memdata   memdata;

void show_progress(std::string msg, unsigned int i, unsigned int tot_num_blocks)	{
    unsigned int num_checked_blocks;
    std::string buff;

    hipDeviceSynchronize();						
    num_checked_blocks =  i + GRIDSIZE <= tot_num_blocks? i + GRIDSIZE: tot_num_blocks; 
    // log MEM stress test - start message
    msg += ": " + std::to_string(num_checked_blocks) + " out of " + std::to_string(tot_num_blocks) + " blocks finished"; 
    buff = "[" + memdata.action_name + "] " + MODULE_NAME + " " + std::to_string(memdata.gpu_idx) + msg;
    rvs::lp::Log(buff, rvs::loginfo);
}



unsigned int error_checking(std::string pmsg, unsigned int blockidx)
{
    unsigned long host_err_addr[MAX_ERR_RECORD_COUNT];
    unsigned long host_err_expect[MAX_ERR_RECORD_COUNT];
    unsigned long host_err_current[MAX_ERR_RECORD_COUNT];
    unsigned long host_err_second_read[MAX_ERR_RECORD_COUNT];
    unsigned int  numOfErrors = 0;
    unsigned int  i;
    std::string   msg;
    unsigned int  reported_errors;

    
    HIP_CHECK(hipMemcpy(&numOfErrors, (void*)ptCntOfError, sizeof(unsigned int), hipMemcpyDeviceToHost));
    if(numOfErrors == 0){ // No point to continue 
			return 0;
		}
    HIP_CHECK(hipMemcpy(&host_err_addr[0], (void*)ptFailedAdress, sizeof(unsigned long)*MAX_ERR_RECORD_COUNT, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&host_err_expect[0], (void*)ptExpectedValue, sizeof(unsigned long)*MAX_ERR_RECORD_COUNT, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&host_err_current[0], (void*)ptCurrentValue, sizeof(unsigned long)*MAX_ERR_RECORD_COUNT, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&host_err_second_read[0], (void*)ptValueOfSecondRead, sizeof(unsigned long)*MAX_ERR_RECORD_COUNT, hipMemcpyDeviceToHost));
    reported_errors = MIN(MAX_ERR_RECORD_COUNT, numOfErrors);
    msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " + pmsg + " block id :" + std::to_string(blockidx);
    rvs::lp::Log(msg, rvs::loginfo);

    msg = "[" + memdata.action_name + "] " + MODULE_NAME + " Number of errors :" + std::to_string(numOfErrors);
    rvs::lp::Log(msg, rvs::loginfo);

    msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " + "ERROR: the last : " +  
              std::to_string(reported_errors) + " : error addresses are: \n";
    rvs::lp::Log(msg, rvs::loginfo);

	  for (i = 0; i < reported_errors ; i++){

            msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " +  
                       std::to_string(host_err_addr[i]) + " \n ";
            rvs::lp::Log(msg, rvs::loginfo);
	  }
    msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " + "ERROR: the last :" + 
              std::to_string(reported_errors) + " : error details are : \n";
    rvs::lp::Log(msg, rvs::loginfo);
	  for (i =0; i < reported_errors; i++){

          msg = "[" + memdata.action_name + "] " + MODULE_NAME + " "  +  
                    " ERROR:" + std::to_string(i) + " th error, expected value=0x" +  std::to_string(host_err_expect[i]) +  
                    "current value=0x" + std::to_string(host_err_current[i]) + "current value=0x" + 
                    std::to_string(host_err_current[i]) + 
                    " second_ read=0x " + std::to_string(host_err_second_read[i]) +  "\n \n";
          rvs::lp::Log(msg, rvs::loginfo);
	  }


	  hipMemset(ptCntOfError, 0, sizeof(unsigned int));
	  hipMemset((void*)&ptFailedAdress[0], 0, sizeof(unsigned long)*MAX_ERR_RECORD_COUNT);;
    hipMemset((void*)&ptExpectedValue[0], 0, sizeof(unsigned long)*MAX_ERR_RECORD_COUNT);;
	  hipMemset((void*)&ptCurrentValue[0], 0, sizeof(unsigned long)*MAX_ERR_RECORD_COUNT);;

	  hipDeviceReset();
	  exit(ERR_BAD_STATE);

}

unsigned int get_random_num(void) {
    struct timeval t0;

    if (gettimeofday(&t0, NULL) !=0){

	       fprintf(stderr, "ERROR: gettimeofday() failed\n");
	       exit(ERR_GENERAL);
    }

    unsigned int seed= (unsigned int)t0.tv_sec;
    srand(seed);

    return rand_r(&seed);
}



uint64_t get_random_num_long(void)
{
    unsigned int a = get_random_num(); 
    unsigned int b = get_random_num();

    uint64_t ret =  ((uint64_t)a) << 32;
    ret |= ((uint64_t)b);

    return ret;
}

__global__  void kernel_test0_global_write(char* _ptr, char* _end_ptr)
 {
     unsigned int* ptr = (unsigned int*)_ptr;
     unsigned int* end_ptr = (unsigned int*)_end_ptr;
     unsigned int* orig_ptr = ptr;
     unsigned int pattern = 1;
     unsigned long mask = 4;

     *ptr = pattern;

     while(ptr < end_ptr){
         ptr = (unsigned int*) ( ((unsigned long)orig_ptr) | mask);

         if (ptr == orig_ptr){
             mask = mask <<1;
             continue;
         }

         if (ptr >= end_ptr){
             break;
         }

         *ptr = pattern;

         pattern = pattern << 1;
         mask = mask << 1;
     }
     return;
 }

 __global__ void kernel_test0_write(char* _ptr, char* end_ptr)
{
    unsigned int* orig_ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);
    unsigned int* ptr = orig_ptr;

    if (ptr >= (unsigned int*) end_ptr) {
	      return;
    }

    unsigned int* block_end = orig_ptr + BLOCKSIZE/sizeof(unsigned int);
    unsigned int pattern = 1;
    unsigned long mask = 4;

    *ptr = pattern;

    while(ptr < block_end){
	    ptr = (unsigned int*) ( ((unsigned long)orig_ptr) | mask);

	    if (ptr == orig_ptr){
	        mask = mask <<1;
	        continue;
	    }

	    if (ptr >= block_end){
	        break;
	    }

	    *ptr = pattern;

	    pattern = pattern << 1;
	    mask = mask << 1;
    }

    return;
}

__global__ void kernel_test0_global_read(char* _ptr, char* _end_ptr, unsigned int* ptErrCount, unsigned long* ptFailedAdress,
		  unsigned long* ptExpectedValue, unsigned long* ptCurrentValue, unsigned long* ptValueofSecondRead)
{
    unsigned long* ptr = (unsigned long*)_ptr;
    unsigned long* end_ptr = (unsigned long*)_end_ptr;
    unsigned long* orig_ptr = ptr;
    unsigned int pattern = 1;
    unsigned long mask = 4;

    if (*ptr != pattern){
	    if( *ptErrCount < MAX_ERR_RECORD_COUNT) {
           ptFailedAdress[*ptErrCount] = (unsigned long)ptr;        
           ptExpectedValue[*ptErrCount] = (unsigned long)pattern;  
           ptCurrentValue[*ptErrCount++] = (unsigned long)*ptr;   
	      }
    }

    while(ptr < end_ptr){
        ptr = (unsigned long*) ( ((unsigned long)orig_ptr) | mask);

        if (ptr == orig_ptr){
	          mask = mask << 1;
	          continue;
        }

	      if (ptr >= end_ptr){
		        break;
	      }
	      if( *ptErrCount < MAX_ERR_RECORD_COUNT ) {
             ptFailedAdress[*ptErrCount] = (unsigned long)ptr;
             ptExpectedValue[*ptErrCount] = (unsigned long)pattern;
             ptCurrentValue[*ptErrCount++] = (unsigned long)*ptr;
        }

	      pattern = pattern << 1;
	      mask = mask << 1;
    }

    return;
}

__global__ void kernel_test0_read(char* _ptr, char* end_ptr, unsigned int* ptErrCount, unsigned long* ptFailedAdress,
		  unsigned long* ptExpectedValue, unsigned long* ptCurrentValue, unsigned long* ptValueOfSecondRead)
{
    unsigned int* orig_ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);;
    unsigned int* ptr = orig_ptr;

    if (ptr >= (unsigned int*) end_ptr) {
	    return;
    }

    unsigned int* block_end = orig_ptr + BLOCKSIZE/sizeof(unsigned int);

    unsigned int pattern = 1;

    unsigned long mask = 4;

    if (*ptr != pattern){
	      if( *ptErrCount < MAX_ERR_RECORD_COUNT ) {
             ptFailedAdress[*ptErrCount] = (unsigned long)ptr;                
             ptExpectedValue[*ptErrCount] = (unsigned long)pattern;   
             ptCurrentValue[*ptErrCount++] = (unsigned long)*ptr;   
        }
    }

    while(ptr < block_end){
	      ptr = (unsigned int*) ( ((unsigned long)orig_ptr) | mask);
	      if (ptr == orig_ptr){
	          mask = mask <<1;
	          continue;
	      }

	      if (ptr >= block_end){
	          break;
	      }

	      if (*ptr != pattern){
	          if( *ptErrCount < MAX_ERR_RECORD_COUNT ) {
                 ptFailedAdress[*ptErrCount] = (unsigned long)ptr;          
                 ptExpectedValue[*ptErrCount] = (unsigned long)pattern;  
                 ptCurrentValue[*ptErrCount++] = (unsigned long)*ptr;   
	          }
	      }

	      pattern = pattern << 1;
	      mask = mask << 1;
    }

}

/************************************************************************
 * Test0 [Walking 1 bit]
 * This test changes one bit a time in memory address to see it
 * goes to a different memory location. It is designed to test
 * the address wires.
 *
 **************************************************************************/

void test0(char* _ptr, unsigned int tot_num_blocks)
{
    unsigned int    i;
    char *ptr = _ptr;
    char* end_ptr = ptr + tot_num_blocks* BLOCKSIZE;
    std::string msg;
   
    msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " + "Test 1: Change one bit memory addresss  ";
    rvs::lp::Log(msg, rvs::logresults);

    //test global address
    hipLaunchKernelGGL(kernel_test0_global_write,
        dim3(memdata.blocks), dim3(memdata.threadsPerBlock),  0, 0, ptr, end_ptr);

    hipLaunchKernelGGL(kernel_test0_global_read, 
        dim3(memdata.blocks), dim3(memdata.threadsPerBlock),  0, 0, ptr, end_ptr, 
        ptCntOfError, ptFailedAdress, ptExpectedValue, ptCurrentValue, ptValueOfSecondRead); 

    msg = " Test 1 on global address";
    error_checking(msg,  0);

    for(unsigned int ite = 0; ite < memdata.num_passes; ite++){

        for (i = 0; i < tot_num_blocks; i += GRIDSIZE){
	          dim3 grid;

            grid.x= GRIDSIZE;
            hipLaunchKernelGGL(kernel_test0_write,  
                dim3(memdata.blocks), dim3(memdata.threadsPerBlock),  0, 0, ptr + i * BLOCKSIZE, end_ptr); 
		        show_progress(" Test 1 on writing :", i, tot_num_blocks);
	      }

	      for (i=0;i < tot_num_blocks; i+= GRIDSIZE){
	          dim3 grid;

	          grid.x= GRIDSIZE;

            hipLaunchKernelGGL(kernel_test0_read,
                dim3(memdata.blocks), dim3(memdata.threadsPerBlock),  0, 0, ptr + i * BLOCKSIZE, end_ptr, 
                ptCntOfError, ptFailedAdress, ptExpectedValue, ptCurrentValue, ptValueOfSecondRead); 

		        error_checking("Test 1",  i);
		        show_progress(" Test 1 on reading :", i, tot_num_blocks);
	        }

    }

    msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " + "Test 1 : PASS";
    rvs::lp::Log(msg, rvs::logresults);

    return;

}

/*********************************************************************************
 * test1
 * Each Memory location is filled with its own address. The next kernel checks if the
 * value in each memory location still agrees with the address.
 *
 ********************************************************************************/
__global__ void kernel_test1_write(char* _ptr, char* end_ptr, unsigned int* err)
{
    unsigned int i;
    unsigned long* ptr = (unsigned long*) (_ptr + blockIdx.x*BLOCKSIZE);

    if (ptr >= (unsigned long*) end_ptr) {
	      return;
    }

    for (i = 0;i < BLOCKSIZE/sizeof(unsigned long); i++){
	      ptr[i] =(unsigned long) & ptr[i];
    }

    return;
}

__global__ void 
kernel_test1_read(char* _ptr, char* end_ptr, unsigned int* ptErrCount, unsigned long* ptFailedAdress,
		  unsigned long* ptExpectedValue, unsigned long* ptCurrentValue, unsigned long* ptValueOfSecondRead)
{
    unsigned int i;
    unsigned long* ptr = (unsigned long*) (_ptr + blockIdx.x*BLOCKSIZE);

    if (ptr >= (unsigned long*) end_ptr) {
	      return;
    }

    for (i = 0;i < BLOCKSIZE/sizeof(unsigned long); i++){
	    if (ptr[i] != (unsigned long)& ptr[i]){
	       if((*ptErrCount >= 0) && (*ptErrCount < MAX_ERR_RECORD_COUNT)) {
                   ptFailedAdress[*ptErrCount] = (unsigned long)&ptr[i];      
                   ptExpectedValue[*ptErrCount] = (unsigned long)&ptr[i];   
                   ptCurrentValue[*ptErrCount++] = (unsigned long)ptr[i];   
	       }
	    }
    }

    return;
}

void test1(char* ptr, unsigned int tot_num_blocks)
{
    unsigned int err;
    unsigned int i;
    char*        end_ptr = ptr + tot_num_blocks * BLOCKSIZE;
    std::string  msg;

    msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " + "Test 2: Each Memory location is filled with its own address";
    rvs::lp::Log(msg, rvs::logresults);

    for (i = 0; i < tot_num_blocks; i += GRIDSIZE){
	    dim3 grid;

	    grid.x= GRIDSIZE;
            hipLaunchKernelGGL(kernel_test1_write, 
                     dim3(memdata.blocks), dim3(memdata.threadsPerBlock),  0/*dynamic shared*/, 0/*stream*/,     /* launch config*/
	                   (ptr + (i * BLOCKSIZE)) , end_ptr, ptCntOfError); 

	    show_progress("Test1 on writing", i, tot_num_blocks);
    }

    for (i=0;i < tot_num_blocks; i+= GRIDSIZE){
	    dim3 grid;

	    grid.x= GRIDSIZE;
            hipLaunchKernelGGL(kernel_test1_read,
                            dim3(memdata.blocks), dim3(memdata.threadsPerBlock), 0/*dynamic shared*/, 0/*stream*/,     /* launch config*/
	                          ptr + (i * BLOCKSIZE), end_ptr, ptCntOfError, 
                            ptFailedAdress, ptExpectedValue, ptCurrentValue, ptValueOfSecondRead);

            err += error_checking("Test2 checking :: ",  i);
	    show_progress("\nTest2 on reading", i, tot_num_blocks);
    }

    if(!err) {
      msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " + "Test 2 : PASS";
      rvs::lp::Log(msg, rvs::logresults);
    }
    return;
}



/******************************************************************************
 * Test 2 [Moving inversions, ones&zeros]
 * This test uses the moving inversions algorithm with patterns of all
 * ones and zeros.
 *
 ****************************************************************************/

__global__ void 
kernel_move_inv_write(char* _ptr, char* end_ptr, unsigned int pattern)
{
    unsigned int *ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);
    unsigned int  i;

    if (ptr >= (unsigned int*) end_ptr) {
	    return;
    }

    for (i = 0;i < BLOCKSIZE/sizeof(unsigned int); i++){
	      ptr[i] = pattern;
    }

    return;
}


__global__ void 
kernel_move_inv_readwrite(char* _ptr, char* end_ptr, unsigned int p1, unsigned int p2, unsigned int* ptErrCount,
			  unsigned long* ptFailedAdress, unsigned long* ptExpectedValue, unsigned long* ptCurrentValue, unsigned long* ptValueOfSecondRead)
{
    unsigned int i;
    unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x * BLOCKSIZE);

    if (ptr >= (unsigned int*) end_ptr) {
	      return;
    }

    for (i = 0;i < BLOCKSIZE/sizeof(unsigned int); i++){
	 if (ptr[i] != p1){
               if((*ptErrCount >= 0) && (*ptErrCount < MAX_ERR_RECORD_COUNT)) {
                     ptFailedAdress[*ptErrCount] = (unsigned long)&ptr[i];        
                     ptExpectedValue[*ptErrCount] = (unsigned long)p1;   
                     ptCurrentValue[*ptErrCount++] = (unsigned long)ptr[i];   
               }
	 }

	 ptr[i] = p2;
    }

    return;
}


__global__ void 
kernel_move_inv_read(char* _ptr, char* end_ptr,  unsigned int pattern, unsigned int* ptErrCount,
		     unsigned long* ptFailedAdress, unsigned long* ptExpectedValue, unsigned long* ptCurrentValue, unsigned long* ptValueOfSecondRead )
{
    unsigned int i;
    unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x * BLOCKSIZE);

    if (ptr >= (unsigned int*) end_ptr) {
	      return;
    }

    for (i = 0;i < BLOCKSIZE/sizeof(unsigned int); i++){
        if (ptr[i] != pattern){
            if((*ptErrCount >= 0) && (*ptErrCount < MAX_ERR_RECORD_COUNT)) {
                  ptFailedAdress[*ptErrCount] = (unsigned long)&ptr[i];        
                  ptExpectedValue[*ptErrCount] = (unsigned long)pattern;   
                  ptCurrentValue[*ptErrCount++] = (unsigned long)ptr[i];   
            }
	}
    }

    return;
}


unsigned int  move_inv_test(char* ptr, unsigned int tot_num_blocks, unsigned int p1, unsigned p2)
{
    unsigned int i;
    unsigned int err = 0;
    char* end_ptr = ptr + tot_num_blocks* BLOCKSIZE;

    for (i= 0;i < tot_num_blocks; i+= GRIDSIZE){
        dim3 grid;

        grid.x= GRIDSIZE;
        hipLaunchKernelGGL(kernel_move_inv_write,
                         dim3(memdata.blocks), dim3(memdata.threadsPerBlock), 0/*dynamic shared*/, 0/*stream*/,     /* launch config*/
	                 ptr + i * BLOCKSIZE, end_ptr,  p1); 

        show_progress("move_inv_write", i, tot_num_blocks);

    }


    for (i=0; i < tot_num_blocks; i+= GRIDSIZE){
        dim3 grid;

        grid.x= GRIDSIZE;
        hipLaunchKernelGGL(kernel_move_inv_readwrite,
                         dim3(memdata.blocks), dim3(memdata.threadsPerBlock), 0/*dynamic shared*/, 0/*stream*/,     /* launch config*/
	                 ptr + i*BLOCKSIZE, end_ptr, p1, p2, ptCntOfError, ptFailedAdress, ptExpectedValue, ptCurrentValue, ptValueOfSecondRead); 

        err += error_checking("Move inv reading and writing to blocks",  i);
        show_progress("move_inv_readwrite", i, tot_num_blocks);
    }

    for (i=0; i < tot_num_blocks; i+= GRIDSIZE){
        dim3 grid;

        grid.x= GRIDSIZE;
        hipLaunchKernelGGL(kernel_move_inv_read,
                         dim3(memdata.blocks), dim3(memdata.threadsPerBlock), 0/*dynamic shared*/, 0/*stream*/,     /* launch config*/
	                       ptr + i*BLOCKSIZE, end_ptr, p2, ptCntOfError, ptFailedAdress, ptExpectedValue, ptCurrentValue, ptValueOfSecondRead); 
        err += error_checking("Move inv reading from blocks",  i);
        show_progress("move_inv_read", i, tot_num_blocks);
    }

    return err;

}


void test2(char* ptr, unsigned int tot_num_blocks)
{
    unsigned int p1 = 0;
    unsigned int p2 = ~p1;
    unsigned int err = 0;
    std::string  msg;

    msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " + "Test 3 [Moving inversions, ones&zeros] " +
                         std::to_string(p1) + " and " + std::to_string(p2) + "\n";
    rvs::lp::Log(msg, rvs::logresults);

    msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " + "Test 3: Moving inversions test, with pattern " 
      + std::to_string(p1) + " and " + std::to_string(p2) + "\n";
    rvs::lp::Log(msg, rvs::loginfo);

    err = move_inv_test(ptr, tot_num_blocks, p1, p2);

    if(!err) {
       msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " + "Test 3 Moving inversions test p1 p2 passed, no errors detected \n";
       rvs::lp::Log(msg, rvs::loginfo);
    }

    msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " + "Test 3: Moving inversions test, with pattern " + 
                  std::to_string(p2) + " and " + std::to_string(p1) + "\n";
    rvs::lp::Log(msg, rvs::loginfo);

    err = move_inv_test(ptr, tot_num_blocks, p2, p1);

    if(!err) {
        msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " + "Test 3 : PASS ";
        rvs::lp::Log(msg, rvs::logresults);
    }
}


/*************************************************************************
 *
 * Test 3 [Moving inversions, 8 bit pat]
 * This is the same as test 1 but uses a 8 bit wide pattern of
 * "walking" ones and zeros.  This test will better detect subtle errors
 * in "wide" memory chips.
 *
 **************************************************************************/


void test3(char* ptr, unsigned int tot_num_blocks)
{
    unsigned int p0=0x80;
    unsigned int p1 = p0 | (p0 << 8) | (p0 << 16) | (p0 << 24);
    unsigned int p2 = ~p1;
    unsigned int err = 0;
    std::string  msg;

    msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " + "Test 4 [Moving inversions, 8 bit pat]"
                   + std::to_string(p1) + " and " + std::to_string(p2) + "\n";
    rvs::lp::Log(msg, rvs::logresults);

    err = move_inv_test(ptr, tot_num_blocks, p1, p2);

    if(!err) {
         msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " + "Moving inversions successful";
         rvs::lp::Log(msg, rvs::loginfo);
    }

    msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " + "Test 4 [Moving inversions, 8 bit pat, reverse]"
                   + std::to_string(p2) + " and " + std::to_string(p1) + "\n";
    rvs::lp::Log(msg, rvs::loginfo);
    err = move_inv_test(ptr, tot_num_blocks, p2, p1);

    if(!err) {
         msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " + "Test 4 : PASS";
         rvs::lp::Log(msg, rvs::logresults);
    }
}


/************************************************************************************
 * Test 4 [Moving inversions, random pattern]
 * Test 4 uses the same algorithm as test 1 but the data pattern is a
 * random number and it's complement. This test is particularly effective
 * in finding difficult to detect data sensitive errors. A total of 60
 * patterns are used. The random number sequence is different with each pass
 * so multiple passes increase effectiveness.
 *
 *************************************************************************************/

void test4(char* ptr, unsigned int tot_num_blocks)
{
    unsigned int p1;
    std::string  msg;

    msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " + "Test 5 [Moving inversions, random pattern] \n";
    rvs::lp::Log(msg, rvs::logresults);

    if (memdata.global_pattern == 0){
	    p1 = get_random_num();
    }else{
	    p1 = memdata.global_pattern;
    }

    unsigned int p2 = ~p1;
    unsigned int err = 0;
    unsigned int iteration = 0;

    msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " + "Random number :: p1" + std::to_string(p1) + " p2 :: " + std::to_string(p2); 
    rvs::lp::Log(msg, rvs::loginfo);

    repeat:
          err += move_inv_test(ptr, tot_num_blocks, p1, p2);

          if (err == 0 && iteration == 0){

            msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " + "Test 5 PASS, no errors detected , iterations are zero here";
            rvs::lp::Log(msg, rvs::logresults);
	          return;
          }

          if (iteration < memdata.num_iterations){
	          iteration++;
            msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " + "th repeating test4 because there are" 
                            + std::to_string(err) + "errors found in last run\n";
            rvs::lp::Log(msg, rvs::loginfo);
	          err = 0;
	          goto repeat;
          }

    if(!err) {
        msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " + "Test 5 : PASS";
        rvs::lp::Log(msg, rvs::logresults);
    }
}


/************************************************************************************
 * Test 5 [Block move, 64 moves]
 * This test stresses memory by moving block memories. Memory is initialized
 * with shifting patterns that are inverted every 8 bytes.  Then blocks
 * of memory are moved around.  After the moves
 * are completed the data patterns are checked.  Because the data is checked
 * only after the memory moves are completed it is not possible to know
 * where the error occurred.  The addresses reported are only for where the
 * bad pattern was found.
 *
 *
 *************************************************************************************/

__global__ void kernel_test5_init(char* _ptr, char* end_ptr)
{
    unsigned int i;
    unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x * BLOCKSIZE);

    if (ptr >= (unsigned int*) end_ptr) {
	      return;
    }

    unsigned int p1 = 1;

    for (i = 0;i < BLOCKSIZE/sizeof(unsigned int); i+=16){
	      unsigned int p2 = ~p1;

	      ptr[i] = p1;
	      ptr[i+1] = p1;
	      ptr[i+2] = p2;
	      ptr[i+3] = p2;
	      ptr[i+4] = p1;
	      ptr[i+5] = p1;
	      ptr[i+6] = p2;
	      ptr[i+7] = p2;
	      ptr[i+8] = p1;
	      ptr[i+9] = p1;
	      ptr[i+10] = p2;
	      ptr[i+11] = p2;
	      ptr[i+12] = p1;
	      ptr[i+13] = p1;
	      ptr[i+14] = p2;
	      ptr[i+15] = p2;

	      p1 = p1<<1;

	      if (p1 == 0){
	          p1 = 1;
	      }
    }

    return;
}


__global__ void 
kernel_test5_move(char* _ptr, char* end_ptr)
{
    unsigned int i;
    unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x * BLOCKSIZE);

    if (ptr >= (unsigned int*) end_ptr) {
        return;
    }

    unsigned int half_count = BLOCKSIZE/sizeof(unsigned int)/2;
    unsigned int* ptr_mid = ptr + half_count;

    for (i = 0;i < half_count; i++){
	ptr_mid[i] = ptr[i];
    }

    for (i=0;i < half_count - 8; i++){
	ptr[i + 8] = ptr_mid[i];
    }

    for (i=0;i < 8; i++){
	ptr[i] = ptr_mid[half_count - 8 + i];
    }

    return;
}


__global__ void 
kernel_test5_check(char* _ptr, char* end_ptr, unsigned int* ptErrCount, unsigned long* ptFailedAdress,
		   unsigned long* ptExpectedValue, unsigned long* ptCurrentValue, unsigned long* ptValueOfSecondRead)
{
    unsigned int i;
    unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);

    if (ptr >= (unsigned int*) end_ptr) {
	      return;
    }

    for (i=0;i < BLOCKSIZE/sizeof(unsigned int); i+=2){
	if (ptr[i] != ptr[i+1]){
            if((*ptErrCount >= 0) && (*ptErrCount < MAX_ERR_RECORD_COUNT)) {
                  ptFailedAdress[*ptErrCount] = (unsigned long)&ptr[i];        
                  ptExpectedValue[*ptErrCount] = (unsigned long)ptr[i + 1];
                  ptCurrentValue[*ptErrCount++] = (unsigned long)ptr[i];   
            }
	}
    }

    return;
}

/************************************************************************************
 * Test 5 [Block move, 64 moves]
 * This test stresses memory by moving block memories. Memory is initialized
 * with shifting patterns that are inverted every 8 bytes.  Then blocks
 * of memory are moved around.  After the moves
 * are completed the data patterns are checked.  Because the data is checked
 * only after the memory moves are completed it is not possible to know
 * where the error occurred.  The addresses reported are only for where the
 * bad pattern was found.
 *
 *
 *************************************************************************************/

void test5(char* ptr, unsigned int tot_num_blocks)
{

    unsigned int i;
    unsigned int err;
    char* end_ptr = ptr + tot_num_blocks* BLOCKSIZE;
    string msg;

    msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " + "Test 6 [Block move, 64 moves]";
    rvs::lp::Log(msg, rvs::logresults);

    for (i=0;i < tot_num_blocks; i+= GRIDSIZE){
        dim3 grid;

	error_checking("Intializing Test 6 ",  i);
        grid.x= GRIDSIZE;
        hipLaunchKernelGGL(kernel_test5_init,
                            dim3(memdata.blocks), dim3(memdata.threadsPerBlock), 0/*dynamic shared*/, 0/*stream*/,     /* launch config*/
	                           ptr + i*BLOCKSIZE, end_ptr);
        show_progress("Test 6[init]", i, tot_num_blocks);
    }


    for (i=0;i < tot_num_blocks; i+= GRIDSIZE){
        dim3 grid;

        grid.x= GRIDSIZE;
        hipLaunchKernelGGL(kernel_test5_move,
                            dim3(memdata.blocks), dim3(memdata.threadsPerBlock), 0/*dynamic shared*/, 0/*stream*/,     /* launch config*/
	                           ptr + i*BLOCKSIZE, end_ptr);
        show_progress("Test 6[move]", i, tot_num_blocks);
    }


    for (i=0;i < tot_num_blocks; i+= GRIDSIZE){
        dim3 grid;

        grid.x= GRIDSIZE;
        hipLaunchKernelGGL(kernel_test5_check,
                            dim3(memdata.blocks), dim3(memdata.threadsPerBlock), 0/*dynamic shared*/, 0/*stream*/,     /* launch config*/
                            ptr + i*BLOCKSIZE, end_ptr, ptCntOfError, ptFailedAdress, ptExpectedValue, ptCurrentValue, ptValueOfSecondRead);
        err = error_checking("Test 6 checking complete :: ",  i);
	      show_progress("Test 6 [check]", i, tot_num_blocks);
    }

    if(!err) {
      msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " + "Test 6 : PASS";
      rvs::lp::Log(msg, rvs::logresults);
    }

    return;

}

/*****************************************************************************************
 * Test 6 [Moving inversions, 32 bit pat]
 * This is a variation of the moving inversions algorithm that shifts the data
 * pattern left one bit for each successive address. The starting bit position
 * is shifted left for each pass. To use all possible data patterns 32 passes
 * are required.  This test is quite effective at detecting data sensitive
 * errors but the execution time is long.
 *
 ***************************************************************************************/


  __global__ void 
kernel_movinv32_write(char* _ptr, char* end_ptr, unsigned int pattern,
		unsigned int lb, unsigned int sval, unsigned int offset)
{
    unsigned int i;
    unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);

    if (ptr >= (unsigned int*) end_ptr) {
	      return;
    }

    unsigned int k = offset;
    unsigned pat = pattern;

    for (i = 0;i < BLOCKSIZE/sizeof(unsigned int); i++){
	      ptr[i] = pat;
	      k++;

	      if (k >= 32){
	          k=0;
	          pat = lb;
	      }else{
	        pat = pat << 1;
	        pat |= sval;
	      }
    }

    return;
}


__global__ void 
kernel_movinv32_readwrite(char* _ptr, char* end_ptr, unsigned int pattern,
			  unsigned int lb, unsigned int sval, unsigned int offset, unsigned int *ptErrCount,
			  unsigned long *ptFailedAdress, unsigned long *ptExpectedValue, unsigned long *ptCurrentValue, unsigned long *ptValueOfSecondRead)
{
    unsigned int i;
    unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x * BLOCKSIZE);

    if (ptr >= (unsigned int*) end_ptr) {
	    return;
    }

    unsigned int k = offset;
    unsigned pat = pattern;

    for (i = 0;i < BLOCKSIZE/sizeof(unsigned int); i++){
	  if (ptr[i] != pat){
              if((*ptErrCount >= 0) && (*ptErrCount < MAX_ERR_RECORD_COUNT)) {
                   ptFailedAdress[*ptErrCount] = (unsigned long)&ptr[i];        
                   ptExpectedValue[*ptErrCount] = (unsigned long)pat;
                   ptCurrentValue[*ptErrCount++] = (unsigned long)ptr[i];   
              }
	  }

        ptr[i] = ~pat;

        k++;

        if (k >= 32){
             k=0;
             pat = lb;
        }else{
           pat = pat << 1;
           pat |= sval;
        }
    }

    return;
}



__global__ void 
kernel_movinv32_read(char* _ptr, char* end_ptr, unsigned int pattern,
		     unsigned int lb, unsigned int sval, unsigned int offset, unsigned int * ptErrCount,
		     unsigned long* ptFailedAdress, unsigned long* ptExpectedValue, unsigned long* ptCurrentValue, unsigned long* ptValueOfSecondRead)
{
    unsigned int i;
    unsigned int* ptr = (unsigned int*) (_ptr + hipBlockDim_x * BLOCKSIZE);

    if (ptr >= (unsigned int*) end_ptr) {
	      return;
    }

    unsigned int k = offset;
    unsigned pat = pattern;

    for (i = 0;i < BLOCKSIZE/sizeof(unsigned int); i++){
        if (ptr[i] != ~pat){
             if((*ptErrCount >= 0) && (*ptErrCount < MAX_ERR_RECORD_COUNT)) {
                   ptFailedAdress[*ptErrCount] = (unsigned long)&ptr[i];        
                   ptExpectedValue[*ptErrCount] = (unsigned long)~pat;
                   ptCurrentValue[*ptErrCount++] = (unsigned long)ptr[i];   
              }
        }

        k++;

        if (k >= 32){
             k=0;
             pat = lb;
        }else{
            pat = pat << 1;
            pat |= sval;
        }
    }

   return;
}


int movinv32(char* ptr, unsigned int tot_num_blocks, unsigned int pattern,
	 unsigned int lb, unsigned int sval, unsigned int offset)
{

    char* end_ptr = ptr + tot_num_blocks * BLOCKSIZE;
    unsigned int i;
    unsigned int err = 0;

    for (i=0;i < tot_num_blocks; i+= GRIDSIZE){
        dim3 grid;

        grid.x= GRIDSIZE;

        hipLaunchKernelGGL(kernel_movinv32_write,
                                   dim3(memdata.blocks), dim3(memdata.threadsPerBlock), 0/*dynamic shared*/, 0/*stream*/,     /* launch config*/
	                           ptr + i*BLOCKSIZE, end_ptr, pattern, lb,sval, offset); 
        show_progress("\nTest 7[moving inversion 32 write]", i, tot_num_blocks);
    }

    for (i=0;i < tot_num_blocks; i+= GRIDSIZE){
      dim3 grid;

      grid.x= GRIDSIZE;
      hipLaunchKernelGGL(kernel_movinv32_readwrite,
                            dim3(memdata.blocks), dim3(memdata.threadsPerBlock), 0/*dynamic shared*/, 0/*stream*/,     /* launch config*/
                            ptr + i*BLOCKSIZE, end_ptr, pattern, lb,sval, offset, ptCntOfError, ptFailedAdress, ptExpectedValue, ptCurrentValue, ptValueOfSecondRead); 

      err += error_checking("Test 7[movinv32], checking for errors :: ",  i);
      show_progress("\nTest7[moving inversion 32 readwrite]", i, tot_num_blocks);
    }

   for (i=0;i < tot_num_blocks; i+= GRIDSIZE){
       dim3 grid;

       grid.x= GRIDSIZE;
       hipLaunchKernelGGL(kernel_movinv32_read,
                            dim3(memdata.blocks), dim3(memdata.threadsPerBlock), 0/*dynamic shared*/, 0/*stream*/,     /* launch config*/
                             ptr + i*BLOCKSIZE, end_ptr, pattern, lb,sval, offset, ptCntOfError, ptFailedAdress, ptExpectedValue, ptCurrentValue, ptValueOfSecondRead); 
       err += error_checking("Test 7 [movinv32]",  i);
       show_progress("\nTest 7[moving inversion 32 read]", i, tot_num_blocks);
   }

   return err;

}

void test6(char* ptr, unsigned int tot_num_blocks)
{
    unsigned int i;
    unsigned int err= 0;
    unsigned int pattern;
    std::string  msg;

    msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " + "Test 7 [Moving inversions, 32 bit pat]";
    rvs::lp::Log(msg, rvs::logresults);

    for (i= 0, pattern = 1;i < 32; pattern = pattern << 1, i++){

         err += movinv32(ptr, tot_num_blocks, pattern, 1, 0, i);

	 err += movinv32(ptr, tot_num_blocks, ~pattern, 0xfffffffe, 1, i);
    }
    if(!err) {
       msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " + "Test 7 : PASS";
       rvs::lp::Log(msg, rvs::logresults);
    }
}

/******************************************************************************
 * Test 7 [Random number sequence]
 *
 * This test writes a series of random numbers into memory.  A block (1 MB) of memory
 * is initialized with random patterns. These patterns and their complements are
 * used in moving inversions test with rest of memory.
 *
 *
 *******************************************************************************/

  __global__ void 
kernel_test7_write(char* _ptr, char* end_ptr, char* _start_ptr, unsigned int* err)
{
    unsigned int i;
    unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x * BLOCKSIZE);
    unsigned int* start_ptr = (unsigned int*) _start_ptr;

    if (ptr >= (unsigned int*) end_ptr) {
	      return;
    }

    for (i = 0;i < BLOCKSIZE/sizeof(unsigned int); i++){
	      ptr[i] = start_ptr[i];
    }

    return;
}



__global__ void 
kernel_test7_readwrite(char* _ptr, char* end_ptr, char* _start_ptr, unsigned int* ptErrCount,
		       unsigned long* ptFailedAdress, unsigned long* ptExpectedValue, unsigned long* ptCurrentValue, unsigned long* ptValueOfSecondRead)
{
    unsigned int i;
    unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x * BLOCKSIZE);
    unsigned int* start_ptr = (unsigned int*) _start_ptr;

    if (ptr >= (unsigned int*) end_ptr) {
	      return;
    }


    for (i = 0;i < BLOCKSIZE/sizeof(unsigned int); i++){
	 if (ptr[i] != start_ptr[i]){
               if((*ptErrCount >= 0) && (*ptErrCount < MAX_ERR_RECORD_COUNT)) {
                     ptFailedAdress[*ptErrCount] = (unsigned long)&ptr[i];        
                     ptExpectedValue[*ptErrCount] = (unsigned long)start_ptr[i];
                     ptCurrentValue[*ptErrCount++] = (unsigned long)ptr[i];   
               }
	 }

	 ptr[i] = ~(start_ptr[i]);
    }

    return;
}

__global__ void 
kernel_test7_read(char* _ptr, char* end_ptr, char* _start_ptr, unsigned int* ptErrCount, unsigned long* ptFailedAdress,
		  unsigned long* ptExpectedValue, unsigned long* ptCurrentValue, unsigned long* ptValueOfSecondRead)
{
    unsigned int i;
    unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x  * BLOCKSIZE);
    unsigned int* start_ptr = (unsigned int*) _start_ptr;

    if (ptr >= (unsigned int*) end_ptr) {
	      return;
    }


    for (i = 0;i < BLOCKSIZE/sizeof(unsigned int); i++){
	      if (ptr[i] != ~(start_ptr[i])){
                   if((*ptErrCount >= 0) && (*ptErrCount < MAX_ERR_RECORD_COUNT)) {
                          ptFailedAdress[*ptErrCount] = (unsigned long)&ptr[i];        
                          ptExpectedValue[*ptErrCount] = (unsigned long)~start_ptr[i];
                          ptCurrentValue[*ptErrCount++] = (unsigned long)ptr[i];   
                    }
	      }
    }

    return;
}


void test7(char* ptr, unsigned int tot_num_blocks)
{

    unsigned int* host_buf = (unsigned int*)malloc(BLOCKSIZE);
    unsigned int err = 0;
    unsigned int i;
    unsigned int iteration = 0;
    std::string   msg;

    msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " + "Test 8 [Random number sequence]";
    rvs::lp::Log(msg, rvs::logresults);

    for (i = 0;i < BLOCKSIZE/sizeof(unsigned int);i++){
      	host_buf[i] = get_random_num();
    }

    HIP_CHECK(hipMemcpy(ptr, host_buf, BLOCKSIZE, hipMemcpyHostToDevice));

    char* end_ptr = ptr + tot_num_blocks* BLOCKSIZE;

    repeat:

        for (i=1;i < tot_num_blocks; i+= GRIDSIZE){
	        dim3 grid;

	        grid.x= GRIDSIZE;
          hipLaunchKernelGGL(kernel_test7_write,
                            dim3(memdata.blocks), dim3(memdata.threadsPerBlock), 0/*dynamic shared*/, 0/*stream*/,     /* launch config*/
	                                        ptr + i* BLOCKSIZE, end_ptr, ptr, ptCntOfError); 
          show_progress("test8_write", i, tot_num_blocks);
        }


        for (i=1;i < tot_num_blocks; i+= GRIDSIZE){
	        dim3 grid;

	        grid.x= GRIDSIZE;
          hipLaunchKernelGGL(kernel_test7_readwrite,
                            dim3(memdata.blocks), dim3(memdata.threadsPerBlock), 0/*dynamic shared*/, 0/*stream*/,     /* launch config*/
	                            ptr + i*BLOCKSIZE, end_ptr, ptr, ptCntOfError, ptFailedAdress, ptExpectedValue, ptCurrentValue, ptValueOfSecondRead);
	        err += error_checking("test8_readwrite",  i);
          show_progress("test8_readwrite", i, tot_num_blocks);
        }


        for (i=1;i < tot_num_blocks; i+= GRIDSIZE){
	          dim3 grid;

	          grid.x= GRIDSIZE;
            hipLaunchKernelGGL(kernel_test7_read,
                                 dim3(memdata.blocks), dim3(memdata.threadsPerBlock), 0/*dynamic shared*/, 0/*stream*/,     /* launch config*/
	                               ptr + i*BLOCKSIZE, end_ptr, ptr, ptCntOfError, ptFailedAdress, ptExpectedValue, ptCurrentValue, ptValueOfSecondRead); 
	          err += error_checking("test8_read",  i);
            show_progress("test8_read", i, tot_num_blocks); 
        }


        if (err == 0 && iteration == 0){
            msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " + "TEST 8 : PASS \n no errors detected, iterations are zero here";
            rvs::lp::Log(msg, rvs::logresults);
	          return;
        }

        if (iteration <  memdata.num_iterations){
            msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " + "repeating Test 8 because there are" + std::to_string(err) + " errors found in last run";
            rvs::lp::Log(msg, rvs::loginfo);
	          iteration++;
	          err = 0;
	          goto repeat;
        }

        if(!err) {
            msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " + "Test 8 : PASS";
            rvs::lp::Log(msg, rvs::logresults);
        }
}
/***********************************************************************************
 * Test 8 [Modulo 20, random pattern]
 *
 * A random pattern is generated. This pattern is used to set every 20th memory location
 * in memory. The rest of the memory location is set to the complimemnt of the pattern.
 * Repeat this for 20 times and each time the memory location to set the pattern is shifted right.
 *
 *
 **********************************************************************************/

__global__ void 
kernel_modtest_write(char* _ptr, char* end_ptr, unsigned int offset, unsigned int p1, unsigned int p2)
{
    unsigned int i;
    unsigned int* ptr = (unsigned int*) (_ptr + hipBlockDim_x * BLOCKSIZE);

    if (ptr >= (unsigned int*) end_ptr) {
       return;
    }

    for (i = offset;i < BLOCKSIZE/sizeof(unsigned int); i+=MOD_SZ){
        ptr[i] =p1;
    }

    for (i = 0;i < BLOCKSIZE/sizeof(unsigned int); i++){
      if (i % MOD_SZ != offset){
          ptr[i] =p2;
      }
    }

    return;
}


__global__ void 
kernel_modtest_read(char* _ptr, char* end_ptr, unsigned int offset, unsigned int p1, unsigned int* ptErrCount,
		    unsigned long* ptFailedAdress, unsigned long* ptExpectedValue, unsigned long* ptCurrentValue, unsigned long* ptValueOfSecondRead)
{
    unsigned int i;
    unsigned int* ptr = (unsigned int*) (_ptr + hipBlockDim_x * BLOCKSIZE);

    if (ptr >= (unsigned int*) end_ptr) {
	      return;
    }

    for (i = offset;i < BLOCKSIZE/sizeof(unsigned int); i+=MOD_SZ){
       if (ptr[i] !=p1){
            if((*ptErrCount >= 0) && (*ptErrCount < MAX_ERR_RECORD_COUNT)) {
                   ptFailedAdress[*ptErrCount] = (unsigned long)&ptr[i];        
                   ptExpectedValue[*ptErrCount] = (unsigned long)p1;
                   ptCurrentValue[*ptErrCount++] = (unsigned long)ptr[i];   
            }
       }
    }

    return;
}

unsigned int modtest(char* ptr, unsigned int tot_num_blocks, unsigned int offset, unsigned int p1, unsigned int p2)
{

    char* end_ptr = ptr + tot_num_blocks* BLOCKSIZE;
    unsigned int i;
    unsigned int err = 0;

    for (i= 0;i < tot_num_blocks; i+= GRIDSIZE){
          dim3 grid;

          grid.x= GRIDSIZE;
          hipLaunchKernelGGL(kernel_modtest_write,
                         dim3(memdata.blocks), dim3(memdata.threadsPerBlock), 0/*dynamic shared*/, 0/*stream*/,     /* launch config*/
                         ptr + i*BLOCKSIZE, end_ptr, offset, p1, p2); 
          show_progress("test9[mod test, write]", i, tot_num_blocks);
    }

    for (i= 0;i < tot_num_blocks; i+= GRIDSIZE){
         dim3 grid;

         grid.x= GRIDSIZE;
         hipLaunchKernelGGL(kernel_modtest_read,
                         dim3(memdata.blocks), dim3(memdata.threadsPerBlock), 0/*dynamic shared*/, 0/*stream*/,     /* launch config*/
                         ptr + i*BLOCKSIZE, end_ptr, offset, p1, ptCntOfError, ptFailedAdress, ptExpectedValue, ptCurrentValue, ptValueOfSecondRead); 
         err += error_checking("test9[mod test, read", i);
         show_progress("test9[mod test, read]", i, tot_num_blocks);
    }

    return err;

}

void test8(char* ptr, unsigned int tot_num_blocks)
{
    unsigned int i;
    unsigned int err = 0;
    unsigned int iteration = 0;
    unsigned int p1;
    std::string msg;

    msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " + " Test 9 [Modulo 20, random pattern]";
    rvs::lp::Log(msg, rvs::logresults);

    if (memdata.global_pattern){
	    p1 = memdata.global_pattern;
    }else{
	    p1= get_random_num();
    }

    unsigned int p2 = ~p1;

    msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " + " Pattern  p1 " + std::to_string(p1) + "pattern  p2 " + std::to_string(p2);
    rvs::lp::Log(msg, rvs::loginfo);
 repeat:
    for (i = 0;i < MOD_SZ; i++){
	    err += modtest(ptr, tot_num_blocks,i, p1, p2);
    }
    if (err == 0 && iteration == 0){
	    msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " + "Test 9 : PASS \n" +
		    "no errors detected, iterations are zero here";
       rvs::lp::Log(msg, rvs::logresults);
	      return;
    }
    if (iteration < memdata.num_iterations){

        msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " + std::to_string(iteration) + 
          "th repeating Test 9 because there are " + std::to_string(err) + "errors found in last run, p1= " 
          + std::to_string(p1) + " p2= " + std::to_string(p2) + "\n";
        rvs::lp::Log(msg, rvs::loginfo);

	      iteration++;
	      err = 0;
	      goto repeat;
    }
    if(!err) {
       msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " + "Test 9 : PASS";
       rvs::lp::Log(msg, rvs::logresults);
    }
}

/************************************************************************************
 *
 * Test 9 [Bit fade test, 90 min, 2 patterns]
 * The bit fade test initializes all of memory with a pattern and then
 * sleeps for 90 minutes. Then memory is examined to see if any memory bits
 * have changed. All ones and all zero patterns are used. This test takes
 * 3 hours to complete.  The Bit Fade test is disabled by default
 *
 **********************************************************************************/

void test9(char* ptr, unsigned int tot_num_blocks)
{

    unsigned int p1 = 0;
    unsigned int p2 = ~p1;
    unsigned int err = 0;
    std::string  msg;

    unsigned int i;
    char* end_ptr = ptr + tot_num_blocks* BLOCKSIZE;

    msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " + "Test 10 [Bit fade test, 90 min, 2 patterns]";
    rvs::lp::Log(msg, rvs::logresults);

    for (i= 0;i < tot_num_blocks; i+= GRIDSIZE){
        dim3 grid;

        grid.x= GRIDSIZE;
        hipLaunchKernelGGL(kernel_move_inv_write,
                               dim3(memdata.blocks), dim3(memdata.threadsPerBlock), 0/*dynamic shared*/, 0/*stream*/,     /* launch config*/
                               ptr + i*BLOCKSIZE, end_ptr, p1); 
        show_progress("test 10[bit fade test, write]: ", i, tot_num_blocks);
    }

    //sleep(60*90);
    std::this_thread::sleep_for(std::chrono::milliseconds(10000));

    for (i=0;i < tot_num_blocks; i+= GRIDSIZE){
             dim3 grid;

             grid.x= GRIDSIZE;
             hipLaunchKernelGGL(kernel_move_inv_readwrite,
                               dim3(memdata.blocks), dim3(memdata.threadsPerBlock), 0/*dynamic shared*/, 0/*stream*/,     /* launch config*/
                               ptr + i*BLOCKSIZE, end_ptr, p1, p2, ptCntOfError, ptFailedAdress, ptExpectedValue, ptCurrentValue, ptValueOfSecondRead); 
	    err += error_checking("test 10[bit fade test, readwrite] :",  i);
            show_progress("test 10[bit fade test, readwrite] : ", i, tot_num_blocks);
    }

    //sleep(60*90);
    std::this_thread::sleep_for(std::chrono::milliseconds(10000));

    for (i=0;i < tot_num_blocks; i+= GRIDSIZE){
           dim3 grid;
           grid.x= GRIDSIZE;

            hipLaunchKernelGGL(kernel_move_inv_read,
                                 dim3(memdata.blocks), dim3(memdata.threadsPerBlock), 0/*dynamic shared*/, 0/*stream*/,     /* launch config*/
	                          ptr + i*BLOCKSIZE, end_ptr, p2, ptCntOfError, ptFailedAdress, ptExpectedValue, ptCurrentValue, ptValueOfSecondRead); 
	    err += error_checking("test 10[bit fade test, read] : ",  i);
            show_progress("test 10[bit fade test, read] : ", i, tot_num_blocks);
    }

    if(!err) {
       msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " + "Test 10 : PASS"; 
       rvs::lp::Log(msg, rvs::logresults);
    }

    return;
}

/**************************************************************************************
 * Test10 [memory stress test]
 *
 * Stress memory as much as we can. A random pattern is generated and a kernel of large grid size
 * and block size is launched to set all memory to the pattern. A new read and write kernel is launched
 * immediately after the previous write kernel to check if there is any errors in memory and set the
 * memory to the compliment. This process is repeated for 1000 times for one pattern. The kernel is
 * written as to achieve the maximum bandwidth between the global memory and GPU.
 * This will increase the chance of catching software error. In practice, we found this test quite useful
 * to flush hardware errors as well.
 *
 */

__global__ void  
test10_kernel_write(char* ptr, int memsize, TYPE p1)
{
    int i;
    int avenumber = memsize/(hipGridDim_x * hipGridDim_y);
    TYPE* mybuf = (TYPE*)(ptr + blockIdx.x* avenumber);
    int n = avenumber/(hipBlockDim_x * sizeof(TYPE));

    for(i=0;i < n;i++){
        int index = i* hipBlockDim_x + threadIdx.x;
        mybuf[index]= p1;
    }
    int index = n * hipBlockDim_x + threadIdx.x;
    if (index*sizeof(TYPE) < avenumber){
        mybuf[index] = p1;
    }

    return;
}

__global__ void  
test10_kernel_readwrite(char* ptr, int memsize, TYPE p1, TYPE p2,  unsigned int* ptErrCount,
					unsigned long* ptFailedAdress, unsigned long* ptExpectedValue, unsigned long* ptCurrentValue, unsigned long* ptValueOfSecondRead)
{
    int   avenumber   = memsize/(gridDim.x*gridDim.y);
    TYPE* mybuf       = (TYPE*)(ptr +  blockIdx.x * avenumber);
    int   n           = avenumber/( blockDim.x * sizeof(TYPE));
    TYPE  localp;
    int   i;

    for(i=0; i < n; i++ ){
        int index = i * blockDim.x  + threadIdx.x;

        localp = mybuf[index];
        if (localp != p1){
            if((*ptErrCount >= 0) && (*ptErrCount < MAX_ERR_RECORD_COUNT)) {
                  ptFailedAdress[*ptErrCount] = (unsigned long)&mybuf[index];
                  ptExpectedValue[*ptErrCount] = (unsigned long)p1;
                  ptCurrentValue[*ptErrCount++] = (unsigned long)localp;
             }
        }

	mybuf[index] = p2;
    }

    int index = n * blockDim.x + threadIdx.x;

    if (index*sizeof(TYPE) < avenumber){
	      localp = mybuf[index];

	      if (localp!= p1){
                  if((*ptErrCount >= 0) && (*ptErrCount < MAX_ERR_RECORD_COUNT)) {
                        ptFailedAdress[*ptErrCount] = (unsigned long)&mybuf[index];
                        ptExpectedValue[*ptErrCount] = (unsigned long)p1;
                        ptCurrentValue[*ptErrCount++] = (unsigned long)localp;
                  }
	      }
	      mybuf[index] = p2;
    }

    return;
}

void test10(char* ptr, unsigned int tot_num_blocks)
{
    unsigned int err = 0;
    TYPE    p1;
    std::string msg;;

    msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " + "Test 11 [memory stress test]";
    rvs::lp::Log(msg, rvs::logresults);

    if (memdata.global_pattern_long){
	      p1 = memdata.global_pattern_long;
    }else{
	      p1 = get_random_num_long();
    }

    TYPE p2 = ~p1;

    hipStream_t stream;
    hipEvent_t start, stop;

    msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " + " Test 11 with pattern :" + std::to_string(p1);
    rvs::lp::Log(msg, rvs::loginfo);


    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    int n = memdata.num_iterations;
    float elapsedtime;

    msg = "[" + memdata.action_name + "] " + MODULE_NAME + " Total number of blocks :" + std::to_string(tot_num_blocks) 
                  + " Number of iterations :" + std::to_string(n);
    rvs::lp::Log(msg, rvs::logtrace);

    dim3 gridDim(STRESS_GRIDSIZE);
    dim3 blockDim(STRESS_BLOCKSIZE);
    HIP_CHECK(hipEventRecord(start, stream));

    hipLaunchKernelGGL(test10_kernel_write,
                         gridDim, blockDim, 0/*dynamic shared*/, stream,     /* launch config*/
                          ptr, tot_num_blocks*BLOCKSIZE, p1); 

    for(unsigned long i =0;i < n ;i ++){
        hipLaunchKernelGGL(test10_kernel_readwrite,
                                gridDim, blockDim, 0/*dynamic shared*/, stream,     /* launch config*/
	                        ptr, tot_num_blocks*BLOCKSIZE, p1, p2,
			        ptCntOfError, ptFailedAdress, ptExpectedValue, ptCurrentValue, ptValueOfSecondRead); 
	        p1 = ~p1;
	        p2 = ~p2;
    }

    hipEventRecord(stop, stream);
    hipEventSynchronize(stop);

    err += error_checking("test11[Memory stress test]",  0);
    hipEventElapsedTime(&elapsedtime, start, stop);
    msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " + "Test 11: elapsedtime = " 
      + std::to_string(elapsedtime) + " bandwidth = " + std::to_string((2*n+1)*tot_num_blocks/elapsedtime) + "GB/s ";
    rvs::lp::Log(msg, rvs::logresults);

    hipEventDestroy(start);
    hipEventDestroy(stop);

    hipStreamDestroy(stream);

    if(!err) {
       msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " + "Test 11 : PASS ";
       rvs::lp::Log(msg, rvs::logresults);
    }
}

void allocate_small_mem(void)
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

    HIP_CHECK(hipMalloc((void**)&ptValueOfSecondRead, sizeof(unsigned long) * MAX_ERR_RECORD_COUNT));
    HIP_CHECK(hipMemset(ptValueOfSecondRead, 0, sizeof(unsigned long) * MAX_ERR_RECORD_COUNT));
}

void free_small_mem(void)
{
    //Initialize memory
    hipFree((void*)ptCntOfError);

    hipFree((void*)ptFailedAdress);

    hipFree((void*)ptExpectedValue);

    hipFree((void*)ptCurrentValue);

    hipFree((void*)ptValueOfSecondRead);
}
