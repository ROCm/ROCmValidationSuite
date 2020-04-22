/*
 * Illinois Open Source License
 *
 * University of Illinois/NCSA
 * Open Source License
 *
 * Copyright 2009,    University of Illinois.  All rights reserved.
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

#ifndef __RVS_MEMTEST_H__
#define __RVS_MEMTEST_H__

#include <iostream>
#include <sstream>

#include "include/rvsthreadbase.h"
#include <pthread.h>

#define TDIFF(tb, ta) (tb.tv_sec - ta.tv_sec + 0.000001*(tb.tv_usec - ta.tv_usec))
#define DIM(x) (sizeof(x)/sizeof(x[0]))
#define MIN(x,y) (x < y? x: y)
#define MOD_SZ 20
#define MAILFILE "/bin/mail"
#define MAX_STR_LEN 256
#define ERR_BAD_STATE  -1
#define ERR_GENERAL -999


#define KNRM "\x1B[0m"
#define KRED "\x1B[31m"
#define KGRN "\x1B[32m"
#define KYEL "\x1B[33m"
#define KBLU "\x1B[34m"
#define KMAG "\x1B[35m"
#define KCYN "\x1B[36m"
#define KWHT "\x1B[37m"

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t atomic_mutex = PTHREAD_MUTEX_INITIALIZER;

#define passed()                                                                                   \
    printf("%sPASSED!%s\n", KGRN, KNRM);                                                           \
    exit(0);

#define failed(...)                                                                                \
    printf("%serror: ", KRED);                                                                     \
    printf(__VA_ARGS__);                                                                           \
    printf("\n");                                                                                  \
    printf("error: TEST FAILED\n%s", KNRM);                                                        \
    abort();

#define warn(...)                                                                                  \
    printf("%swarn: ", KYEL);                                                                      \
    printf(__VA_ARGS__);                                                                           \
    printf("\n");                                                                                  \
    printf("warn: TEST WARNING\n%s", KNRM);

#define MAX_GPU_NUM  4
#define BLOCKSIZE ((unsigned long)(1024*1024))
#define GRIDSIZE 128


typedef  void (*test_func_t)(char* , unsigned int );

typedef struct rvs_memtest_s{
    test_func_t func;
    const char* desc;
    unsigned int enabled;
}rvs_memtest_t;

   unsigned int num_passes; 
   volatile int gpu_temp[MAX_GPU_NUM];
   unsigned int verbose;
   unsigned int interactive;
   char hostname[64];
   unsigned int monitor_temp; 
   unsigned int global_pattern;
   unsigned int global_pattern_long;
   uint64_t     blocks;
   uint64_t     threadsPerBlock;
   uint64_t        num_iterations;
   bool            useMappedMemory;
   void*           mappedHostPtr;
   unsigned long   gpu_idx;
   unsigned long   devSerialNum;
   std::mutex      mtx_mem_test;
   unsigned int    *ptCntOfError;
   unsigned long   *ptFailedAdress;
   unsigned long   *expectedValue;
   unsigned long   *ptCurrentValue;
   unsigned long   *ptValueOfStartAddr;
   unsigned long   *ptDebugValue;
   unsigned int    tot_num_blocks;
   unsigned int    max_num_blocks;
   unsigned int    exit_on_error;

   char* time_string(void);
   void  atomic_inc(unsigned int* value);
   void  prepare_rvsMemTest();
   void  list_tests_info(void);
   void  allocate_small_mem(void);
   void  free_small_mem(void);
   void  movinv32(char* ptr, unsigned int tot_num_blocks, unsigned int pattern,
   unsigned int lb, unsigned int sval, unsigned int offset, unsigned int p1, unsigned int p2);
   uint64_t get_random_num_long(void);
   unsigned int atomic_read(unsigned int* value);
   unsigned int get_random_num(void);

   unsigned int  move_inv_test(char* ptr, unsigned int tot_num_blocks, unsigned int p1, unsigned p2);
   unsigned int error_checking(const char* msg, unsigned int blockidx);
   unsigned int modtest(char* ptr, unsigned int tot_num_blocks, unsigned int offset, unsigned int p1, unsigned int p2);

   void kernel_test0_global_write(char* _ptr, char* _end_ptr);

   void test0(char* ptr, unsigned int tot_num_blocks);
   void test1(char* ptr, unsigned int tot_num_blocks);
   void test2(char* ptr, unsigned int tot_num_blocks);
   void test3(char* ptr, unsigned int tot_num_blocks);
   void test4(char* ptr, unsigned int tot_num_blocks);
   void test5(char* ptr, unsigned int tot_num_blocks);
   void test6(char* ptr, unsigned int tot_num_blocks);
   void test7(char* ptr, unsigned int tot_num_blocks);
   void test8(char* ptr, unsigned int tot_num_blocks);
   void test9(char* ptr, unsigned int tot_num_blocks);
   void test10(char* ptr, unsigned int tot_num_blocks);

   void usage(char** argv); 
   void rvs_memtest();
   void run_tests(char* ptr, unsigned int tot_num_blocks);


#endif
