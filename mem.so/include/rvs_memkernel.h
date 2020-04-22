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

#ifndef __RVS_MEMKERN_H__
#define __RVS_MEMKERN_H__

#include <iostream>
#include "include/rvsthreadbase.h"

 #define TYPE unsigned long

__global__ void kernel_move_inv_write(char* _ptr, char* end_ptr, unsigned int pattern);

__global__ void kernel_move_inv_readwrite(char* _ptr, char* end_ptr, unsigned int p1, unsigned int p2, unsigned int* err,
  unsigned long* err_addr, unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read);

__global__ void kernel_move_inv_read(char* _ptr, char* end_ptr,  unsigned int pattern, unsigned int* err,
   unsigned long* err_addr, unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read );

__global__ void kernel_test0_global_read(char* _ptr, char* _end_ptr, unsigned int* err, unsigned long* err_addr,
 unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read);

__global__ void kernel_test0_write(char* _ptr, char* end_ptr);

__global__ void kernel_test0_read(char* _ptr, char* end_ptr, unsigned int* err, unsigned long* err_addr,
unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read);

void kernel_test1_write(char* _ptr, char* end_ptr, unsigned int* err);

void kernel_test1_read(char* _ptr, char* end_ptr, unsigned int* err, unsigned long* err_addr,
unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read);

void kernel_test5_init(char* _ptr, char* end_ptr);

void kernel_test5_move(char* _ptr, char* end_ptr);

void kernel_test5_check(char* _ptr, char* end_ptr, unsigned int* err, unsigned long* err_addr,
 unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read);

void kernel_movinv32_write(char* _ptr, char* end_ptr, unsigned int pattern,
signed int lb, unsigned int sval, unsigned int offset);

void kernel_movinv32_readwrite(char* _ptr, char* end_ptr, unsigned int pattern,
  unsigned int lb, unsigned int sval, unsigned int offset, unsigned int * err,
  unsigned long* err_addr, unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read);

void kernel_movinv32_read(char* _ptr, char* end_ptr, unsigned int pattern,
   unsigned int lb, unsigned int sval, unsigned int offset, unsigned int * err,
   unsigned long* err_addr, unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read);

void kernel_test7_write(char* _ptr, char* end_ptr, char* _start_ptr, unsigned int* err);

void kernel_test7_readwrite(char* _ptr, char* end_ptr, char* _start_ptr, unsigned int* err,
     unsigned long* err_addr, unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read);


void kernel_test7_read(char* _ptr, char* end_ptr, char* _start_ptr, unsigned int* err, unsigned long* err_addr,
unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read);


void kernel_modtest_write(char* _ptr, char* end_ptr, unsigned int offset, unsigned int p1, unsigned int p2);


void kernel_modtest_read(char* _ptr, char* end_ptr, unsigned int offset, unsigned int p1, unsigned int* err,
  unsigned long* err_addr, unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read);

void  test10_kernel_write(char* ptr, int memsize, TYPE p1);

void  test10_kernel_readwrite(char* ptr, int memsize, TYPE p1, TYPE p2,  unsigned int* err,
    unsigned long* err_addr, unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read);


#endif
