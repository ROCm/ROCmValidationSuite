/********************************************************************************
 *
 * Copyright (c) 2018-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef RVS_TEST_EXT_TIMER_H_
#define RVS_TEST_EXT_TIMER_H_

#include "include/rvstimer.h"

template<class T>
class ext_timer : public rvs::timer<T> {
 public:
  // construtor
  ext_timer(void(T::*cbFunc)(), T* cbArg): rvs::timer<T>::timer(cbFunc, cbArg) {
  }
  // return used class
  T* get_cbarg() {
    return *rvs::timer<T>::cbarg;
  }
  // return brun
  // NOTE: true for the duration of timer activity
  bool get_brun() {
    return rvs::timer<T>::brun;
  }
  // return brunonce
  // NOTE: true if timer is to fire only once
  bool get_brunonce() {
    return rvs::timer<T>::brunonce;
  }
  // return timeset
  // NOTE: timer interval (ms)
  int get_timeset() {
    return rvs::timer<T>::timeset;
  }
};

#endif  // RVS_TEST_EXT_TIMER_H_

