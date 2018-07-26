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

#ifndef ROCM_VALIDATION_SUITE_MYTIME_H_
#define ROCM_VALIDATION_SUITE_MYTIME_H_

// Will use AMD timer and general Linux timer based on users'
// need --> compilation flag. Support for windows platform is
// not currently available

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <x86intrin.h>
#include <string.h>

#include <iostream>
#include <vector>
#include <string>

using namespace std;

#include <sys/time.h>

#define HSA_FAILURE 1
#define HSA_SUCCESS 0

class PerfTimer {

 private:

  struct Timer {
    string name;       /* < name name of time object*/
    long long _freq;   /* < _freq frequency*/
    long long _clocks; /* < _clocks number of ticks at end*/
    long long _start;  /* < _start start point ticks*/
  };

  std::vector<Timer*> _timers; /*< _timers vector to Timer objects */
  double freq_in_100mhz;

 public:

  PerfTimer();
  ~PerfTimer();

 private:

  // AMD timing method
  uint64_t CoarseTimestampUs();
  uint64_t MeasureTSCFreqHz();

  // General Linux timing method

 public:
  
  int CreateTimer();
  int StartTimer(int index);
  int StopTimer(int index);
  void ResetTimer(int index);

 public:
 
  // retrieve time
  double ReadTimer(int index);
  
  // write into a file
  double WriteTimer(int index);

 public:
  void Error(string str);
};

#endif    //  ROCM_VALIDATION_SUITE_MYTIME_H_
