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

#include "rvstimer.h"

#define NANOSECONDS_PER_SECOND 1000000000

PerfTimer::PerfTimer() {
  freq_in_100mhz = MeasureTSCFreqHz();
}

PerfTimer::~PerfTimer() {
  while (!_timers.empty()) {
    Timer *temp = _timers.back();
    _timers.pop_back();
    delete temp;
  }
}

// Create a new timer instance and return its index
int PerfTimer::CreateTimer() {

  Timer *newTimer = new Timer;
  newTimer->_start = 0.0;
  newTimer->_clocks = 0.0;

  #ifdef  __linux__
  newTimer->_freq = NANOSECONDS_PER_SECOND;
  #endif

  // Save the timer object in timer list
  _timers.push_back(newTimer);
  return (int)(_timers.size() - 1);
}

int PerfTimer::StartTimer(int index) {

  if (index >= (int)_timers.size()) {
    Error("Cannot reset timer. Invalid handle.");
    return HSA_FAILURE;
  }

  #ifdef  __linux__
    // General Linux timing method
    #ifndef _AMD
      struct timespec s;
      clock_gettime(CLOCK_MONOTONIC, &s);
      _timers[index]->_start =
      (long long)s.tv_sec * NANOSECONDS_PER_SECOND + (long long)s.tv_nsec;
    // AMD Linux timing method
    #else
      unsigned int unused;
    _timers[index]->_start = __rdtscp(&unused);
    #endif
  #endif

  return HSA_SUCCESS;
}

int PerfTimer::StopTimer(int index) {

  long long n = 0;
  if (index >= (int)_timers.size()) {
    Error("Cannot reset timer. Invalid handle.");
    return HSA_FAILURE;
  }
  
  #ifdef  __linux__
    // General Linux timing method
    #ifndef _AMD
      struct timespec s;
      clock_gettime(CLOCK_MONOTONIC, &s);
      n = (long long)s.tv_sec * NANOSECONDS_PER_SECOND + (long long)s.tv_nsec;
    // AMD Linux timing
    #else
      unsigned int unused;
      n = __rdtscp(&unused);
    #endif
  #endif

  n -= _timers[index]->_start;
  _timers[index]->_start = 0;

  #ifndef _AMD
    _timers[index]->_clocks += n;
  #endif

  #ifdef  __linux__
    //_timers[index]->_clocks += 10 * n /freq_in_100mhz;      // unit is ns
    _timers[index]->_clocks += 1.0E-6 * 10 * n / freq_in_100mhz;  // convert to ms
    // cout << "_AMD is enabled!!!" << endl;
  #endif

  return HSA_SUCCESS;
}

void PerfTimer::Error(string str) { cout << str << endl; }

double PerfTimer::ReadTimer(int index) {

  if (index >= (int)_timers.size()) {
    Error("Cannot read timer. Invalid handle.");
    return HSA_FAILURE;
  }

  double reading = double(_timers[index]->_clocks);

  reading = double(reading / _timers[index]->_freq);

  return reading;
}

void PerfTimer::ResetTimer(int index) {
  
  // Check if index value is over the timer's size
  if (index >= (int)_timers.size()) {
    Error("Invalid index value\n");
    exit(1);
  }

  _timers[index]->_clocks = 0.0;
  _timers[index]->_start = 0.0;
}

uint64_t PerfTimer::CoarseTimestampUs() {
  
  #ifdef  __linux__
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return uint64_t(ts.tv_sec) * 1000000 + ts.tv_nsec / 1000;
  #endif
}

uint64_t PerfTimer::MeasureTSCFreqHz() {
  
  // Make a coarse interval measurement of TSC ticks for 1 gigacycles.
  unsigned int unused;
  uint64_t tscTicksEnd;

  uint64_t coarseBeginUs = CoarseTimestampUs();
  uint64_t tscTicksBegin = __rdtscp(&unused);
  do {
    tscTicksEnd = __rdtscp(&unused);
  } while (tscTicksEnd - tscTicksBegin < 1000000000);

  uint64_t coarseEndUs = CoarseTimestampUs();

  // Compute the TSC frequency and round to nearest 100MHz.
  uint64_t coarseIntervalNs = (coarseEndUs - coarseBeginUs) * 1000;
  uint64_t tscIntervalTicks = tscTicksEnd - tscTicksBegin;
  return (tscIntervalTicks * 10 + (coarseIntervalNs / 2)) / coarseIntervalNs;
}
