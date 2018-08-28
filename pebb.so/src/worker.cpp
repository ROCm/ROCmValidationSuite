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
#include "worker.h"

#include <chrono>
#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>

#ifdef __cplusplus
extern "C" {
#endif
#include <pci/pci.h>
#include <linux/pci.h>
#ifdef __cplusplus
}
#endif

#include "rvs_module.h"
#include "pci_caps.h"
#include "gpu_util.h"
#include "rvsloglp.h"

using std::string;
using std::vector;
using std::map;

Worker::Worker() {
  bfiltergpu = false;
}
Worker::~Worker() {}

/**
 * @brief Sets GPU IDs for filtering
 * @arg GpuIds Array of GPU GpuIds
 */
void Worker::set_gpuids(const std::vector<int>& GpuIds) {
  gpuids = GpuIds;
  bfiltergpu = true;
}

/**
 * @brief Thread function
 *
 * Loops while brun == TRUE and performs polled monitoring avery 1msec.
 *
 * */
void Worker::run() {
  brun = true;

  unsigned int sec;
  unsigned int usec;
  void* r;

  // get timestamp
  rvs::lp::get_ticks(&sec, &usec);

  // add string output
  string msg("[" + action_name + "] pebb " + strgpuids + " started");
  rvs::lp::Log(msg, rvs::logresults, sec, usec);
  // get timestamp
  rvs::lp::get_ticks(&sec, &usec);

  // add JSON output
  r = rvs::lp::LogRecordCreate("pebb", action_name.c_str(), rvs::logresults,
                               sec, usec);
  rvs::lp::AddString(r, "msg", "started");
  rvs::lp::AddString(r, "device", strgpuids);
  rvs::lp::LogRecordFlush(r);

  // worker thread has started
  while (brun) {
    rvs::lp::Log("[" + action_name + "] pebb worker thread is running...",
                 rvs::logtrace);

    // add PEBB specific processing hereby
    sleep(10);
  }

  // get timestamp
  rvs::lp::get_ticks(&sec, &usec);

  // add string output
  msg = "[" + stop_action_name + "] pebb all stopped";
  rvs::lp::Log(msg, rvs::logresults, sec, usec);

  // add JSON output
  r = rvs::lp::LogRecordCreate("PESM",
                               stop_action_name.c_str(), rvs::logresults,
                               sec, usec);
  rvs::lp::AddString(r, "msg", "stopped");
  rvs::lp::LogRecordFlush(r);

  rvs::lp::Log("[" + stop_action_name + "] pebb worker thread has finished",
               rvs::logdebug);
}

/**
 * @brief Stop worker thread
 *
 * Sets brun member to FALSE thus signaling end of monitoring.
 * Then it waits for std::thread to exit before returning.
 *
 * */
void Worker::stop() {
  rvs::lp::Log("[" + stop_action_name + "] pebb in Worker::stop()",
               rvs::logtrace);
  // reset "run" flag
  brun = false;
  // (give thread chance to finish processing and exit)
  sleep(200);

  // wait a bit to make sure thread has exited
  try {
    if (t.joinable())
      t.join();
  }
  catch(...) {
  }
}
