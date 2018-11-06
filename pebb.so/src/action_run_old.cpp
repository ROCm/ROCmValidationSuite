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
#include "action.h"

extern "C" {
  #include <pci/pci.h>
  #include <linux/pci.h>
}
#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "pci_caps.h"
#include "gpu_util.h"
#include "rvs_util.h"
#include "rvsloglp.h"
#include "rvshsa.h"
#include "rvstimer.h"
#include "hsa/hsa.h"


#include "rvs_module.h"
#include "worker.h"

#define RVS_CONF_LOG_INTERVAL_KEY "log_interval"
#define DEFAULT_LOG_INTERVAL 1000
#define DEFAULT_DURATION 10000

#define MODULE_NAME "pebb"
#define JSON_CREATE_NODE_ERROR "JSON cannot create node"
#define RVS_CONF_BLOCK_SIZE_KEY "block_size"

using std::cout;
using std::endl;
using std::cerr;
using std::string;
using std::vector;

/**
 * @brief Main action execution entry point. Implements test logic.
 *
 * @return 0 - if successfull, non-zero otherwise
 *
 * */
int pebbaction::run() {
  int sts;
  string msg;

  RVSTRACE_
  if (property.find("cli.-j") != property.end()) {
    bjson = true;
  }

  if (!get_all_common_config_keys())
    return -1;
  if (!get_all_pebb_config_keys())
    return -1;

  // log_interval must be less than duration
  if (prop_log_interval > 0 && gst_run_duration_ms > 0) {
    if (static_cast<uint64_t>(prop_log_interval) > gst_run_duration_ms) {
      cerr << "RVS-PEBB: action: " << action_name <<
          "  log_interval must be less than duration" << std::endl;
      return -1;
    }
  }

  sts = create_threads();

  if (sts != 0) {
    return sts;
  }

  if (gst_runs_parallel) {
    sts = run_parallel();
  } else {
    sts = run_single();
  }

  destroy_threads();

  return sts;
}

/**
 * @brief Execute test transfers one by one, in round robin fashion, for the
 * duration of the action.
 *
 * @return 0 - if successfull, non-zero otherwise
 *
 * */
int pebbaction::run_single() {
  RVSTRACE_
  // define timers
  rvs::timer<pebbaction> timer_running(&pebbaction::do_running_average, this);
  rvs::timer<pebbaction> timer_final(&pebbaction::do_final_average, this);

  // let the test run
  brun = true;

  unsigned int iter = gst_run_count > 0 ? gst_run_count : 1;
  unsigned int step = gst_run_count == 0 ? 0 : 1;

  // start timers
  if (gst_run_duration_ms) {
    RVSTRACE_
    timer_final.start(gst_run_duration_ms, true);  // ticks only once
  }

  if (prop_log_interval) {
    RVSTRACE_
    timer_running.start(prop_log_interval);        // ticks continuously
  }

  // iterate through test array and invoke tests one by one
  do {
    RVSTRACE_
    for (auto it = test_array.begin(); brun && it != test_array.end(); ++it) {
      RVSTRACE_
      (*it)->do_transfer();

      // if log interval is zero, print current results immediately
      if (prop_log_interval == 0) {
        print_running_average(*it);
      }
      sleep(1);

      if (rvs::lp::Stopping()) {
        RVSTRACE_
        brun = false;
        break;
      }
    }

    RVSTRACE_
    iter -= step;

    // insert wait between runs if needed
    if (iter > 0 && gst_run_wait_ms > 0) {
      RVSTRACE_
      sleep(gst_run_wait_ms);
    }
  } while (brun && iter);

  RVSTRACE_
  timer_running.stop();
  timer_final.stop();

  print_final_average();

  RVSTRACE_
  return rvs::lp::Stopping() ? -1 : 0;
}

/**
 * @brief Execute test transfers all at once, for the
 * duration of the action.
 *
 * @return 0 - if successfull, non-zero otherwise
 *
 * */
int pebbaction::run_parallel() {
  RVSTRACE_
  // define timers
  rvs::timer<pebbaction> timer_running(&pebbaction::do_running_average, this);
  rvs::timer<pebbaction> timer_final(&pebbaction::do_final_average, this);

  // let the test run
  brun = true;

  // start all worker threads
  for (auto it = test_array.begin(); it != test_array.end(); ++it) {
    (*it)->start();
  }

  // start timers
  if (gst_run_duration_ms) {
    timer_final.start(gst_run_duration_ms, true);  // ticks only once
  }

  if (prop_log_interval) {
    timer_running.start(prop_log_interval);        // ticks continuously
  }

  // wait for test to complete
  while (brun) {
    if (rvs::lp::Stopping()) {
      RVSTRACE_
      brun = false;
    }
    sleep(1);
  }

  timer_running.stop();
  timer_final.stop();

  // signal all worker threads to stop
  for (auto it = test_array.begin(); it != test_array.end(); ++it) {
    (*it)->stop();
  }
  sleep(10);

  // join all worker threads
  for (auto it = test_array.begin(); it != test_array.end(); ++it) {
    (*it)->join();
  }

  print_final_average();

  return rvs::lp::Stopping() ? -1 : 0;
}
