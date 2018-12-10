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
#include "include/action.h"

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

#include "include/rvs_key_def.h"
#include "include/pci_caps.h"
#include "include/gpu_util.h"
#include "include/rvs_util.h"
#include "include/rvsloglp.h"
#include "include/rvshsa.h"
#include "include/rvstimer.h"

#include "include/rvs_module.h"
#include "include/worker.h"


#define MODULE_NAME "pqt"
#define MODULE_NAME_CAPS "PQT"

using std::string;
using std::vector;


/**
 * @brief Main action execution entry point. Implements test logic.
 *
 * @return 0 - if successfull, non-zero otherwise
 *
 * */
int pqt_action::run() {
  int sts;
  string msg;

  rvs::lp::Log("int pqt_action::run()", rvs::logtrace);

  if (property.find("cli.-j") != property.end()) {
    bjson = true;
  }

  if (!get_all_common_config_keys()) {
    msg = "Error in get_all_common_config_keys()";
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
    return -1;
  }
  if (!get_all_pqt_config_keys()) {
    msg = "Error in get_all_pqt_config_keys()";
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
    return -1;
  }

  // log_interval must be less than duration
  if (property_log_interval > 0 && property_duration > 0) {
    if (static_cast<uint64_t>(property_log_interval) > property_duration) {
      msg = "log_interval must be less than duration";
      rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
      return -1;
    }
  }

  sts = create_threads();
  if (sts) {
    RVSTRACE_
    return sts;
  }

  if (!prop_test_bandwidth || test_array.size() < 1) {
    RVSTRACE_
    // do cleanup
    destroy_threads();
    return 0;
  }

  RVSTRACE_
  // define timers
  rvs::timer<pqt_action> timer_running(&pqt_action::do_running_average, this);
  rvs::timer<pqt_action> timer_final(&pqt_action::do_final_average, this);

  unsigned int iter = property_count > 0 ? property_count : 1;
  unsigned int step = 1;

  do {
    RVSTRACE_
    // let the test run in this iteration
    brun = true;

    // start timers
    if (property_duration) {
      RVSTRACE_
      timer_final.start(property_duration, true);  // ticks only once
    }

    if (property_log_interval) {
      RVSTRACE_
      timer_running.start(property_log_interval);        // ticks continuously
    }

    do {
      RVSTRACE_

      if (property_parallel) {
        sts = run_parallel();
      } else {
        sts = run_single();
      }
    } while (brun);

    RVSTRACE_
    timer_running.stop();
    timer_final.stop();

    iter -= step;

    // insert wait between runs if needed
    if (iter > 0 && property_wait > 0) {
      RVSTRACE_
      sleep(property_wait);
    }
  } while (iter && !rvs::lp::Stopping());

  RVSTRACE_
  sts = rvs::lp::Stopping() ? -1 : 0;

  print_final_average();


  // do cleanup
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
int pqt_action::run_single() {
  RVSTRACE_
  int sts = 0;

  // iterate through test array and invoke tests one by one
  for (auto it = test_array.begin(); brun && it != test_array.end(); ++it) {
    RVSTRACE_
    (*it)->do_transfer();

    // if log interval is zero, print current results immediately
    if (property_log_interval == 0) {
      print_running_average(*it);
    }

    if (rvs::lp::Stopping()) {
      RVSTRACE_
      brun = false;
      sts = -1;
      break;
    }
  }

  return sts;
}

/**
 * @brief Execute test transfers all at once, for the
 * duration of the action.
 *
 * @return 0 - if successfull, non-zero otherwise
 *
 * */
int pqt_action::run_parallel() {
  RVSTRACE_

  // start all worker threads
  for (auto it = test_array.begin(); it != test_array.end(); ++it) {
    (*it)->start();
  }

  // join all worker threads
  for (auto it = test_array.begin(); it != test_array.end(); ++it) {
    (*it)->join();
  }

  return rvs::lp::Stopping() ? -1 : 0;
}


