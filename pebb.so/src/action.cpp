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

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <algorithm>

#include "rvs_module.h"
#include "worker.h"
#include "pci_caps.h"
#include "gpu_util.h"
#include "rvs_util.h"
#include "rvsloglp.h"


using std::string;
using std::cout;
using std::cerr;
using std::endl;
using std::hex;


extern Worker* pworker;

//! Default constructor
action::action() {
}

//! Default destructor
action::~action() {
  property.clear();
}

/**
 * @brief Implements action functionality
 *
 * Functionality:
 *
 * - If "do_gpu_list" property is set,
 *   it lists all AMD GPUs present in the system and exits
 * - If "monitor" property is set to "true",
 *   it creates Worker thread and initiates monitoring and exits
 * - If "monitor" property is not set or is not set to "true",
 *   it stops the Worker thread and exits
 *
 * @return 0 - success. non-zero otherwise
 *
 * */
int action::run(void) {
  rvs::lp::Log("[" + property["name"]+ "] pebb in run()", rvs::logtrace);

  // debugging help
  string val;
  if (has_property("debugwait", &val)) {
    sleep(std::stoi(val));
  }

  // start of monitoring?
  if (property["monitor"] == "true") {
    if (pworker) {
      rvs::lp::Log("[" + property["name"]+ "] pebb monitoring already started",
                  rvs::logresults);
      return 0;
    }

    rvs::lp::Log("[" + property["name"]+
    "] pebb property[\"monitor\"] == \"true\"", rvs::logtrace);

    // create worker thread object
    rvs::lp::Log("[" + property["name"]+ "] pebb creating Worker",
                 rvs::logtrace);

    pworker = new Worker();
    pworker->set_name(property["name"]);

    // check if  -j flag is passed
    if (has_property("cli.-j")) {
      pworker->json(true);
    }

    // checki if deviceid filtering is required
    string sdevid;
    if (has_property("deviceid", &sdevid)) {
      if (::is_positive_integer(sdevid)) {
        try {
          pworker->set_deviceid(std::stoi(sdevid));
        }
        catch(...) {
          cerr << "RVS-PEBB: action: " << property["name"] <<
          "  invalide 'deviceid' key value: " << sdevid << std::endl;
          return -1;
        }
      } else {
        cerr << "RVS-PEBB: action: " << property["name"] <<
        "  invalide 'deviceid' key value: " << sdevid << std::endl;
        return -1;
      }
    }

    // check if GPU id filtering is requied
    string sdev;
    if (has_property("device", &sdev)) {
      pworker->set_strgpuids(sdev);
      if (sdev != "all") {
        vector<string> sarr = str_split(sdev, YAML_DEVICE_PROP_DELIMITER);
        vector<int> iarr;
        int sts = rvs_util_strarr_to_intarr(sarr, &iarr);
        if (sts < 0) {
          cerr << "RVS-PEBB: action: " << property["name"] <<
          "  invalide 'device' key value: " << sdev << std::endl;
          return -1;
        }
        pworker->set_gpuids(iarr);
      }
    } else {
          cerr << "RVS-PEBB: action: " << property["name"] <<
          "  key 'device' not found" << std::endl;
          return -1;
    }

    // start worker thread
    rvs::lp::Log("[" + property["name"]+ "] pebb starting Worker",
                 rvs::logtrace);
    pworker->start();
    sleep(2);

    rvs::lp::Log("[" + property["name"]+ "] pebb Worker thread started",
                 rvs::logtrace);
  } else {
    rvs::lp::Log("[" + property["name"]+
    "] pebb property[\"monitor\"] != \"true\"", rvs::logtrace);
    if (pworker) {
      // (give thread chance to start)
      sleep(2);
      pworker->set_stop_name(property["name"]);
      pworker->stop();
      delete pworker;
      pworker = nullptr;
    }
    rvs::lp::Log("[" + property["name"]+ "] pebb Worker stopped",
                 rvs::logtrace);
  }
  return 0;
}
