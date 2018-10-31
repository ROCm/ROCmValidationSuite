/*******************************************************************************
 *
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

#include <iostream>
#include <sstream>
#include <string>
#include <map>
#include <vector>
#include <memory>

#ifdef __cplusplus
extern "C" {
#endif
#include <pci/pci.h>
#ifdef __cplusplus
}
#endif

#include "rvs_key_def.h"
#include "rvsloglp.h"
#include "rvs_module.h"
#include "rvs_util.h"
#include "worker.h"
#include "pci_caps.h"
#include "gpu_util.h"

#define JSON_CREATE_NODE_ERROR          "JSON cannot create node"
#define MODULE_NAME                     "gm"
#define MODULE_NAME_CAPS                "GM"

using std::map;
using std::string;
using std::vector;
using std::thread;
using std::endl;

extern Worker* pworker;

/**
 * default class constructor
 */
action::action() {
    bjson = false;
    json_root_node = NULL;
}

/**
 * class destructor
 */
action::~action() {
    property.clear();
}

/**
 * @brief Implements action functionality
 *
 * Functionality:
 * 
 * @return 0 - success. non-zero otherwise
 *
 * */
int action::run(void) {
    map<string, string>::iterator it;  // module's properties map iterator

    string msg;
    int sample_interval = 1000, log_interval = 0;
    int error = 0;
    bool metric_true, metric_bound;
    int metric_min, metric_max;
    bool terminate = false;
    uint64_t duration = 1;
    std::vector<uint16_t> gpu_id;

    gpu_get_all_gpu_id(&gpu_id);

    if (rvs::actionbase::has_property("sample_interval")) {
        sample_interval =
        rvs::actionbase::property_get_sample_interval(&error);
    }

    if (rvs::actionbase::has_property("log_interval")) {
        log_interval = rvs::actionbase::property_get_log_interval(&error);
        if ( log_interval < sample_interval ) {
          msg = property["name"] +
          "Log interval has the lower value than the sample interval ";
          rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
          return -1;
        }
    }

    if (rvs::actionbase::has_property("terminate")) {
        terminate = rvs::actionbase::property_get_terminate(&error);
    }

    // start of monitoring?
    if (property["monitor"] == "true") {
      if (pworker) {
        rvs::lp::Log("[" + property["name"]+ "] gm monitoring already started",
                  rvs::logresults);
      return 0;
      }

      pworker = new Worker();
      pworker->set_name(property["name"]);
      pworker->set_sample_int(sample_interval);
      pworker->set_log_int(log_interval);
      pworker->set_terminate(terminate);
      if (property["force"] == "true")
        pworker->set_force(true);

      if (rvs::actionbase::has_property("duration")) {
        rvs::actionbase::property_get_run_duration(&error);
        duration = rvs::actionbase::gst_run_duration_ms;
      }

      for (it = property.begin(); it != property.end(); ++it) {
        metric_bound = false;
        string word;
        string s = it->first;
        if (s.find(".") != std::string::npos && s.substr(0, s.find(".")) ==
          "metrics") {
          string metric = s.substr(s.find(".")+1);
          s = it->second;
          vector<string> values = str_split(s, YAML_DEVICE_PROP_DELIMITER);

          if (values.size() == 3) {
            metric_true = (values[0] == "true") ? true : false;
            metric_max = std::stoi(values[1]);
            metric_min = std::stoi(values[2]);
            metric_bound = true;
          } else {
            msg = property["name"] +" Wrong number of metric parameters ";
            rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
            return -1;
          }

        pworker->set_metr_mon(metric, metric_true);
        pworker->set_bound(metric, metric_bound, metric_max, metric_min);
        values.clear();
      }
    }

    // check if  -j flag is passed
    if (has_property("cli.-j")) {
      bjson = true;
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
          msg = property["name"] +
          "  invalide 'deviceid' key value: " + sdevid;
          rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
          return -1;
        }
      } else {
        msg = property["name"] +
        "  invalide 'deviceid' key value: " + sdevid;
        rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
        return -1;
      }
    }

    // check if GPU id filtering is requied
    string sdev;
    if (has_property("device", &sdev)) {
      vector<uint16_t> iarr;
      if (sdev != "all") {
        vector<string> sarr = str_split(sdev, YAML_DEVICE_PROP_DELIMITER);
        int sts = rvs_util_strarr_to_uintarr(sarr, &iarr);
        if (sts < 0) {
          msg = property["name"] +
          "  invalide 'device' key value: " + sdev;
          rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
          return -1;
        }
        pworker->set_gpuids(iarr);
      } else {
         pworker->set_gpuids(gpu_id);
      }
    } else {
          msg = property["name"] +
          "  key 'device' not found";
          rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
          return -1;
    }
    // set stop name before start
    pworker->set_stop_name(property["name"]);
    // start worker thread
    rvs::lp::Log("[" + property["name"]+ "] gm starting Worker",
                 rvs::logtrace);
    pworker->start();
    sleep(duration);

    rvs::lp::Log("[" + property["name"]+ "] gm Monitoring started",
                 rvs::logtrace);
  } else {
    rvs::lp::Log("[" + property["name"]+
    "] gm property[\"monitor\"] != \"true\"", rvs::logtrace);
    if (pworker) {
      // (give thread chance to start)
      sleep(2);
      pworker->set_stop_name(property["name"]);
      pworker->stop();
      delete pworker;
      pworker = nullptr;
    }
  }

     if (bjson && json_root_node != NULL) {  // json logging stuff
         rvs::lp::LogRecordFlush(json_root_node);
     }

  return 0;
}
