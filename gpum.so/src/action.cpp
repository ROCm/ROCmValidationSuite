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

#include <assert.h>

#include <string>
#include <map>
#include <vector>
#include <memory>
#include "rvsloglp.h"
#include "rvs_module.h"
#include "rvs_util.h"
#include "action.h"
#include "rocm_smi/rocm_smi.h"

#define JSON_CREATE_NODE_ERROR          "JSON cannot create node"
#define MODULE_NAME                     "gm"

using std::map;
using std::string;
using std::vector;
/*using namespace amd::smi;*/

// collection of allowed io links properties
const char* metric_names[] =
        { "temp", "clock", "mem_clock", "fan", "power"
        };

// Call-back function to append to a vector of Devices
static bool GetMonitorDevices(const std::shared_ptr<amd::smi::Device> &d,
            void *p) {
  std::string val_str;

  assert(p != nullptr);

  std::vector<std::shared_ptr<amd::smi::Device>> *device_list =
    reinterpret_cast<std::vector<std::shared_ptr<amd::smi::Device>> *>(p);

  if (d->monitor() != nullptr) {
    device_list->push_back(d);
  }
  return false;
}

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
    int error = 0;
    string err_msg;
    amd::smi::RocmSMI hw;
    std::vector<std::shared_ptr<amd::smi::Device>> monitor_devices;

    // DiscoverDevices() will seach for devices and monitors and update internal
    // data structures.
    hw.DiscoverDevices();

    // IterateSMIDevices will iterate through all the known devices and apply
    // the provided call-back to each device found.
    hw.IterateSMIDevices(GetMonitorDevices,
        reinterpret_cast<void *>(&monitor_devices));

    std::string val_str;
    std::vector<std::string> val_vec;
    uint32_t value;
    uint32_t value2;
    int ret;

    // get the action name
    rvs::actionbase::property_get_action_name(&error);
    if (error == 2) {
      err_msg = "action field is missing in gpum module";
      log(err_msg.c_str(), rvs::logerror);
      return -1;
    }

    bjson = false;  // already initialized in the default constructor

    // check for -j flag (json logging)
    if (property.find("cli.-j") != property.end()) {
        unsigned int sec;
        unsigned int usec;
        rvs::lp::get_ticks(sec, usec);

        bjson = true;

        json_root_node = rvs::lp::LogRecordCreate(MODULE_NAME,
        action_name.c_str(), rvs::loginfo, sec, usec);
        if (json_root_node == NULL) {
            // log the error
            string msg = action_name + " " + MODULE_NAME + " " +
            JSON_CREATE_NODE_ERROR;
            log(msg.c_str(), rvs::logerror);
        }
    }

    string msg = action_name + " " + MODULE_NAME + " " +
                            "gpu_id" + " " + " started";
    log(msg.c_str(), rvs::logresults);

    auto metric_length = std::end(metric_names) - std::begin(metric_names);
    for (int i = 0; i < metric_length; i++) {
        msg = action_name + " " + MODULE_NAME + " " +
            "gpu_id" + " " + " monitoring " + metric_names[i] +
            " bounds min:" + "min_metric" +
            " max:" + "max_metric";
        log(msg.c_str(), rvs::loginfo);
    }

    msg = action_name + " " + MODULE_NAME + " " +
                            "gpu_id" + " " + " stopped";
    log(msg.c_str(), rvs::logresults);

    // TODO(bsimeunovic) Iterate through the list of devices and print out
    // information related to that device.

    if (bjson && json_root_node != NULL) {  // json logging stuff
        rvs::lp::LogRecordFlush(json_root_node);
    }

  return 0;
}
