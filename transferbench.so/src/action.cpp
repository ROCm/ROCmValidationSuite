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

#include <string>
#include <vector>
#include <iostream>
#include <regex>
#include <utility>
#include <algorithm>
#include <map>

#define __HIP_PLATFORM_HCC__

#include "include/rvs_key_def.h"
#include "include/tfb_worker.h"
#include "include/rvsactionbase.h"
#include "include/rvsloglp.h"

using std::string;
using std::vector;
using std::map;
using std::regex;

#define RVS_CONF_TFB_TYPE_KEY       "transfer-type"
#define MODULE_NAME                 "transfer-bench"
#define MODULE_NAME_CAPS            "TRANSFER-BENCH"
#define TFB_DEFAULT_TYPE            "p2p"
#define TFB_NO_COMPATIBLE_GPUS          "No AMD compatible GPU found!"

#define FLOATING_POINT_REGEX            "^[0-9]*\\.?[0-9]+$"

#define JSON_CREATE_NODE_ERROR          "JSON cannot create node"

/**
 * @brief default class constructor
 */
tfb_action::tfb_action() {
    bjson = false;
}

/**
 * @brief class destructor
 */
tfb_action::~tfb_action() {
    property.clear();
}

/**
 * @brief runs the hip test session
 * @return true if no error occured, false otherwise
 */
bool tfb_action::start_tfb_runners() {
    size_t k = 0;
    // one worker sufficient, as test runner
    tfbWorker worker;
    worker.set_name(action_name);
    worker.set_transfer_type(m_transfer_type);
    worker.start();
    worker.join();

    return rvs::lp::Stopping() ? false : true;
}

/**
 * @brief reads all GST-related configuration keys from
 * the module's properties collection
 * @return true if no fatal error occured, false otherwise
 */
bool tfb_action::get_all_tfb_config_keys(void) {
    int error;
    string msg;
    bool bsts = true;

    if (property_get<std::string>(RVS_CONF_TFB_TYPE_KEY, &m_transfer_type,
            TFB_DEFAULT_TYPE)) {
         msg = "invalid '" +
         std::string(RVS_CONF_TFB_TYPE_KEY) + "' key value";
         rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
         bsts = false;
    }

    return bsts;
}

/**
 * @brief reads all common configuration keys from
 * the module's properties collection
 * @return true if no fatal error occured, false otherwise
 */
bool tfb_action::get_all_common_config_keys(void) {
    string msg, sdevid, sdev;
    int error;
    bool bsts = true;


    // place holder for later

    return bsts;
}

/**
 * @brief gets the number of ROCm compatible AMD GPUs
 * @return run number of GPUs
 */
int tfb_action::get_num_amd_gpu_devices(void) {
    int hip_num_gpu_devices;
    string msg;

    hipGetDeviceCount(&hip_num_gpu_devices);
    if (hip_num_gpu_devices == 0) {  // no AMD compatible GPU
        msg = action_name + " " + MODULE_NAME + " " + TFB_NO_COMPATIBLE_GPUS;
        rvs::lp::Log(msg, rvs::logerror);

        if (bjson) {
            unsigned int sec;
            unsigned int usec;
            rvs::lp::get_ticks(&sec, &usec);
            void *json_root_node = rvs::lp::LogRecordCreate(MODULE_NAME,
                            action_name.c_str(), rvs::loginfo, sec, usec, true);
            if (!json_root_node) {
                // log the error
                string msg = std::string(JSON_CREATE_NODE_ERROR);
                rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
                return -1;
            }

            rvs::lp::AddString(json_root_node, "ERROR", TFB_NO_COMPATIBLE_GPUS);
            rvs::lp::LogRecordFlush(json_root_node, rvs::loginfo);
        }
        return 0;
    }
    return hip_num_gpu_devices;
}

/**
 * @brief gets all selected GPUs and starts the worker threads
 * @return run result
 */
int tfb_action::run_transferbench(void) {
    int hip_num_gpu_devices;
    std::string msg;

    hip_num_gpu_devices = get_num_amd_gpu_devices();
     
    if (hip_num_gpu_devices > 0) {
        if (start_tfb_runners())
            return 0;

        return -1;
    } else {
      msg = "No devices match criteria from the test configuation.";
      rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
      return -1;
    }

    return 0;
}

/**
 * @brief runs the whole GST logic
 * @return run result
 */
int tfb_action::run(void) {
    string msg;

    // get the action name
    if (property_get(RVS_CONF_NAME_KEY, &action_name)) {
      rvs::lp::Err("Action name missing", MODULE_NAME_CAPS);
      return -1;
    }

    // check for -j flag (json logging)
    if (property.find("cli.-j") != property.end())
        bjson = true;
    if (!get_all_common_config_keys())
        return -1;
    if (!get_all_tfb_config_keys())
        return -1;

    return run_transferbench();
}
