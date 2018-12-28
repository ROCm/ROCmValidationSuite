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
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"

#include "include/rvs_key_def.h"
#include "include/gst_worker.h"
#include "include/gpu_util.h"
#include "include/rvs_util.h"
#include "include/rvsactionbase.h"
#include "include/rvsloglp.h"

using std::string;
using std::vector;
using std::map;
using std::regex;

#define RVS_CONF_RAMP_INTERVAL_KEY      "ramp_interval"
#define RVS_CONF_LOG_INTERVAL_KEY       "log_interval"
#define RVS_CONF_MAX_VIOLATIONS_KEY     "max_violations"
#define RVS_CONF_COPY_MATRIX_KEY        "copy_matrix"
#define RVS_CONF_TARGET_STRESS_KEY      "target_stress"
#define RVS_CONF_TOLERANCE_KEY          "tolerance"
#define RVS_CONF_MATRIX_SIZE_KEY        "matrix_size"

#define MODULE_NAME                     "gst"
#define MODULE_NAME_CAPS                "GST"

#define GST_DEFAULT_RAMP_INTERVAL       5000
#define GST_DEFAULT_LOG_INTERVAL        1000
#define GST_DEFAULT_MAX_VIOLATIONS      0
#define GST_DEFAULT_TOLERANCE           0.1
#define GST_DEFAULT_COPY_MATRIX         true
#define GST_DEFAULT_MATRIX_SIZE         5760

#define RVS_DEFAULT_PARALLEL            false
#define RVS_DEFAULT_DURATION            0

#define GST_NO_COMPATIBLE_GPUS          "No AMD compatible GPU found!"

#define FLOATING_POINT_REGEX            "^[0-9]*\\.?[0-9]+$"

#define JSON_CREATE_NODE_ERROR          "JSON cannot create node"

/**
 * @brief default class constructor
 */
gst_action::gst_action() {
    bjson = false;
}

/**
 * @brief class destructor
 */
gst_action::~gst_action() {
    property.clear();
}

/**
 * @brief reads the module's properties collection to see whether the GST should
 * copy the matrices to GPU before each SGEMM/DGEMM operation
 * @param error pointer to a memory location where the error code will be stored
 */
void gst_action::property_get_gst_copy_matrix(int *error) {
    *error = 0;
    gst_copy_matrix = GST_DEFAULT_COPY_MATRIX;
    map<string, string>::iterator it = property.find(RVS_CONF_COPY_MATRIX_KEY);
    if (it != property.end()) {
        if (it->second == "true")
            gst_copy_matrix = true;
        else
        if (it->second == "false")
            gst_copy_matrix = false;
        else
            *error = 1;
        property.erase(it);
    }
}

/**
 * @brief reads the GFLOPS value (that the GST will try to achieve) from
 * the module's properties collection
 * @param error pointer to a memory location where the error code will be stored
 */
void gst_action::property_get_gst_target_stress(int *error) {
    *error = 0;  // init with 'no error'
    map<string, string>::iterator it =
                            property.find(RVS_CONF_TARGET_STRESS_KEY);
    if (it != property.end()) {
        try {
            regex float_number_regex(FLOATING_POINT_REGEX);
            if (!regex_match(it->second, float_number_regex)) {
                *error = 1;  // not a floating point number
            } else {
                gst_target_stress = std::stof(it->second);
            }
        } catch (const std::regex_error& e) {
            *error = 1;  // something went wrong with the regex
        }
    } else {
        *error = 1;
    }
}

/**
 * @brief reads the maximum GFLOPS tolerance from
 * the module's properties collection
 * @param error pointer to a memory location where the error code will be stored
 */
void gst_action::property_get_gst_tolerance(int *error) {
    *error = 0;
    gst_tolerance = GST_DEFAULT_TOLERANCE;
    map<string, string>::iterator it = property.find(RVS_CONF_TOLERANCE_KEY);
    if (it != property.end()) {
        try {
            regex float_number_regex(FLOATING_POINT_REGEX);
            if (regex_match(it->second, float_number_regex)) {
                gst_tolerance = std::stof(it->second);
            } else {
                *error = 1;  // not a floating point number
            }
        } catch (const std::regex_error& e) {
            *error = 1;
        }
        property.erase(it);
    }
}

/**
 * @brief runs the GST test stress session
 * @param gst_gpus_device_index <gpu_index, gpu_id> map
 * @return true if no error occured, false otherwise
 */
bool gst_action::do_gpu_stress_test(map<int, uint16_t> gst_gpus_device_index) {
    size_t k = 0;
    for (;;) {
        unsigned int i = 0;
        if (property_wait != 0)  // delay gst execution
            sleep(property_wait);

        vector<GSTWorker> workers(gst_gpus_device_index.size());

        map<int, uint16_t>::iterator it;

        // all worker instances have the same json settings
        GSTWorker::set_use_json(bjson);

        for (it = gst_gpus_device_index.begin();
                it != gst_gpus_device_index.end(); ++it) {
            // set worker thread stress test params
            workers[i].set_name(action_name);
            workers[i].set_gpu_id(it->second);
            workers[i].set_gpu_device_index(it->first);
            workers[i].set_run_wait_ms(property_wait);
            workers[i].set_run_duration_ms(property_duration);
            workers[i].set_ramp_interval(gst_ramp_interval);
            workers[i].set_log_interval(property_log_interval);
            workers[i].set_max_violations(gst_max_violations);
            workers[i].set_copy_matrix(gst_copy_matrix);
            workers[i].set_target_stress(gst_target_stress);
            workers[i].set_tolerance(gst_tolerance);
            workers[i].set_matrix_size(gst_matrix_size);
            i++;
        }

        if (property_parallel) {
            for (i = 0; i < gst_gpus_device_index.size(); i++)
                workers[i].start();

            // join threads
            for (i = 0; i < gst_gpus_device_index.size(); i++)
                workers[i].join();
        } else {
            for (i = 0; i < gst_gpus_device_index.size(); i++) {
                workers[i].start();
                workers[i].join();

                // check if stop signal was received
                if (rvs::lp::Stopping())
                    return false;
            }
        }

        // check if stop signal was received
        if (rvs::lp::Stopping())
            return false;

        if (property_count != 0) {
            k++;
            if (k == property_count)
                break;
        }
    }

    return rvs::lp::Stopping() ? false : true;
}

/**
 * @brief reads all GST-related configuration keys from
 * the module's properties collection
 * @return true if no fatal error occured, false otherwise
 */
bool gst_action::get_all_gst_config_keys(void) {
    int error;
    string msg, ststress;

    if (has_property(RVS_CONF_TARGET_STRESS_KEY, &ststress)) {
        property_get_gst_target_stress(&error);
        if (error) {  // <target_stress> is mandatory => GST cannot continue
            msg = "invalid '" + std::string(RVS_CONF_TARGET_STRESS_KEY) +
                "' key value " + ststress;
            rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
            return false;
        }
    } else {
        msg = "key '" + std::string(RVS_CONF_TARGET_STRESS_KEY) +
            "' was not found";
        rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
        return false;
    }

    if (property_get_int<uint64_t>(RVS_CONF_RAMP_INTERVAL_KEY,
      &gst_ramp_interval, GST_DEFAULT_RAMP_INTERVAL)) {
        msg = "invalid '" +
        std::string(RVS_CONF_RAMP_INTERVAL_KEY) + "' key value";
        rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
        return false;
    }

    if (property_get_int<uint64_t>(RVS_CONF_LOG_INTERVAL_KEY,
      &property_log_interval, GST_DEFAULT_LOG_INTERVAL)) {
        msg = "invalid '" +
        std::string(RVS_CONF_LOG_INTERVAL_KEY) + "' key value";
        rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
        return false;
    }

    if (property_get_int<int>(RVS_CONF_MAX_VIOLATIONS_KEY, &gst_max_violations,
     GST_DEFAULT_MAX_VIOLATIONS)) {
        msg = "invalid '" +
        std::string(RVS_CONF_MAX_VIOLATIONS_KEY) + "' key value";
        rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
        return false;
    }

    property_get_gst_copy_matrix(&error);
    if (error) {
        msg = "invalid '" +
        std::string(RVS_CONF_COPY_MATRIX_KEY) + "' key value";
        rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
        return false;
    }

    property_get_gst_tolerance(&error);
    if (error) {
        msg = "invalid '" +
        std::string(RVS_CONF_TOLERANCE_KEY) + "' key value";
        rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
        return false;
    }

    error = property_get_int<uint64_t>
    (RVS_CONF_MATRIX_SIZE_KEY, &gst_matrix_size, GST_DEFAULT_MATRIX_SIZE);
    if (error == 1) {
        msg = "invalid '" +
        std::string(RVS_CONF_MATRIX_SIZE_KEY) + "' key value";
        rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
        return false;
    }
    return true;
}

/**
 * @brief reads all common configuration keys from
 * the module's properties collection
 * @return true if no fatal error occured, false otherwise
 */
bool gst_action::get_all_common_config_keys(void) {
    string msg, sdevid, sdev;
    int error;

    // get <device> property value (a list of gpu id)
    if (int sts = property_get_device()) {
      switch (sts) {
      case 1:
        msg = "Invalid 'device' key value.";
        break;
      case 2:
        msg = "Missing 'device' key.";
        break;
      }
      rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
      return -1;
    }

    // get the <deviceid> property value if provided
    if (property_get_int<uint16_t>(RVS_CONF_DEVICEID_KEY,
                                  &property_device_id, 0u)) {
      msg = "Invalid 'deviceid' key value.";
      rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
      return -1;
    }

    // get the other action/GST related properties
    if (property_get(RVS_CONF_PARALLEL_KEY, &property_parallel, false)) {
        msg = "invalid '" +
            std::string(RVS_CONF_PARALLEL_KEY) + "' key value";
        rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
        return false;
    }

    error = property_get_int<uint64_t>
    (RVS_CONF_COUNT_KEY, &property_count, DEFAULT_COUNT);
    if (error != 0) {
        msg = "invalid '" +
            std::string(RVS_CONF_COUNT_KEY) + "' key value";
        rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
        return false;
    }

    error = property_get_int<uint64_t>
    (RVS_CONF_WAIT_KEY, &property_wait, DEFAULT_WAIT);
    if (error == 1) {
        msg = "invalid '" +
            std::string(RVS_CONF_WAIT_KEY) + "' key value";
        return false;
    }

    error = property_get_int<uint64_t>
    (RVS_CONF_DURATION_KEY, &property_duration, RVS_DEFAULT_DURATION);
    if (error == 1) {
        msg = "invalid '" +
            std::string(RVS_CONF_DURATION_KEY) + "' key value";
        rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
        return false;
    }

    return true;
}

/**
 * @brief gets the number of ROCm compatible AMD GPUs
 * @return run number of GPUs
 */
int gst_action::get_num_amd_gpu_devices(void) {
    int hip_num_gpu_devices;
    string msg;

    hipGetDeviceCount(&hip_num_gpu_devices);
    if (hip_num_gpu_devices == 0) {  // no AMD compatible GPU
        msg = action_name + " " + MODULE_NAME + " " + GST_NO_COMPATIBLE_GPUS;
        rvs::lp::Log(msg, rvs::logerror);

        if (bjson) {
            unsigned int sec;
            unsigned int usec;
            rvs::lp::get_ticks(&sec, &usec);
            void *json_root_node = rvs::lp::LogRecordCreate(MODULE_NAME,
                            action_name.c_str(), rvs::loginfo, sec, usec);
            if (!json_root_node) {
                // log the error
                string msg = std::string(JSON_CREATE_NODE_ERROR);
                rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
                return -1;
            }

            rvs::lp::AddString(json_root_node, "ERROR", GST_NO_COMPATIBLE_GPUS);
            rvs::lp::LogRecordFlush(json_root_node);
        }
        return 0;
    }
    return hip_num_gpu_devices;
}

/**
 * @brief gets all selected GPUs and starts the worker threads
 * @return run result
 */
int gst_action::get_all_selected_gpus(void) {
    int hip_num_gpu_devices;
    bool amd_gpus_found = false;
    map<int, uint16_t> gst_gpus_device_index;

    hip_num_gpu_devices = get_num_amd_gpu_devices();
    if (hip_num_gpu_devices < 1)
        return hip_num_gpu_devices;

    // iterate over all available & compatible AMD GPUs
    for (int i = 0; i < hip_num_gpu_devices; i++) {
        // get GPU device properties
        hipDeviceProp_t props;
        hipGetDeviceProperties(&props, i);

        // compute device location_id (needed in order to identify this device
        // in the gpus_id/gpus_device_id list
        unsigned int dev_location_id =
            ((((unsigned int) (props.pciBusID)) << 8) | (props.pciDeviceID));

        uint16_t devId;
        if (rvs::gpulist::location2device(dev_location_id, &devId)) {
          continue;
        }

        // filter by device id if needed
        if (property_device_id > 0 && property_device_id != devId)
          continue;

        // check if this GPU is part of the GPU stress test
        // (device = "all" or the gpu_id is in the device: <gpu id> list)
        bool cur_gpu_selected = false;
        uint16_t gpu_id;
        // if not and AMD GPU just continue
        if (rvs::gpulist::location2gpu(dev_location_id, &gpu_id))
          continue;


        if (property_device_all) {
            cur_gpu_selected = true;
        } else {
            // search for this gpu in the list
            // provided under the <device> property
            auto it_gpu_id = find(property_device.begin(),
                                  property_device.end(),
                                  gpu_id);

            if (it_gpu_id != property_device.end())
                cur_gpu_selected = true;
        }

        if (cur_gpu_selected) {
            gst_gpus_device_index.insert
                (std::pair<int, uint16_t>(i, gpu_id));
            amd_gpus_found = true;
        }
    }

    if (amd_gpus_found) {
        if (do_gpu_stress_test(gst_gpus_device_index))
            return 0;

        return -1;
    }

    return 0;
}

/**
 * @brief runs the whole GST logic
 * @return run result
 */
int gst_action::run(void) {
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
    if (!get_all_gst_config_keys())
        return -1;

    if (property_duration > 0 && (property_duration < gst_ramp_interval)) {
        msg = "'" +
            std::string(RVS_CONF_DURATION_KEY) + "' cannot be less than '" +
            std::string(RVS_CONF_RAMP_INTERVAL_KEY) + "'";
        rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
        return -1;
    }

    return get_all_selected_gpus();
}
