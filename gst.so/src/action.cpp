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

#include <string>
#include <vector>
#include <algorithm>
#include <regex>
#include <map>
#include <utility>

#define __HIP_PLATFORM_HCC__
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"

#include "gst_worker.h"
#include "gpu_util.h"
#include "rvs_util.h"
#include "rvs_module.h"
#include "rvsloglp.h"

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

#define MODULE_NAME                     "gst"

#define GST_DEFAULT_RAMP_INTERVAL       5000
#define GST_DEFAULT_LOG_INTERVAL        1000
#define GST_DEFAULT_MAX_VIOLATIONS      0
#define GST_DEFAULT_TOLERANCE           0.1

#define FLOATING_POINT_REGEX            "^[0-9]*\\.?[0-9]+$"

#define JSON_CREATE_NODE_ERROR          "JSON cannot create node"

/**
 * @brief default class constructor
 */
action::action() {
    bjson = false;
    json_root_node = NULL;
}

/**
 * @brief class destructor
 */
action::~action() {
    property.clear();
}


/**
 * @brief reads the stress test's ramp-up time from the module's properties collection
 */
void action::property_get_gst_ramp_interval(void) {
    gst_ramp_interval = GST_DEFAULT_RAMP_INTERVAL;
    map<string, string>::iterator it =
                            property.find(RVS_CONF_RAMP_INTERVAL_KEY);
    if (it != property.end()) {
        if (is_positive_integer(it->second))
            gst_ramp_interval = std::stoul(it->second);
        property.erase(it);
    }
}

/**
 * @brief reads the log interval from the module's properties collection
 */
void action::property_get_gst_log_interval(void) {
    gst_log_interval = GST_DEFAULT_LOG_INTERVAL;
    map<string, string>::iterator it =
                            property.find(RVS_CONF_LOG_INTERVAL_KEY);
    if (it != property.end()) {
        if (is_positive_integer(it->second))
            gst_log_interval = std::stoul(it->second);
        property.erase(it);
    }
}

/**
 * @brief reads the max_violations (maximum allowed number of GFLOPS violations)
 * from the module's properties collection
 */
void action::property_get_gst_max_violations(void) {
    gst_max_violations = GST_DEFAULT_MAX_VIOLATIONS;
    map<string, string>::iterator it =
                            property.find(RVS_CONF_MAX_VIOLATIONS_KEY);
    if (it != property.end()) {
        if (is_positive_integer(it->second))
            gst_max_violations = std::stoi(it->second);
        property.erase(it);
    }
}

/**
 * @brief reads the module's properties collection to see whether the GST should
 * copy the matrix to GPU for each SGEMM/DGEMM operation
 */
void action::property_get_gst_copy_matrix(void) {
    gst_copy_matrix = true;
    map<string, string>::iterator it = property.find(RVS_CONF_COPY_MATRIX_KEY);
    if (it != property.end()) {
        if (it->second == "false")
            gst_copy_matrix = false;
        property.erase(it);
    }
}

/**
 * @brief reads the maximum GFLOPS (that the GST will try to achieve) from
 * the module's properties collection
 * @param error pointer to a memory location where the error code will be stored
 */
void action::property_get_gst_target_stress(int *error) {
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
        property.erase(it);
    } else {
        *error = 1;
    }
}

/**
 * @brief reads the maximum GFLOPS tolerance from
 * the module's properties collection
 */
void action::property_get_gst_tolerance(void) {
    gst_tolerance = GST_DEFAULT_TOLERANCE;
    map<string, string>::iterator it = property.find(RVS_CONF_TOLERANCE_KEY);
    if (it != property.end()) {
        try {
            regex float_number_regex(FLOATING_POINT_REGEX);
            if (regex_match(it->second, float_number_regex)) {
                gst_tolerance = std::stof(it->second);
            }
        } catch (const std::regex_error& e) {
        }

        property.erase(it);
    }
}

/**
 * @brief logs an error & the GST module's result as FALSE
 * @param error the actual error message
 */
void action::log_module_error(const string &error) {
    string msg = action_name + " " + MODULE_NAME + " " + error;
    log(msg.c_str(), rvs::logerror);

    // log the module's result (FALSE) and abort GST
    msg = action_name + " " + MODULE_NAME + " " + GST_RESULT_FAIL_MESSAGE;
    log(msg.c_str(), rvs::logresults);
}

/**
 * @brief runs the test stress session
 * @param gst_gpus_device_index <gpu_index, gpu_id> map
 */
void action::do_gpu_stress_test(map<int, uint16_t> gst_gpus_device_index) {
    for (int k = 0; k < gst_run_count; k++) {
        unsigned int i = 0;
        if (gst_run_wait_ms != 0)  // delay gst execution
            sleep(gst_run_wait_ms);

        GSTWorker *worker = new GSTWorker[gst_gpus_device_index.size()];
        if (worker) {
            map<int, uint16_t>::iterator it;

            for (it = gst_gpus_device_index.begin();
                    it != gst_gpus_device_index.end(); ++it) {
                // set worker thread stress test params
                worker[i].set_name(action_name);
                worker[i].set_gpu_id(it->second);
                worker[i].set_gpu_device_index(it->first);
                worker[i].set_run_wait_ms(gst_run_wait_ms);
                worker[i].set_run_duration_ms(gst_run_duration_ms);
                worker[i].set_ramp_interval(gst_ramp_interval);
                worker[i].set_log_interval(gst_log_interval);
                worker[i].set_max_violations(gst_max_violations);
                worker[i].set_copy_matrix(gst_copy_matrix);
                worker[i].set_target_stress(gst_target_stress);
                worker[i].set_tolerance(gst_tolerance);
                i++;
            }

            if (gst_runs_parallel) {
                for (i = 0; i < gst_gpus_device_index.size(); i++)
                    worker[i].start();

                // join threads
                for (i = 0; i < gst_gpus_device_index.size(); i++)
                    worker[i].join();
            } else {
                for (i = 0; i < gst_gpus_device_index.size(); i++) {
                    worker[i].start();
                    worker[i].join();
                }
            }
            delete []worker;
        } else {
            // TODO(Tudor) log the error
        }
    }
}

/**
 * @brief checks if JSON logging is enabled (-j flag) and 
 * initializes the root node
 */
void action::init_json_logging(void) {
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
            string msg = action_name + " " + MODULE_NAME + " "
                                        + JSON_CREATE_NODE_ERROR;
            log(msg.c_str(), rvs::logerror);
        }
    }
}

/**
 * @brief runs the whole GST logic
 * @return run result
 */
int action::run(void) {
    string msg;
    bool gst_module_result = true;
    bool amd_gpus_found = false;
    bool device_all_selected = false;
    bool device_id_filtering = false;
    int error = 0;
    int hip_num_gpu_devices;
    uint16_t deviceid;
    vector<uint16_t> gpus_location_id, gpus_device_id, gpus_id;
    map<int, uint16_t> gst_gpus_device_index;

    // get the action name
    rvs::actionbase::property_get_action_name(&error);
    if(error == 2) {
      msg = "action field is missing in gst module";
      log(msg.c_str(), rvs::logerror);
      return -1;
    }
    

    init_json_logging();

    hipGetDeviceCount(&hip_num_gpu_devices);
    if (hip_num_gpu_devices == 0) {  // no AMD compatible GPU
        msg = action_name + " " + MODULE_NAME + " " + GST_RESULT_FAIL_MESSAGE;
        log(msg.c_str(), rvs::logresults);
        return 0;
    }

    // get <device> property value (a list of gpu id)
    device_all_selected = property_get_device(&error);
    if (error) {  // log the error & abort GST
        log_module_error(YAML_DEVICE_PROPERTY_ERROR);
        return 1;
    }

    // get the <deviceid> property value
    int devid = property_get_deviceid(&error);
    if (!error) {
        if (devid != -1) {
            deviceid = static_cast<uint16_t>(devid);
            device_id_filtering = true;
        }
    }

    // get all gpu_id for all AMD compatible GPUs that are registered
    // into the system, via kfd querying
    gpu_get_all_gpu_id(gpus_id);
    // get all GPU location_id (all values are unique and point to the sysfs)
    // this list is "synced" (in regards to the elements position) with gpus_id
    gpu_get_all_location_id(gpus_location_id);
    // get all GPU device_id
    // this list is "synced" (in regards to the elements position) with gpus_id
    gpu_get_all_device_id(gpus_device_id);

    if (gpus_id.empty() || gpus_location_id.empty() || gpus_device_id.empty()) {
        // basically, if we got to this point then gpus_id, gpus_location_id
        // and gpus_device_id lists cannot be empty unless an error occurred
        // while querying the kfd
        log_module_error(KFD_QUERYING_ERROR);  // log the error & abort GST
        return 1;
    }

    property_get_gst_target_stress(&error);
    if (error) {  // <target_stress> is mandatory => GST cannot continue
        log_module_error(YAML_TARGET_STRESS_PROP_ERROR);
        return 1;
    }

    // get the other action/GST related properties
    rvs::actionbase::property_get_run_parallel(&error);
    if (error == 1) {
      msg = "run parallel field is not in the correct format in gst module";
      log(msg.c_str(), rvs::loginfo);
    }
    
    
    rvs::actionbase::property_get_run_count(&error);
    if (error == 1) {
      msg = "count field is not in the correct format  in gst module";
      log(msg.c_str(), rvs::loginfo);
    }
    rvs::actionbase::property_get_run_wait(&error);
    if (error == 1) {
      msg = "wait field is not in the correct format in gst module";
      log(msg.c_str(), rvs::loginfo);
    }
    rvs::actionbase::property_get_run_duration(&error);
    if (error == 1) {
      msg = "wait field is not in the correct format in gst module";
      log(msg.c_str(), rvs::loginfo);
    }
    else if(error == 2) {
      msg = "wait field is missing in gst module";
      log(msg.c_str(), rvs::loginfo);
    }
    property_get_gst_ramp_interval();
    property_get_gst_log_interval();
    property_get_gst_max_violations();
    property_get_gst_copy_matrix();
    property_get_gst_tolerance();

    // iterate over available & compatible AMD GPUs
    for (int i = 0; i < hip_num_gpu_devices; i++) {
        // get GPU device properties
        hipDeviceProp_t props;
        hipGetDeviceProperties(&props, i);

        // compute device location_id (needed in order to identify this device
        // in the gpus_id/gpus_device_id list
        unsigned int dev_location_id =
            ((((unsigned int) (props.pciBusID)) << 8) | (props.pciDeviceID));

        vector<uint16_t>::iterator it_location_id = find(
                    gpus_location_id.begin(), gpus_location_id.end(),
                        dev_location_id);

        uint16_t index_loc_id = std::distance(gpus_location_id.begin(),
                                              it_location_id);

        // check for deviceid filtering
        if (!device_id_filtering ||
            (device_id_filtering && gpus_device_id.at(index_loc_id)
                                                        == deviceid)) {
            // check if this GPU is part of the GPU stress test
            // (either device: all or the gpu_id is in
            // the device: <gpu id> list
            bool cur_gpu_selected = false;
            uint16_t gpu_id = gpus_id.at(index_loc_id);
            if (device_all_selected) {
                cur_gpu_selected = true;
            } else {
                // search for this gpu in the list
                // provided under the <device> property
                vector<string>::iterator it_gpu_id = find(
                    device_prop_gpu_id_list.begin(),
                    device_prop_gpu_id_list.end(),
                    std::to_string(gpu_id));

                if (it_gpu_id != device_prop_gpu_id_list.end())
                    cur_gpu_selected = true;
            }

            if (cur_gpu_selected) {
                gst_gpus_device_index.insert
                    (std::pair<int, uint16_t>(i, gpus_id.at(index_loc_id)));
                amd_gpus_found = true;
            }
        }
    }

    if (!amd_gpus_found)
        gst_module_result = false;
    else
        do_gpu_stress_test(gst_gpus_device_index);

    return 0;
}
