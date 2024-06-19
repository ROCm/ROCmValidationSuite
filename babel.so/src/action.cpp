/********************************************************************************
 *
 * Copyright (c) 2018-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"

#include <string>
#include <vector>
#include <iostream>
#include <regex>
#include <utility>
#include <algorithm>
#include <map>

#include "include/rvs_key_def.h"
#include "include/rvs_util.h"
#include "include/rvsactionbase.h"
#include "include/rvsloglp.h"
#include "include/action.h"
#include "include/rvs_memworker.h"
#include "include/gpu_util.h"

using std::string;
using std::vector;
using std::map;
using std::regex;

/**
 * @brief default class constructor
 */
mem_action::mem_action() {
    module_name = MODULE_NAME;
    bjson = false;
}

/**
 * @brief class destructor
 */
mem_action::~mem_action() {
    property.clear();
}

/**
 * @brief runs the MEM test stress session
 * @param mem_gpus_device_index <gpu_index, gpu_id> map
 * @return true if no error occured, false otherwise
 */
bool mem_action::do_mem_stress_test(map<int, uint16_t> mem_gpus_device_index) {
    size_t k = 0;
    string    msg;

    for (;;) {
        unsigned int i = 0;
        if (property_wait != 0)  // delay mem execution
            sleep(property_wait);

        vector<MemWorker> workers(mem_gpus_device_index.size());

        map<int, uint16_t>::iterator it;

        // all worker instances have the same json settings
        MemWorker::set_use_json(bjson);

        msg = "[" + action_name + "] " + MODULE_NAME + " " +
            " " + " Starting all workers"; 
        rvs::lp::Log(msg, rvs::logtrace);

        for (it = mem_gpus_device_index.begin();
                it != mem_gpus_device_index.end(); ++it) {

            // set worker thread stress test params
            workers[i].set_name(action_name);
            workers[i].set_gpu_id(it->second);
            workers[i].set_gpu_device_index(it->first);
            workers[i].set_run_wait_ms(property_wait);
            workers[i].set_run_duration_ms(property_duration);
            workers[i].set_array_size(array_size);
            workers[i].set_test_type(test_type);
            workers[i].set_mibibytes(mibibytes);
            workers[i].set_output_csv(output_csv);
            workers[i].set_num_iterations(num_iterations);
            workers[i].set_subtest_type(subtest);

            i++;
        }

        if (property_parallel) {
            for (i = 0; i < mem_gpus_device_index.size(); i++)
                workers[i].start();

            // join threads
            for (i = 0; i < mem_gpus_device_index.size(); i++)
                workers[i].join();
        } else {
            for (i = 0; i < mem_gpus_device_index.size(); i++) {
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
 * @brief reads all MEM-related configuration keys from
 * the module's properties collection
 * @return true if no fatal error occured, false otherwise
 */
bool mem_action::get_all_mem_config_keys(void) {
    string    ststress;
    bool      bsts;
    string    msg;

    bsts = true;

    msg = "[" + action_name + "] " + MODULE_NAME + " " +
            " " + " Getting all mem properties"; 
    rvs::lp::Log(msg, rvs::logtrace);

    if (property_get_int<uint64_t>(RVS_CONF_ARRAY_SIZE,
                     &array_size, MEM_DEFAULT_ARRAY_SIZE)) {
        msg = "invalid '" +
        std::string(RVS_CONF_ARRAY_SIZE) + "' key value";
        rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
        bsts = false;
    }

    if (property_get_int<int>(RVS_CONF_TEST_TYPE,
                     &test_type, MEM_DEFAULT_TEST_TYPE)) {
        msg = "invalid '" +
        std::string(RVS_CONF_TEST_TYPE) + "' key value";
        rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
        bsts = false;
    }

    if (property_get_int<int>(RVS_CONF_SUBTEST,
                     &subtest, MEM_DEFAULT_SUBTEST)) {
        msg = "invalid '" +
        std::string(RVS_CONF_SUBTEST) + "' key value";
        rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
        bsts = false;
    }

    if (property_get_int<uint64_t>(RVS_CONF_NUM_ITER,
                     &num_iterations, MEM_DEFAULT_NUM_ITER)) {
        msg = "invalid '" +
        std::string(RVS_CONF_NUM_ITER) + "' key value";
        rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
        bsts = false;
    }

    if (property_get<bool>(RVS_CONF_MEM_MIBIBYTE,
                     &mibibytes, MEM_DEFAULT_MEM_MIBIBYTE)) {
        msg = "invalid '" +
        std::string(RVS_CONF_MEM_MIBIBYTE) + "' key value";
        rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
        bsts = false;
    }

    if (property_get<bool>(RVS_CONF_OP_CSV,
                     &output_csv, MEM_DEFAULT_OP_CSV)) {
        msg = "invalid '" +
        std::string(RVS_CONF_OP_CSV) + "' key value";
        rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
        bsts = false;
    }

    return bsts;
}


/**
 * @brief gets the number of ROCm compatible AMD GPUs
 * @return run number of GPUs
 */
int mem_action::get_num_amd_gpu_devices(void) {
    int hip_num_gpu_devices;
    string msg;

    hipGetDeviceCount(&hip_num_gpu_devices);
    if (hip_num_gpu_devices == 0) {  // no AMD compatible GPU
        msg = action_name + " " + MODULE_NAME + " " + MEM_NO_COMPATIBLE_GPUS;
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

            rvs::lp::AddString(json_root_node, "ERROR", MEM_NO_COMPATIBLE_GPUS);
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
int mem_action::get_all_selected_gpus(void) {
    int hip_num_gpu_devices;
    bool amd_gpus_found = false;
    map<int, uint16_t> mem_gpus_device_index;
    std::string msg;

    hip_num_gpu_devices = get_num_amd_gpu_devices();
    if (hip_num_gpu_devices < 1)
        return hip_num_gpu_devices;

    msg = "[" + action_name + "] " + MODULE_NAME + " " +
            " " + "Scan for GPU IDs"; 
    rvs::lp::Log(msg, rvs::logtrace);

    // iterate over all available & compatible AMD GPUs
    amd_gpus_found = fetch_gpu_list(hip_num_gpu_devices, mem_gpus_device_index,
                    property_device, property_device_id, property_device_all); 
    if (amd_gpus_found) {
        if (do_mem_stress_test(mem_gpus_device_index))
            return 0;

        return -1;
    } else {
      msg = "No devices match criteria from the test configuration.";
      rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
      return -1;
    }

    msg = "[" + action_name + "] " + MODULE_NAME + " " +
            " " + "Got all the GPU IDs"; 
    rvs::lp::Log(msg, rvs::logtrace);

    return 0;
}

/**
 * @brief runs the whole MEM logic
 * @return run result
 */
int mem_action::run(void) {
  string msg;
  rvs::action_result_t action_result;


  if (!get_all_common_config_keys()) {

    action_result.state = rvs::actionstate::ACTION_COMPLETED;
    action_result.status = rvs::actionstatus::ACTION_FAILED;
    action_result.output = "Error in common configuration keys.";
    action_callback(&action_result);
    return -1;
  }

  if (!get_all_mem_config_keys()) {

    action_result.state = rvs::actionstate::ACTION_COMPLETED;
    action_result.status = rvs::actionstatus::ACTION_FAILED;
    action_result.output = "Error in MEM configuration keys.";
    action_callback(&action_result);
    return -1;
  }

  auto ret = get_all_selected_gpus();

  action_result.state = rvs::actionstate::ACTION_COMPLETED;
  action_result.status = (!ret) ? rvs::actionstatus::ACTION_SUCCESS : rvs::actionstatus::ACTION_FAILED;
  action_result.output = "BABEL Module action " + action_name + " completed";
  action_callback(&action_result);

  return ret;
}

