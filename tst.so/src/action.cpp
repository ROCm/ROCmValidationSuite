/********************************************************************************
 *
 * Copyright (c) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <fstream>
#include <regex>
#include <utility>
#include <algorithm>
#include <memory>
#include <map>

#ifdef __cplusplus
extern "C" {
#endif
#include <pci/pci.h>
#ifdef __cplusplus
}
#endif
#include <dirent.h>

#define __HIP_PLATFORM_HCC__
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"

#include "include/rvs_key_def.h"
#include "include/tst_worker.h"
#include "include/gpu_util.h"
#include "include/rvs_util.h"
#include "include/rvs_module.h"
#include "include/rvsactionbase.h"
#include "include/rvsloglp.h"
#include "include/rsmi_util.h"

using std::string;
using std::vector;
using std::map;
using std::regex;
using std::fstream;


#define RVS_CONF_THROTTLE_TEMP_KEY      "throttle_temp"
#define RVS_CONF_TARGET_TEMP_KEY        "target_temp"
#define RVS_CONF_RAMP_INTERVAL_KEY      "ramp_interval"
#define RVS_CONF_TOLERANCE_KEY          "tolerance"
#define RVS_CONF_MAX_VIOLATIONS_KEY     "max_violations"
#define RVS_CONF_SAMPLE_INTERVAL_KEY    "sample_interval"
#define RVS_CONF_LOG_INTERVAL_KEY       "log_interval"
#define RVS_CONF_MATRIX_SIZE_KEY        "matrix_size"
#define RVS_CONF_TST_OPS_TYPE           "ops_type"
#define RVS_CONF_MATRIX_SIZE_KEYA       "matrix_size_a"
#define RVS_CONF_MATRIX_SIZE_KEYB       "matrix_size_b"
#define RVS_CONF_MATRIX_SIZE_KEYC       "matrix_size_c"
#define RVS_CONF_TST_OPS_TYPE           "ops_type"
#define RVS_CONF_TRANS_A                "transa"
#define RVS_CONF_TRANS_B                "transb"
#define RVS_CONF_ALPHA_VAL              "alpha"
#define RVS_CONF_BETA_VAL               "beta"
#define RVS_CONF_LDA_OFFSET             "lda"
#define RVS_CONF_LDB_OFFSET             "ldb"
#define RVS_CONF_LDC_OFFSET             "ldc"
#define RVS_CONF_LDD_OFFSET             "ldd"
#define RVS_CONF_TT_FLAG                "targettemp_met"
#define RVS_TT_MESSAGE                  "target_temp"
#define RVS_DTYPE_MESSAGE               "dtype"



#define TST_DEFAULT_RAMP_INTERVAL       5000
#define TST_DEFAULT_LOG_INTERVAL        1000
#define TST_DEFAULT_MAX_VIOLATIONS      0
#define TST_DEFAULT_TOLERANCE           0.1
#define TST_DEFAULT_SAMPLE_INTERVAL     100
#define TST_DEFAULT_MATRIX_SIZE         5760
#define RVS_DEFAULT_PARALLEL            false
#define RVS_DEFAULT_DURATION            500
#define TST_DEFAULT_OPS_TYPE            "sgemm"
#define TST_DEFAULT_TRANS_A             0
#define TST_DEFAULT_TRANS_B             1
#define TST_DEFAULT_ALPHA_VAL           1
#define TST_DEFAULT_BETA_VAL            1
#define TST_DEFAULT_LDA_OFFSET          0
#define TST_DEFAULT_LDB_OFFSET          0
#define TST_DEFAULT_LDC_OFFSET          0
#define TST_DEFAULT_LDD_OFFSET          0
#define TST_DEFAULT_TT_FLAG             false

#define TST_NO_COMPATIBLE_GPUS          "No AMD compatible GPU found!"
#define PCI_ALLOC_ERROR                 "pci_alloc() error"
#define FLOATING_POINT_REGEX            "^[0-9]*\\.?[0-9]+$"
#define JSON_CREATE_NODE_ERROR          "JSON cannot create node"

static constexpr auto MODULE_NAME = "tst";
static constexpr auto MODULE_NAME_CAPS = "TST";

/**
 * @brief default class constructor
 */
tst_action::tst_action() {
  bjson = false;
  module_name = MODULE_NAME;
}

/**
 * @brief class destructor
 */
tst_action::~tst_action() {
    property.clear();
}


/**
 * @brief reads all TST's related configuration keys from
 * the module's properties collection
 * @return true if no fatal error occured, false otherwise
 */
bool tst_action::get_all_tst_config_keys(void) {
    int error;
    string msg, ststress;
    bool bsts = true;

    if ((error =
      property_get(RVS_CONF_TARGET_TEMP_KEY, &tst_target_temp))) {
      switch (error) {
        case 1:
          msg = "invalid '" + std::string(RVS_CONF_TARGET_TEMP_KEY) +
              "' key value " + ststress;
          rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
          break;

        case 2:
          msg = "key '" + std::string(RVS_CONF_TARGET_TEMP_KEY) +
          "' was not found";
          rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
      }
      bsts = false;
    }

    if ((error =
      property_get(RVS_CONF_THROTTLE_TEMP_KEY, &tst_throttle_temp))) {
      switch (error) {
        case 1:
          msg = "invalid '" + std::string(RVS_CONF_THROTTLE_TEMP_KEY) +
              "' key value " + ststress;
          rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
          break;

        case 2:
          msg = "key '" + std::string(RVS_CONF_THROTTLE_TEMP_KEY) +
          "' was not found";
          rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
      }
      bsts = false;
    }

    if (property_get_int<uint64_t>(RVS_CONF_RAMP_INTERVAL_KEY,
      &tst_ramp_interval, TST_DEFAULT_RAMP_INTERVAL)) {
      msg = "invalid '" + std::string(RVS_CONF_RAMP_INTERVAL_KEY)
      + "' key value";
      rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
      bsts = false;
    }

    if (property_get_int<uint64_t>(RVS_CONF_LOG_INTERVAL_KEY,
      &property_log_interval, TST_DEFAULT_LOG_INTERVAL)) {
      msg = "invalid '" + std::string(RVS_CONF_LOG_INTERVAL_KEY)
      + "' key value";
      rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
      bsts = false;
    }

    if (property_get_int<uint64_t>(RVS_CONF_SAMPLE_INTERVAL_KEY,
      &tst_sample_interval, TST_DEFAULT_SAMPLE_INTERVAL)) {
      msg = "invalid '" + std::string(RVS_CONF_SAMPLE_INTERVAL_KEY)
      + "' key value";
      rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
      bsts = false;
    }

    if (property_get_int<int>(RVS_CONF_MAX_VIOLATIONS_KEY,
      &tst_max_violations, TST_DEFAULT_MAX_VIOLATIONS)) {
      msg = "invalid '" + std::string(RVS_CONF_MAX_VIOLATIONS_KEY)
      + "' key value";
      rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
      bsts = false;
    }

    if (property_get<float>(RVS_CONF_TOLERANCE_KEY,
      &tst_tolerance, TST_DEFAULT_TOLERANCE)) {
      msg = "invalid '" + std::string(RVS_CONF_TOLERANCE_KEY)
      + "' key value";
      rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
      bsts = false;
    }

    if (property_get_int<uint64_t>(RVS_CONF_MATRIX_SIZE_KEY,
      &tst_matrix_size, TST_DEFAULT_MATRIX_SIZE)) {
      msg = "invalid '" + std::string(RVS_CONF_MATRIX_SIZE_KEY)
      + "' key value";
      rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
      bsts = false;
    }

    if (property_get<std::string>(RVS_CONF_TST_OPS_TYPE, &tst_ops_type, TST_DEFAULT_OPS_TYPE)) {
      msg = "invalid '" + std::string(RVS_CONF_TST_OPS_TYPE)
      + "' key value";
      rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
      bsts = false;
    }

    error = property_get_int<uint64_t>(RVS_CONF_MATRIX_SIZE_KEYA, &tst_matrix_size_a, TST_DEFAULT_MATRIX_SIZE);
    if (error == 1) {
        msg = "invalid '" +
        std::string(RVS_CONF_MATRIX_SIZE_KEYA) + "' key value";
        rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
        bsts = false;
    }

    error = property_get_int<uint64_t>(RVS_CONF_MATRIX_SIZE_KEYB, &tst_matrix_size_b, TST_DEFAULT_MATRIX_SIZE);
    if (error == 1) {
        msg = "invalid '" +
        std::string(RVS_CONF_MATRIX_SIZE_KEYB) + "' key value";
        rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
        bsts = false;
    }

    error = property_get_int<uint64_t>(RVS_CONF_MATRIX_SIZE_KEYC, &tst_matrix_size_c, TST_DEFAULT_MATRIX_SIZE);
    if (error == 1) {
        msg = "invalid '" +
        std::string(RVS_CONF_MATRIX_SIZE_KEYC) + "' key value";
        rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
        bsts = false;
    }

    error = property_get_int<int>(RVS_CONF_TRANS_A, &tst_trans_a, TST_DEFAULT_TRANS_A);
    if (error == 1) {
        msg = "invalid '" +
        std::string(RVS_CONF_TRANS_A) + "' key value";
        rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
        bsts = false;
    }

    error = property_get_int<int>(RVS_CONF_TRANS_B, &tst_trans_b, TST_DEFAULT_TRANS_B);
    if (error == 1) {
        msg = "invalid '" +
        std::string(RVS_CONF_TRANS_B) + "' key value";
        rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
        bsts = false;
    }

    error = property_get<float>(RVS_CONF_ALPHA_VAL, &tst_alpha_val, TST_DEFAULT_ALPHA_VAL);
    if (error == 1) {
        msg = "invalid '" +
        std::string(RVS_CONF_ALPHA_VAL) + "' key value";
        rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
        bsts = false;
    }

    error = property_get<float>(RVS_CONF_BETA_VAL, &tst_beta_val, TST_DEFAULT_BETA_VAL);
    if (error == 1) {
        msg = "invalid '" +
        std::string(RVS_CONF_BETA_VAL) + "' key value";
        rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
        bsts = false;
    }

    error = property_get_int<int>(RVS_CONF_LDA_OFFSET, &tst_lda_offset, TST_DEFAULT_LDA_OFFSET);
    if (error == 1) {
        msg = "invalid '" +
        std::string(RVS_CONF_LDA_OFFSET) + "' key value";
        rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
        bsts = false;
    }

    error = property_get_int<int>(RVS_CONF_LDB_OFFSET, &tst_ldb_offset, TST_DEFAULT_LDB_OFFSET);
    if (error == 1) {
        msg = "invalid '" +
        std::string(RVS_CONF_LDB_OFFSET) + "' key value";
        rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
        bsts = false;
    }

    error = property_get_int<int>(RVS_CONF_LDC_OFFSET, &tst_ldc_offset, TST_DEFAULT_LDC_OFFSET);
    if (error == 1) {
        msg = "invalid '" +
        std::string(RVS_CONF_LDC_OFFSET) + "' key value";
        rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
        bsts = false;
    }

    error = property_get_int<int>(RVS_CONF_LDD_OFFSET, &tst_ldd_offset, TST_DEFAULT_LDD_OFFSET);
    if (error == 1) {
        msg = "invalid '" +
        std::string(RVS_CONF_LDD_OFFSET) + "' key value";
        rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
        bsts = false;
    }

    error = property_get<bool>(RVS_CONF_TT_FLAG, &tst_tt_flag, TST_DEFAULT_TT_FLAG);
    if (error == 1) {
        msg = "invalid '" +
        std::string(RVS_CONF_TT_FLAG) + "' key value";
        rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
        bsts = false;
    }

    return bsts;
}


/**
 * @brief maps hip index to smi index
 * 
 */

void tst_action::hip_to_smi_indices(void) {
    int hip_num_gpu_devices;
    hipGetDeviceCount(&hip_num_gpu_devices);
    // map this to smi as only these are visible
    uint32_t smi_num_devices;
    uint64_t val_ui64;
    std::map<uint64_t, int> smi_map;

    rsmi_status_t err = rsmi_num_monitor_devices(&smi_num_devices);
    if( err == RSMI_STATUS_SUCCESS){
        for(auto i = 0; i < smi_num_devices; ++i){
            err = rsmi_dev_pci_id_get(i, &val_ui64);
            smi_map.insert({val_ui64, i});
        }
    }

    for (int i = 0; i < hip_num_gpu_devices; i++) {
        // get GPU device properties
         unsigned int pDom, pBus, pDev, pFun;
	 getBDF(i, pDom, pBus, pDev, pFun);
        // compute device location_id (needed to match this device
        // with one of those found while querying the pci bus
	uint64_t hip_dev_location_id = ( ( ((uint64_t)pDom & 0xffff ) << 32) |
            (((uint64_t) pBus & 0xff ) << 8) | (((uint64_t)pDev & 0x1f ) << 3)| ((uint64_t)pFun ) );
        if(smi_map.find(hip_dev_location_id) != smi_map.end()){
            hip_to_smi_idxs.insert({i, smi_map[hip_dev_location_id]});
        }
    }
}


/**
 * @brief runs the thermal stress test (TST).
 * @return true if no error occured, false otherwise
 */
bool tst_action::do_thermal_test(map<int, uint16_t> tst_gpus_device_index) {

    std::string  msg;
    uint32_t     dev_idx = 0;
    size_t       k = 0;
    int          gpuId;
    bool gpu_masking = false;    // if HIP_VISIBLE_DEVICES is set, this will be true
    int hip_num_gpu_devices;
    hipGetDeviceCount(&hip_num_gpu_devices);

    vector<TSTWorker> workers(tst_gpus_device_index.size());
    for (;;) {
        unsigned int i = 0;
        map<int, uint16_t>::iterator it;

        if (property_wait != 0)  // delay tst execution
            sleep(property_wait);

	// map hip indexes to smi indexes
	hip_to_smi_indices();

        TSTWorker::set_use_json(bjson);
        for (it = tst_gpus_device_index.begin(); it != tst_gpus_device_index.end(); ++it) {
            if(hip_to_smi_idxs.find(it->first) != hip_to_smi_idxs.end()){
                workers[i].set_smi_device_index(hip_to_smi_idxs[it->first]);
            } else{
                workers[i].set_smi_device_index(it->first);
            }
            gpuId = it->second;
            // set worker thread params
            workers[i].set_name(action_name);
            workers[i].set_action(*this);
            workers[i].set_gpu_id(it->second);
            workers[i].set_gpu_device_index(it->first);
            workers[i].set_run_wait_ms(property_wait);
            workers[i].set_run_duration_ms(property_duration);
            workers[i].set_ramp_interval(tst_ramp_interval);
            workers[i].set_log_interval(property_log_interval);
            workers[i].set_sample_interval(tst_sample_interval);
            workers[i].set_max_violations(tst_max_violations);
            workers[i].set_target_temp(tst_target_temp);
            workers[i].set_throttle_temp(tst_throttle_temp);
            workers[i].set_tolerance(tst_tolerance);
            workers[i].set_matrix_size_a(tst_matrix_size_a);
            workers[i].set_matrix_size_b(tst_matrix_size_b);
            workers[i].set_matrix_size_c(tst_matrix_size_c);
            workers[i].set_tst_ops_type(tst_ops_type);
            workers[i].set_matrix_transpose_a(tst_trans_a);
            workers[i].set_matrix_transpose_b(tst_trans_b);
            workers[i].set_alpha_val(tst_alpha_val);
            workers[i].set_beta_val(tst_beta_val);
            workers[i].set_lda_offset(tst_lda_offset);
            workers[i].set_ldb_offset(tst_ldb_offset);
            workers[i].set_ldc_offset(tst_ldc_offset);
            workers[i].set_ldd_offset(tst_ldd_offset);
            workers[i].set_tt_flag(tst_tt_flag);

            i++;
        }

        if (property_parallel) {
            for (i = 0; i < tst_gpus_device_index.size(); i++)
                workers[i].start();
            // join threads
            for (i = 0; i < tst_gpus_device_index.size(); i++) 
                workers[i].join();

        } else {
            for (i = 0; i < tst_gpus_device_index.size(); i++) {
                workers[i].start();
                workers[i].join();

                // check if stop signal was received
                if (rvs::lp::Stopping()) {
                    return false;
                }
            }
        }


        msg = "[" + action_name + "] " + MODULE_NAME + " " + std::to_string(gpuId) + " Shutting down rocm-smi  ";
        rvs::lp::Log(msg, rvs::loginfo);


        // check if stop signal was received
        if (rvs::lp::Stopping())
            return false;

        if (property_count == ++k) {
            break;
        }
    }


    msg = "[" + action_name + "] " + MODULE_NAME + " " + std::to_string(gpuId) + " Done with tst test ";
    rvs::lp::Log(msg, rvs::loginfo);

    sleep(1000);

    return true;
}

/**
 * @brief gets the number of ROCm compatible AMD GPUs
 * @return run number of GPUs
 */
int tst_action::get_num_amd_gpu_devices(void) {
    int hip_num_gpu_devices;
    string msg;

    hipGetDeviceCount(&hip_num_gpu_devices);
    return hip_num_gpu_devices;
}



/**
 * @brief gets all selected GPUs and starts the worker threads
 * @return run result
 */
int tst_action::get_all_selected_gpus(void) {
    int hip_num_gpu_devices;
    bool amd_gpus_found = false;
    map<int, uint16_t> tst_gpus_device_index;
    std::string msg;
    std::stringstream msg_stream;

    hipGetDeviceCount(&hip_num_gpu_devices);
    if (hip_num_gpu_devices < 1)
        return hip_num_gpu_devices;
    rsmi_init(0);
    // find compatible GPUs to run tst tests
    amd_gpus_found = fetch_gpu_list(hip_num_gpu_devices, tst_gpus_device_index,
        property_device, property_device_id, property_device_all,
        property_device_index, property_device_index_all, true);  // MCM checks
    if(!amd_gpus_found){

        msg = "No devices match criteria from the test configuation.";
        rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
        rsmi_shut_down();
        if (bjson) {
          unsigned int sec;
          unsigned int usec;
          rvs::lp::get_ticks(&sec, &usec);
          void *json_root_node = rvs::lp::LogRecordCreate(MODULE_NAME,
            action_name.c_str(), rvs::logerror, sec, usec, true);
          if (!json_root_node) {
            // log the error
            string msg = std::string(JSON_CREATE_NODE_ERROR);
            rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
            return -1;
         }

        rvs::lp::AddString(json_root_node, "ERROR","No AMD compatible GPU found!");
        rvs::lp::LogRecordFlush(json_root_node, rvs::logerror);
       }

        return 0;
    }

    int tst_res = 0;
    if(do_thermal_test(tst_gpus_device_index))
        tst_res = 0;
    else 
        tst_res = -1;
    rsmi_shut_down();
    return tst_res;
}


/**
 * @brief runs the whole TST logic
 * @return run result
 */
int tst_action::run(void) {
  string msg;
  rvs::action_result_t action_result;



  if (!get_all_common_config_keys())
    return -1;

  if (!get_all_tst_config_keys())
    return -1;

  if (property_duration > 0 && (property_duration < tst_ramp_interval)) {
    msg = std::string(RVS_CONF_DURATION_KEY) + "' cannot be less than '" +
      RVS_CONF_RAMP_INTERVAL_KEY + "'";
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
    return -1;
  }
  if(bjson){
        // add prelims for each action,
      json_add_primary_fields(std::string(MODULE_NAME), action_name);
   }

  auto res =  get_all_selected_gpus();
  // append end node to json
  if(bjson){
    rvs::lp::JsonActionEndNodeCreate();
  }

  action_result.state = rvs::actionstate::ACTION_COMPLETED;
  action_result.status = (!res) ? rvs::actionstatus::ACTION_SUCCESS : rvs::actionstatus::ACTION_FAILED;
  action_result.output = "TST Module action " + action_name + " completed";
  action_callback(&action_result);

  return res;
}

