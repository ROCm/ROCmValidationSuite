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
#include "include/iet_worker.h"
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


#define RVS_CONF_TARGET_POWER_KEY       "target_power"
#define RVS_CONF_RAMP_INTERVAL_KEY      "ramp_interval"
#define RVS_CONF_TOLERANCE_KEY          "tolerance"
#define RVS_CONF_MAX_VIOLATIONS_KEY     "max_violations"
#define RVS_CONF_SAMPLE_INTERVAL_KEY    "sample_interval"
#define RVS_CONF_LOG_INTERVAL_KEY       "log_interval"
#define RVS_CONF_MATRIX_SIZE_KEY        "matrix_size"
#define RVS_CONF_IET_OPS_TYPE           "ops_type"
#define RVS_CONF_MATRIX_SIZE_KEYA       "matrix_size_a"
#define RVS_CONF_MATRIX_SIZE_KEYB       "matrix_size_b"
#define RVS_CONF_MATRIX_SIZE_KEYC       "matrix_size_b"
#define RVS_CONF_IET_OPS_TYPE           "ops_type"
#define RVS_CONF_TRANS_A                "transa"
#define RVS_CONF_TRANS_B                "transb"
#define RVS_CONF_ALPHA_VAL              "alpha"
#define RVS_CONF_BETA_VAL               "beta"
#define RVS_CONF_LDA_OFFSET             "lda"
#define RVS_CONF_LDB_OFFSET             "ldb"
#define RVS_CONF_LDC_OFFSET             "ldc"
#define RVS_CONF_TP_FLAG                "targetpower_met"
#define RVS_TP_MESSAGE                  "target_power"
#define RVS_DTYPE_MESSAGE               "dtype"


#define MODULE_NAME                     "iet"
#define MODULE_NAME_CAPS                "IET"

#define IET_DEFAULT_RAMP_INTERVAL       5000
#define IET_DEFAULT_LOG_INTERVAL        1000
#define IET_DEFAULT_MAX_VIOLATIONS      0
#define IET_DEFAULT_TOLERANCE           0.1
#define IET_DEFAULT_SAMPLE_INTERVAL     100
#define IET_DEFAULT_MATRIX_SIZE         5760
#define RVS_DEFAULT_PARALLEL            false
#define RVS_DEFAULT_DURATION            500
#define IET_DEFAULT_OPS_TYPE            "sgemm"
#define IET_DEFAULT_TRANS_A             0
#define IET_DEFAULT_TRANS_B             1
#define IET_DEFAULT_ALPHA_VAL           1
#define IET_DEFAULT_BETA_VAL            1
#define IET_DEFAULT_LDA_OFFSET          0
#define IET_DEFAULT_LDB_OFFSET          0
#define IET_DEFAULT_LDC_OFFSET          0
#define IET_DEFAULT_TP_FLAG             false

#define IET_NO_COMPATIBLE_GPUS          "No AMD compatible GPU found!"
#define PCI_ALLOC_ERROR                 "pci_alloc() error"
#define FLOATING_POINT_REGEX            "^[0-9]*\\.?[0-9]+$"
#define JSON_CREATE_NODE_ERROR          "JSON cannot create node"

/**
 * @brief default class constructor
 */
iet_action::iet_action() {
}

/**
 * @brief class destructor
 */
iet_action::~iet_action() {
    property.clear();
}


/**
 * @brief reads all IET's related configuration keys from
 * the module's properties collection
 * @return true if no fatal error occured, false otherwise
 */
bool iet_action::get_all_iet_config_keys(void) {
    int error;
    string msg, ststress;
    bool bsts = true;

    if ((error =
      property_get(RVS_CONF_TARGET_POWER_KEY, &iet_target_power))) {
      switch (error) {
        case 1:
          msg = "invalid '" + std::string(RVS_CONF_TARGET_POWER_KEY) +
              "' key value " + ststress;
          rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
          break;

        case 2:
          msg = "key '" + std::string(RVS_CONF_TARGET_POWER_KEY) +
          "' was not found";
          rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
      }
      bsts = false;
    }

    if (property_get_int<uint64_t>(RVS_CONF_RAMP_INTERVAL_KEY,
      &iet_ramp_interval, IET_DEFAULT_RAMP_INTERVAL)) {
      msg = "invalid '" + std::string(RVS_CONF_RAMP_INTERVAL_KEY)
      + "' key value";
      rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
      bsts = false;
    }

    if (property_get_int<uint64_t>(RVS_CONF_LOG_INTERVAL_KEY,
      &property_log_interval, IET_DEFAULT_LOG_INTERVAL)) {
      msg = "invalid '" + std::string(RVS_CONF_LOG_INTERVAL_KEY)
      + "' key value";
      rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
      bsts = false;
    }

    if (property_get_int<uint64_t>(RVS_CONF_SAMPLE_INTERVAL_KEY,
      &iet_sample_interval, IET_DEFAULT_SAMPLE_INTERVAL)) {
      msg = "invalid '" + std::string(RVS_CONF_SAMPLE_INTERVAL_KEY)
      + "' key value";
      rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
      bsts = false;
    }

    if (property_get_int<int>(RVS_CONF_MAX_VIOLATIONS_KEY,
      &iet_max_violations, IET_DEFAULT_MAX_VIOLATIONS)) {
      msg = "invalid '" + std::string(RVS_CONF_MAX_VIOLATIONS_KEY)
      + "' key value";
      rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
      bsts = false;
    }

    if (property_get<float>(RVS_CONF_TOLERANCE_KEY,
      &iet_tolerance, IET_DEFAULT_TOLERANCE)) {
      msg = "invalid '" + std::string(RVS_CONF_TOLERANCE_KEY)
      + "' key value";
      rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
      bsts = false;
    }

    if (property_get_int<uint64_t>(RVS_CONF_MATRIX_SIZE_KEY,
      &iet_matrix_size, IET_DEFAULT_MATRIX_SIZE)) {
      msg = "invalid '" + std::string(RVS_CONF_MATRIX_SIZE_KEY)
      + "' key value";
      rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
      bsts = false;
    }

    if (property_get<std::string>(RVS_CONF_IET_OPS_TYPE, &iet_ops_type, IET_DEFAULT_OPS_TYPE)) {
      msg = "invalid '" + std::string(RVS_CONF_IET_OPS_TYPE)
      + "' key value";
      rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
      bsts = false;
    }

    error = property_get_int<uint64_t>(RVS_CONF_MATRIX_SIZE_KEYA, &iet_matrix_size_a, IET_DEFAULT_MATRIX_SIZE);
    if (error == 1) {
        msg = "invalid '" +
        std::string(RVS_CONF_MATRIX_SIZE_KEYA) + "' key value";
        rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
        bsts = false;
    }

    error = property_get_int<uint64_t>(RVS_CONF_MATRIX_SIZE_KEYB, &iet_matrix_size_b, IET_DEFAULT_MATRIX_SIZE);
    if (error == 1) {
        msg = "invalid '" +
        std::string(RVS_CONF_MATRIX_SIZE_KEYB) + "' key value";
        rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
        bsts = false;
    }

    error = property_get_int<uint64_t>(RVS_CONF_MATRIX_SIZE_KEYC, &iet_matrix_size_c, IET_DEFAULT_MATRIX_SIZE);
    if (error == 1) {
        msg = "invalid '" +
        std::string(RVS_CONF_MATRIX_SIZE_KEYC) + "' key value";
        rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
        bsts = false;
    }

    error = property_get_int<int>(RVS_CONF_TRANS_A, &iet_trans_a, IET_DEFAULT_TRANS_A);
    if (error == 1) {
        msg = "invalid '" +
        std::string(RVS_CONF_TRANS_A) + "' key value";
        rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
        bsts = false;
    }

    error = property_get_int<int>(RVS_CONF_TRANS_B, &iet_trans_b, IET_DEFAULT_TRANS_B);
    if (error == 1) {
        msg = "invalid '" +
        std::string(RVS_CONF_TRANS_B) + "' key value";
        rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
        bsts = false;
    }

    error = property_get<float>(RVS_CONF_ALPHA_VAL, &iet_alpha_val, IET_DEFAULT_ALPHA_VAL);
    if (error == 1) {
        msg = "invalid '" +
        std::string(RVS_CONF_ALPHA_VAL) + "' key value";
        rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
        bsts = false;
    }

    error = property_get<float>(RVS_CONF_BETA_VAL, &iet_beta_val, IET_DEFAULT_BETA_VAL);
    if (error == 1) {
        msg = "invalid '" +
        std::string(RVS_CONF_BETA_VAL) + "' key value";
        rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
        bsts = false;
    }

    error = property_get_int<int>(RVS_CONF_LDA_OFFSET, &iet_lda_offset, IET_DEFAULT_LDA_OFFSET);
    if (error == 1) {
        msg = "invalid '" +
        std::string(RVS_CONF_LDA_OFFSET) + "' key value";
        rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
        bsts = false;
    }

    error = property_get_int<int>(RVS_CONF_LDB_OFFSET, &iet_ldb_offset, IET_DEFAULT_LDB_OFFSET);
    if (error == 1) {
        msg = "invalid '" +
        std::string(RVS_CONF_LDB_OFFSET) + "' key value";
        rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
        bsts = false;
    }

    error = property_get_int<int>(RVS_CONF_LDC_OFFSET, &iet_ldc_offset, IET_DEFAULT_LDC_OFFSET);
    if (error == 1) {
        msg = "invalid '" +
        std::string(RVS_CONF_LDC_OFFSET) + "' key value";
        rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
        bsts = false;
    }

    error = property_get<bool>(RVS_CONF_TP_FLAG, &iet_tp_flag, IET_DEFAULT_TP_FLAG);
    if (error == 1) {
        msg = "invalid '" +
        std::string(RVS_CONF_TP_FLAG) + "' key value";
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
bool iet_action::get_all_common_config_keys(void) {
    string msg, sdevid, sdev;
    int error;
    bool bsts = true;

    // get <device> property value (a list of gpu id)
    if ((error = property_get_device())) {
      switch (error) {
      case 1:
        msg = "Invalid 'device' key value.";
        break;
      case 2:
        msg = "Missing 'device' key.";
        break;
      }
      rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
      bsts = false;
    }

    // get the <deviceid> property value if provided
    if (property_get_int<uint16_t>(RVS_CONF_DEVICEID_KEY,
                                  &property_device_id, 0u)) {
      msg = "Invalid 'deviceid' key value.";
      rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
      bsts = false;
    }

    // get <device_index> property value (a list of device indexes)
    if (int sts = property_get_device_index()) {
      switch (sts) {
      case 1:
        msg = "Invalid 'device_index' key value.";
        break;
      case 2:
        msg = "Missing 'device_index' key.";
        break;
      }
      // default set as true
      property_device_index_all = true;
      rvs::lp::Log(msg, rvs::loginfo);
    }

    // get the other action/IET related properties
    if (property_get(RVS_CONF_PARALLEL_KEY, &property_parallel, false)) {
      msg = "invalid '" +
              std::string(RVS_CONF_PARALLEL_KEY) + "' key value";
      rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
      bsts = false;
    }

    error = property_get_int<uint64_t>
    (RVS_CONF_COUNT_KEY, &property_count, DEFAULT_COUNT);
    if (error == 1) {
      msg = "invalid '" +
              std::string(RVS_CONF_COUNT_KEY) + "' key value";
      rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
      bsts = false;
    }

    error = property_get_int<uint64_t>
    (RVS_CONF_WAIT_KEY, &property_wait, DEFAULT_WAIT);
    if (error == 1) {
      msg = "invalid '" +
              std::string(RVS_CONF_WAIT_KEY) + "' key value";
      rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
      bsts = false;
    }

    error = property_get_int<uint64_t>
    (RVS_CONF_DURATION_KEY, &property_duration);
    if (error == 1) {
      msg = "invalid '" +
              std::string(RVS_CONF_DURATION_KEY) + "' key value";
      rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
      bsts = false;
    }

    return bsts;
}

/**
 * @brief maps hip index to smi index
 * 
 */

void iet_action::hip_to_smi_indices(void) {
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
        hipDeviceProp_t props;
        hipGetDeviceProperties(&props, i);

        // compute device location_id (needed to match this device
        // with one of those found while querying the pci bus
        uint16_t hip_dev_location_id =
            ((((uint16_t) (props.pciBusID)) << 8) | (((uint16_t)(props.pciDeviceID)) << 3) );
        if(smi_map.find(hip_dev_location_id) != smi_map.end()){
            hip_to_smi_idxs.insert({i, smi_map[hip_dev_location_id]});
        }
    }
}


/**
 * @brief runs the edp test
 * @return true if no error occured, false otherwise
 */
bool iet_action::do_edp_test(map<int, uint16_t> iet_gpus_device_index) {
    std::string  msg;
    uint32_t     dev_idx = 0;
    size_t       k = 0;
    int          gpuId;
    bool gpu_masking = false;    // if HIP_VISIBLE_DEVICES is set, this will be true
    int hip_num_gpu_devices;

    hipGetDeviceCount(&hip_num_gpu_devices);

    vector<IETWorker> workers(iet_gpus_device_index.size());
    for (;;) {
        unsigned int i = 0;
        map<int, uint16_t>::iterator it;

        if (property_wait != 0)  // delay iet execution
            sleep(property_wait);

        rsmi_init(0);

        uint32_t smi_num_devices;
        rsmi_status_t err = rsmi_num_monitor_devices(&smi_num_devices);
        if(smi_num_devices != hip_num_gpu_devices)
            gpu_masking = true;
        if(gpu_masking){  // this is the case when using HIP_VISIBLE_DEVICES variable to modify GPU visibility
            // smi output wont be affected by the flag and hence indices should be appropriately used.
            hip_to_smi_indices();
        }

        IETWorker::set_use_json(bjson);
        for (it = iet_gpus_device_index.begin(); it != iet_gpus_device_index.end(); ++it) {
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
            workers[i].set_pwr_device_id(dev_idx++);
            workers[i].set_run_wait_ms(property_wait);
            workers[i].set_run_duration_ms(property_duration);
            workers[i].set_ramp_interval(iet_ramp_interval);
            workers[i].set_log_interval(property_log_interval);
            workers[i].set_sample_interval(iet_sample_interval);
            workers[i].set_max_violations(iet_max_violations);
            workers[i].set_target_power(iet_target_power);
            workers[i].set_tolerance(iet_tolerance);
            workers[i].set_matrix_size_a(iet_matrix_size_a);
            workers[i].set_matrix_size_b(iet_matrix_size_b);
            workers[i].set_matrix_size_c(iet_matrix_size_c);
            workers[i].set_iet_ops_type(iet_ops_type);
            workers[i].set_matrix_transpose_a(iet_trans_a);
            workers[i].set_matrix_transpose_b(iet_trans_b);
            workers[i].set_alpha_val(iet_alpha_val);
            workers[i].set_beta_val(iet_beta_val);
            workers[i].set_lda_offset(iet_lda_offset);
            workers[i].set_ldb_offset(iet_ldb_offset);
            workers[i].set_ldc_offset(iet_ldc_offset);
            workers[i].set_tp_flag(iet_tp_flag);

            i++;
        }

        if (property_parallel) {
            for (i = 0; i < iet_gpus_device_index.size(); i++)
                workers[i].start();
            // join threads
            for (i = 0; i < iet_gpus_device_index.size(); i++) 
                workers[i].join();

        } else {
            for (i = 0; i < iet_gpus_device_index.size(); i++) {
                workers[i].start();
                workers[i].join();

                // check if stop signal was received
                if (rvs::lp::Stopping()) {
                    rsmi_shut_down();
                    return false;
                }
            }
        }


        msg = "[" + action_name + "] " + MODULE_NAME + " " + std::to_string(gpuId) + " Shutting down rocm-smi  ";
        rvs::lp::Log(msg, rvs::loginfo);

        rsmi_shut_down(); 

        // check if stop signal was received
        if (rvs::lp::Stopping())
            return false;

        if (property_count == ++k) {
            break;
        }
    }


    msg = "[" + action_name + "] " + MODULE_NAME + " " + std::to_string(gpuId) + " Done with iet test ";
    rvs::lp::Log(msg, rvs::loginfo);

    sleep(1000);

    return true;
}

/**
 * @brief gets the number of ROCm compatible AMD GPUs
 * @return run number of GPUs
 */
int iet_action::get_num_amd_gpu_devices(void) {
    int hip_num_gpu_devices;
    string msg;

    hipGetDeviceCount(&hip_num_gpu_devices);
    return hip_num_gpu_devices;
}

/**
 * @brief retrieves the GPU identification data and adds it to the list of 
 * those that will run the EDPp test
 * @param dev_location_id GPU device location ID
 * @param gpu_id GPU's ID as exported by KFD
 * @param hip_num_gpu_devices number of GPU devices (as reported by HIP API)
 * @return true if all info could be retrieved and the gpu was successfully to
 * the EDPp test list, false otherwise
 */
bool iet_action::add_gpu_to_edpp_list(uint16_t dev_location_id, int32_t gpu_id,
                                  int hip_num_gpu_devices) {
    for (int i = 0; i < hip_num_gpu_devices; i++) {
        // get GPU device properties
        hipDeviceProp_t props;
        hipGetDeviceProperties(&props, i);

        // compute device location_id (needed to match this device
        // with one of those found while querying the pci bus
        uint16_t hip_dev_location_id =
                ((((uint16_t) (props.pciBusID)) << 8) | (((uint16_t)(props.pciDeviceID)) << 3) );
        if (hip_dev_location_id == dev_location_id) {
            gpu_hwmon_info cgpu_info;
            cgpu_info.hip_gpu_deviceid = i;
            cgpu_info.gpu_id = gpu_id;
            cgpu_info.bdf_id = hip_dev_location_id;
            edpp_gpus.push_back(cgpu_info);

            return true;
        }
    }

    return false;
}

/**
 * @brief flushes target power and dtype fields to json file
 * @return
 */

void iet_action::json_add_primary_fields(){
    if (rvs::lp::JsonActionStartNodeCreate(MODULE_NAME, action_name.c_str())){
        rvs::lp::Err("json start create failed", MODULE_NAME_CAPS, action_name);
        return;
    }
    void *json_node = json_node_create(std::string(MODULE_NAME),
                        action_name.c_str(), rvs::loginfo);
    if(json_node){
            rvs::lp::AddString(json_node,RVS_TP_MESSAGE, std::to_string(iet_target_power));
            rvs::lp::LogRecordFlush(json_node, rvs::loginfo);
            json_node = nullptr;
    }
    json_node = json_node_create(std::string(MODULE_NAME),
                        action_name.c_str(), rvs::loginfo);
    if(json_node){
            rvs::lp::AddString(json_node,RVS_DTYPE_MESSAGE, iet_ops_type);
            rvs::lp::LogRecordFlush(json_node, rvs::loginfo);
    }

}

/**
 * @brief gets all selected GPUs and starts the worker threads
 * @return run result
 */
int iet_action::get_all_selected_gpus(void) {
    int hip_num_gpu_devices;
    bool amd_gpus_found = false;
    map<int, uint16_t> iet_gpus_device_index;
    std::string msg;
    std::stringstream msg_stream;

    hipGetDeviceCount(&hip_num_gpu_devices);
    if (hip_num_gpu_devices < 1)
        return hip_num_gpu_devices;
    // find compatible GPUs to run edp tests
    amd_gpus_found = fetch_gpu_list(hip_num_gpu_devices, iet_gpus_device_index,
        property_device, property_device_all,
        property_device_id,
        property_device_index, property_device_index_all, true); // MCM checks
    if(!amd_gpus_found){
      msg = "No devices match criteria from the test configuation.";
      rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
      return 1;
    }

    if(bjson){
        // add prelims for each action, dtype and target stress
        json_add_primary_fields();
    }
    int iet_res = 0;
    if(do_edp_test(iet_gpus_device_index))
        iet_res = 0;
    else 
        iet_res = -1;
    // append end node to json
    if(bjson){
        rvs::lp::JsonActionEndNodeCreate();
    }
    return iet_res;
}


/**
 * @brief runs the whole IET logic
 * @return run result
 */
int iet_action::run(void) {
  string msg;
  rvs::action_result_t action_result;

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

  if (!get_all_iet_config_keys())
    return -1;

  if (property_duration > 0 && (property_duration < iet_ramp_interval)) {
    msg = std::string(RVS_CONF_DURATION_KEY) + "' cannot be less than '" +
      RVS_CONF_RAMP_INTERVAL_KEY + "'";
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
    return -1;
  }

  auto res =  get_all_selected_gpus();

  action_result.state = rvs::actionstate::ACTION_COMPLETED;
  action_result.status = (!res) ? rvs::actionstatus::ACTION_SUCCESS : rvs::actionstatus::ACTION_FAILED;
  action_result.output = "IET Module action " + action_name + " completed";
  action_callback(&action_result);

  return res;
}

void iet_action::cleanup_logs(){
  rvs::lp::JsonEndNodeCreate();
}
