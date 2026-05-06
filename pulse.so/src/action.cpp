/********************************************************************************
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <map>
#include <memory>
#include <barrier>
#include <atomic>

#ifdef __cplusplus
extern "C" {
#endif
#include <pci/pci.h>
#ifdef __cplusplus
}
#endif

#define __HIP_PLATFORM_HCC__
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"

#include "include/rvs_key_def.h"
#include "include/pulse_worker.h"
#include "include/gpu_util.h"
#include "include/rvs_util.h"
#include "include/rvs_module.h"
#include "include/rvsactionbase.h"
#include "include/rvsloglp.h"
#include "include/rsmi_util.h"

using std::string;
using std::vector;
using std::map;

#define RVS_CONF_PULSE_RATE_KEY           "pulse_rate"
#define RVS_CONF_HIGH_PHASE_RATIO_KEY     "high_phase_ratio"
#define RVS_CONF_TOLERANCE_KEY            "tolerance"
#define RVS_CONF_MATRIX_SIZE_KEY          "matrix_size"
#define RVS_CONF_OPS_TYPE_KEY             "ops_type"
#define RVS_CONF_DATA_TYPE_KEY            "data_type"
#define RVS_CONF_OUT_DATA_TYPE_KEY        "out_data_type"
#define RVS_CONF_TRANS_A_KEY              "transa"
#define RVS_CONF_TRANS_B_KEY              "transb"
#define RVS_CONF_ALPHA_VAL_KEY            "alpha"
#define RVS_CONF_BETA_VAL_KEY             "beta"
#define RVS_CONF_LDA_OFFSET_KEY           "lda"
#define RVS_CONF_LDB_OFFSET_KEY           "ldb"
#define RVS_CONF_LDC_OFFSET_KEY           "ldc"
#define RVS_CONF_LDD_OFFSET_KEY           "ldd"
#define RVS_CONF_WORKLOAD_ITERS_KEY       "workload_iterations"
#define RVS_CONF_HALT_ON_ERROR_KEY        "halt_on_error"
#define RVS_CONF_GPU_SYNC_WAIT_KEY        "gpu_sync_wait"
#define RVS_CONF_VERIFY_MODE_KEY          "verify_mode"
#define RVS_CONF_HOT_CALLS_KEY            "hot_calls"
#define RVS_CONF_MATRIX_INIT_KEY          "matrix_init"
#define RVS_CONF_BLAS_SOURCE_KEY          "blas_source"
#define RVS_CONF_COMPUTE_TYPE_KEY         "compute_type"
#define RVS_CONF_MAX_TEMP_C_KEY           "max_temp_c"

#define PULSE_DEFAULT_RATE                2
#define PULSE_DEFAULT_HIGH_PHASE_RATIO    0.5f
#define PULSE_DEFAULT_TOLERANCE           10.0f
#define PULSE_DEFAULT_MATRIX_SIZE         4096
#define PULSE_DEFAULT_OPS_TYPE            "sgemm"
#define PULSE_DEFAULT_DATA_TYPE           ""
#define PULSE_DEFAULT_OUT_DATA_TYPE       ""
#define PULSE_DEFAULT_TRANS_A             0
#define PULSE_DEFAULT_TRANS_B             1
#define PULSE_DEFAULT_ALPHA_VAL           2.0f
#define PULSE_DEFAULT_BETA_VAL            -1.0f
#define PULSE_DEFAULT_LDA_OFFSET          0
#define PULSE_DEFAULT_LDB_OFFSET          0
#define PULSE_DEFAULT_LDC_OFFSET          0
#define PULSE_DEFAULT_LDD_OFFSET          0
#define PULSE_DEFAULT_WORKLOAD_ITERS      128
#define PULSE_DEFAULT_HALT_ON_ERROR       false
#define PULSE_DEFAULT_GPU_SYNC_WAIT       10000
#define PULSE_DEFAULT_VERIFY_MODE         "diff"
#define PULSE_DEFAULT_SAMPLE_INTERVAL     100
#define PULSE_DEFAULT_HOT_CALLS           1
#define PULSE_DEFAULT_MATRIX_INIT         "default"
#define PULSE_DEFAULT_BLAS_SOURCE         "rocblas"
#define PULSE_DEFAULT_COMPUTE_TYPE        "fp32_r"
#define PULSE_DEFAULT_MAX_TEMP_C          105.0f

#define PULSE_NO_COMPATIBLE_GPUS          "No AMD compatible GPU found!"
#define JSON_CREATE_NODE_ERROR            "JSON cannot create node"

static constexpr auto MODULE_NAME = "pulse";
static constexpr auto MODULE_NAME_CAPS = "PULSE";

// Beta banner shown once at the start of every pulse action invocation.
// Plain ASCII so it survives non-UTF-8 terminals, log scrapers, and CI capture.
static const char* kPulseBetaBanner =
"\n"
"##############################################################################\n"
"#                                                                            #\n"
"#                  *** PULSE STRESS TEST - BETA VERSION ***                  #\n"
"#                                                                            #\n"
"#   This pulse test is a BETA version and is NOT to be used in               #\n"
"#   production environments. Pass/fail criteria are still being tuned.       #\n"
"#                                                                            #\n"
"##############################################################################\n";

pulse_action::pulse_action() {
  module_name = MODULE_NAME;
  pulse_max_temp_c = PULSE_DEFAULT_MAX_TEMP_C;
}

pulse_action::~pulse_action() {
  property.clear();
}

bool pulse_action::get_all_pulse_config_keys(void) {
  int error;
  string msg;
  bool bsts = true;

  if (property_get_int<int>(RVS_CONF_PULSE_RATE_KEY,
        &pulse_rate, PULSE_DEFAULT_RATE)) {
    msg = "invalid '" + std::string(RVS_CONF_PULSE_RATE_KEY)
      + "' key value";
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
    bsts = false;
  }

  if (property_get<float>(RVS_CONF_HIGH_PHASE_RATIO_KEY,
        &high_phase_ratio, PULSE_DEFAULT_HIGH_PHASE_RATIO)) {
    msg = "invalid '" + std::string(RVS_CONF_HIGH_PHASE_RATIO_KEY)
      + "' key value";
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
    bsts = false;
  }

  if (property_get<float>(RVS_CONF_TOLERANCE_KEY,
        &pulse_tolerance, PULSE_DEFAULT_TOLERANCE)) {
    msg = "invalid '" + std::string(RVS_CONF_TOLERANCE_KEY)
      + "' key value";
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
    bsts = false;
  }

  if (property_get_int<uint64_t>(RVS_CONF_MATRIX_SIZE_KEY,
        &pulse_matrix_size, (uint64_t)PULSE_DEFAULT_MATRIX_SIZE)) {
    msg = "invalid '" + std::string(RVS_CONF_MATRIX_SIZE_KEY)
      + "' key value";
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
    bsts = false;
  }

  if (property_get<std::string>(RVS_CONF_OPS_TYPE_KEY,
        &pulse_ops_type, std::string(PULSE_DEFAULT_OPS_TYPE))) {
    msg = "invalid '" + std::string(RVS_CONF_OPS_TYPE_KEY)
      + "' key value";
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
    bsts = false;
  }

  if (property_get<std::string>(RVS_CONF_DATA_TYPE_KEY,
        &pulse_data_type, std::string(PULSE_DEFAULT_DATA_TYPE))) {
    msg = "invalid '" + std::string(RVS_CONF_DATA_TYPE_KEY)
      + "' key value";
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
    bsts = false;
  }

  if (property_get<std::string>(RVS_CONF_OUT_DATA_TYPE_KEY,
        &pulse_out_data_type, std::string(PULSE_DEFAULT_OUT_DATA_TYPE))) {
    msg = "invalid '" + std::string(RVS_CONF_OUT_DATA_TYPE_KEY)
      + "' key value";
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
    bsts = false;
  }

  error = property_get_int<int>(RVS_CONF_TRANS_A_KEY,
      &pulse_trans_a, PULSE_DEFAULT_TRANS_A);
  if (error == 1) {
    msg = "invalid '" + std::string(RVS_CONF_TRANS_A_KEY) + "' key value";
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
    bsts = false;
  }

  error = property_get_int<int>(RVS_CONF_TRANS_B_KEY,
      &pulse_trans_b, PULSE_DEFAULT_TRANS_B);
  if (error == 1) {
    msg = "invalid '" + std::string(RVS_CONF_TRANS_B_KEY) + "' key value";
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
    bsts = false;
  }

  error = property_get<float>(RVS_CONF_ALPHA_VAL_KEY,
      &pulse_alpha_val, PULSE_DEFAULT_ALPHA_VAL);
  if (error == 1) {
    msg = "invalid '" + std::string(RVS_CONF_ALPHA_VAL_KEY) + "' key value";
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
    bsts = false;
  }

  error = property_get<float>(RVS_CONF_BETA_VAL_KEY,
      &pulse_beta_val, PULSE_DEFAULT_BETA_VAL);
  if (error == 1) {
    msg = "invalid '" + std::string(RVS_CONF_BETA_VAL_KEY) + "' key value";
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
    bsts = false;
  }

  error = property_get_int<int>(RVS_CONF_LDA_OFFSET_KEY,
      &pulse_lda_offset, PULSE_DEFAULT_LDA_OFFSET);
  if (error == 1) {
    msg = "invalid '" + std::string(RVS_CONF_LDA_OFFSET_KEY) + "' key value";
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
    bsts = false;
  }

  error = property_get_int<int>(RVS_CONF_LDB_OFFSET_KEY,
      &pulse_ldb_offset, PULSE_DEFAULT_LDB_OFFSET);
  if (error == 1) {
    msg = "invalid '" + std::string(RVS_CONF_LDB_OFFSET_KEY) + "' key value";
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
    bsts = false;
  }

  error = property_get_int<int>(RVS_CONF_LDC_OFFSET_KEY,
      &pulse_ldc_offset, PULSE_DEFAULT_LDC_OFFSET);
  if (error == 1) {
    msg = "invalid '" + std::string(RVS_CONF_LDC_OFFSET_KEY) + "' key value";
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
    bsts = false;
  }

  error = property_get_int<int>(RVS_CONF_LDD_OFFSET_KEY,
      &pulse_ldd_offset, PULSE_DEFAULT_LDD_OFFSET);
  if (error == 1) {
    msg = "invalid '" + std::string(RVS_CONF_LDD_OFFSET_KEY) + "' key value";
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
    bsts = false;
  }

  if (property_get_int<int>(RVS_CONF_WORKLOAD_ITERS_KEY,
        &pulse_workload_iterations, PULSE_DEFAULT_WORKLOAD_ITERS)) {
    msg = "invalid '" + std::string(RVS_CONF_WORKLOAD_ITERS_KEY)
      + "' key value";
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
    bsts = false;
  }

  error = property_get<bool>(RVS_CONF_HALT_ON_ERROR_KEY,
      &pulse_halt_on_error, PULSE_DEFAULT_HALT_ON_ERROR);
  if (error == 1) {
    msg = "invalid '" + std::string(RVS_CONF_HALT_ON_ERROR_KEY)
      + "' key value";
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
    bsts = false;
  }

  if (property_get_int<int>(RVS_CONF_GPU_SYNC_WAIT_KEY,
        &pulse_gpu_sync_wait, PULSE_DEFAULT_GPU_SYNC_WAIT)) {
    msg = "invalid '" + std::string(RVS_CONF_GPU_SYNC_WAIT_KEY)
      + "' key value";
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
    bsts = false;
  }

  if (property_get<std::string>(RVS_CONF_VERIFY_MODE_KEY,
        &pulse_verify_mode, std::string(PULSE_DEFAULT_VERIFY_MODE))) {
    msg = "invalid '" + std::string(RVS_CONF_VERIFY_MODE_KEY)
      + "' key value";
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
    bsts = false;
  }

  if (property_get_int<uint64_t>(RVS_CONF_SAMPLE_INTERVAL_KEY,
        &pulse_sample_interval, (uint64_t)PULSE_DEFAULT_SAMPLE_INTERVAL)) {
    msg = "invalid '" + std::string(RVS_CONF_SAMPLE_INTERVAL_KEY)
      + "' key value";
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
    bsts = false;
  }

  if (property_get_int<uint64_t>(RVS_CONF_HOT_CALLS_KEY,
        &pulse_hot_calls, (uint64_t)PULSE_DEFAULT_HOT_CALLS)) {
    msg = "invalid '" + std::string(RVS_CONF_HOT_CALLS_KEY)
      + "' key value";
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
    bsts = false;
  }

  if (property_get<std::string>(RVS_CONF_MATRIX_INIT_KEY,
        &pulse_matrix_init, std::string(PULSE_DEFAULT_MATRIX_INIT))) {
    msg = "invalid '" + std::string(RVS_CONF_MATRIX_INIT_KEY)
      + "' key value";
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
    bsts = false;
  }

  if (property_get<std::string>(RVS_CONF_BLAS_SOURCE_KEY,
        &pulse_blas_source, std::string(PULSE_DEFAULT_BLAS_SOURCE))) {
    msg = "invalid '" + std::string(RVS_CONF_BLAS_SOURCE_KEY)
      + "' key value";
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
    bsts = false;
  }

  if (property_get<std::string>(RVS_CONF_COMPUTE_TYPE_KEY,
        &pulse_compute_type, std::string(PULSE_DEFAULT_COMPUTE_TYPE))) {
    msg = "invalid '" + std::string(RVS_CONF_COMPUTE_TYPE_KEY)
      + "' key value";
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
    bsts = false;
  }

  if (property_get<float>(RVS_CONF_MAX_TEMP_C_KEY,
        &pulse_max_temp_c, PULSE_DEFAULT_MAX_TEMP_C)) {
    msg = "invalid '" + std::string(RVS_CONF_MAX_TEMP_C_KEY)
      + "' key value";
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
    bsts = false;
  }
  if (pulse_max_temp_c < 0.0f ||
      (pulse_max_temp_c > 0.0f && pulse_max_temp_c > 200.0f)) {
    msg = "'" + std::string(RVS_CONF_MAX_TEMP_C_KEY)
      + "' must be 0 to disable the thermal check, or a limit in (0, 200] °C";
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
    bsts = false;
  }

  if (pulse_blas_source != "rocblas" && pulse_blas_source != "hipblaslt") {
    msg = "'" + std::string(RVS_CONF_BLAS_SOURCE_KEY)
      + "' must be 'rocblas' or 'hipblaslt'";
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
    bsts = false;
  }

  // hipBLASLt needs explicit matrix data types; rocBLAS can rely on ops_type alone.
  if (pulse_blas_source == "hipblaslt" && pulse_data_type.empty()) {
    if (pulse_ops_type == "sgemm") {
      pulse_data_type = "fp32_r";
    } else if (pulse_ops_type == "dgemm") {
      pulse_data_type = "fp64_r";
    } else if (pulse_ops_type == "hgemm") {
      pulse_data_type = "fp16_r";
    } else {
      msg = "hipblaslt requires 'data_type' in the action when 'ops_type' is not "
        "sgemm, dgemm, or hgemm";
      rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
      bsts = false;
    }
  }

  // Default compute_type is fp32_r; double GEMM with hipBLASLt needs fp64 compute.
  if (pulse_blas_source == "hipblaslt" && pulse_ops_type == "dgemm" &&
      pulse_compute_type == std::string(PULSE_DEFAULT_COMPUTE_TYPE)) {
    pulse_compute_type = "fp64_r";
  }

  if (high_phase_ratio < 0.0f || high_phase_ratio > 1.0f) {
    msg = "'" + std::string(RVS_CONF_HIGH_PHASE_RATIO_KEY)
      + "' must be between 0.0 and 1.0";
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
    bsts = false;
  }

  if (pulse_rate <= 0) {
    msg = "'" + std::string(RVS_CONF_PULSE_RATE_KEY)
      + "' must be positive";
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
    bsts = false;
  }

  if (pulse_sample_interval < 50) {
    pulse_sample_interval = 50;
  }

  return bsts;
}

void pulse_action::hip_to_smi_indices(void) {
  int hip_num_gpu_devices;
  hipGetDeviceCount(&hip_num_gpu_devices);

  std::map<uint64_t, amdsmi_processor_handle> smi_map;
  smi_map = rvs::get_smi_pci_map();

  for (int i = 0; i < hip_num_gpu_devices; i++) {
    unsigned int pDom, pBus, pDev, pFun;
    getBDF(i, pDom, pBus, pDev, pFun);
    uint64_t hip_dev_location_id = ( ( ((uint64_t)pDom & 0xffff ) << 32) |
        (((uint64_t) pBus & 0xff ) << 8) | (((uint64_t)pDev & 0x1f ) << 3)| ((uint64_t)pFun ) );

    if(smi_map.find(hip_dev_location_id) != smi_map.end()){
      hip_to_smi_idxs.insert({i, smi_map[hip_dev_location_id]});
    }
  }
}

bool pulse_action::do_pulse_test(map<int, uint16_t> pulse_gpus_device_index,
    std::vector<mcm_type_t>& mcm_type) {
  std::string  msg;
  unsigned int i = 0;
  int          gpuId = 0;

  int num_gpus = static_cast<int>(pulse_gpus_device_index.size());
  vector<PulseWorker> workers(num_gpus);

  // Shared flag: when any GPU's duration expires, it sets this before
  // arriving at the barrier so all GPUs see it and exit together.
  std::atomic<bool> done_flag{false};

  // Allocate fine-grained coherent system memory for GPU-side barrier
  int32_t* gpu_arrival_count = nullptr;
  int32_t* gpu_release_flag = nullptr;

  if (property_parallel && num_gpus > 1) {
    if (hipHostMalloc(&gpu_arrival_count, sizeof(int32_t),
          hipHostMallocCoherent) != hipSuccess) {
      msg = "Failed to allocate fine-grained memory for GPU sync barrier";
      rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
      return false;
    }
    if (hipHostMalloc(&gpu_release_flag, sizeof(int32_t),
          hipHostMallocCoherent) != hipSuccess) {
      msg = "Failed to allocate fine-grained memory for GPU sync barrier";
      rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
      hipHostFree(gpu_arrival_count);
      return false;
    }
    *gpu_arrival_count = 0;
    *gpu_release_flag = 0;
  }

  // CPU-side barrier for host thread alignment (C++20)
  std::barrier cpu_barrier(num_gpus);

  for (;;) {
    map<int, uint16_t>::iterator it;

    if (property_wait != 0)
      sleep(property_wait);

    hip_to_smi_indices();

    PulseWorker::set_use_json(bjson);

    i = 0;
    for (it = pulse_gpus_device_index.begin();
         it != pulse_gpus_device_index.end(); ++it) {
      if(hip_to_smi_idxs.find(it->first) != hip_to_smi_idxs.end()){
        workers[i].set_smi_device_handle(hip_to_smi_idxs[it->first]);
      } else {
        workers[i].set_smi_device_handle(nullptr);
        msg = "[" + action_name + "] " + MODULE_NAME + " " +
          std::to_string(i) + " has no SMI handle";
        rvs::lp::Log(msg, rvs::logerror);
      }
      gpuId = it->second;

      workers[i].set_name(action_name);
      workers[i].set_action(*this);
      workers[i].set_gpu_id(it->second);
      workers[i].set_gpu_device_index(it->first);
      workers[i].set_run_duration_ms(property_duration);
      workers[i].set_sample_interval(pulse_sample_interval);
      workers[i].set_log_interval(property_log_interval);
      workers[i].set_pulse_rate(pulse_rate);
      workers[i].set_high_phase_ratio(high_phase_ratio);
      workers[i].set_tolerance(pulse_tolerance);
      workers[i].set_matrix_size(pulse_matrix_size);
      workers[i].set_ops_type(pulse_ops_type);
      workers[i].set_data_type(pulse_data_type);
      workers[i].set_out_data_type(pulse_out_data_type);
      workers[i].set_matrix_transpose_a(pulse_trans_a);
      workers[i].set_matrix_transpose_b(pulse_trans_b);
      workers[i].set_alpha_val(pulse_alpha_val);
      workers[i].set_beta_val(pulse_beta_val);
      workers[i].set_lda_offset(pulse_lda_offset);
      workers[i].set_ldb_offset(pulse_ldb_offset);
      workers[i].set_ldc_offset(pulse_ldc_offset);
      workers[i].set_ldd_offset(pulse_ldd_offset);
      workers[i].set_workload_iterations(pulse_workload_iterations);
      workers[i].set_halt_on_error(pulse_halt_on_error);
      workers[i].set_verify_mode(pulse_verify_mode);
      workers[i].set_hot_calls(pulse_hot_calls);
      workers[i].set_matrix_init(pulse_matrix_init);
      workers[i].set_blas_source(pulse_blas_source);
      workers[i].set_compute_type(pulse_compute_type);
      workers[i].set_max_temp_c(pulse_max_temp_c);
      workers[i].set_mcm_type(mcm_type[i]);
      workers[i].set_num_gpus(num_gpus);
      workers[i].set_worker_index(i);

      if (property_parallel && num_gpus > 1) {
        workers[i].set_sync_resources(&cpu_barrier,
            gpu_arrival_count, gpu_release_flag, &done_flag);
      }

      i++;
    }

    if (property_parallel) {
      for (i = 0; i < pulse_gpus_device_index.size(); i++)
        workers[i].start();
      for (i = 0; i < pulse_gpus_device_index.size(); i++)
        workers[i].join();
    } else {
      for (i = 0; i < pulse_gpus_device_index.size(); i++) {
        workers[i].start();
        workers[i].join();
        if (rvs::lp::Stopping()) {
          if (gpu_arrival_count) hipHostFree(gpu_arrival_count);
          if (gpu_release_flag) hipHostFree(gpu_release_flag);
          return false;
        }
      }
    }

    msg = "[" + action_name + "] " + MODULE_NAME + " " +
      std::to_string(gpuId) + " Completed pulse cycle";
    rvs::lp::Log(msg, rvs::loginfo);

    if (rvs::lp::Stopping()) {
      if (gpu_arrival_count) hipHostFree(gpu_arrival_count);
      if (gpu_release_flag) hipHostFree(gpu_release_flag);
      return false;
    }

    // single iteration for pulse test
    break;
  }

  if (gpu_arrival_count) hipHostFree(gpu_arrival_count);
  if (gpu_release_flag) hipHostFree(gpu_release_flag);

  for (i = 0; i < pulse_gpus_device_index.size(); i++) {
    if(false == workers[i].get_result())
      return false;
  }

  return true;
}

int pulse_action::get_num_amd_gpu_devices(void) {
  int hip_num_gpu_devices;
  hipGetDeviceCount(&hip_num_gpu_devices);
  return hip_num_gpu_devices;
}

int pulse_action::get_all_selected_gpus(void) {
  int hip_num_gpu_devices;
  bool amd_gpus_found = false;
  map<int, uint16_t> pulse_gpus_device_index;
  std::string msg;
  std::vector<mcm_type_t> mcm_type;

  hipGetDeviceCount(&hip_num_gpu_devices);
  if (hip_num_gpu_devices < 1)
    return -1;

  amd_gpus_found = fetch_gpu_list(hip_num_gpu_devices,
      pulse_gpus_device_index,
      property_device, property_device_id, property_device_all,
      property_device_index, property_device_index_all, true, &mcm_type);
  if(!amd_gpus_found){
    msg = "No devices match criteria from the test configuration.";
    rvs::lp::Log(msg, rvs::logerror);
    if (bjson) {
      unsigned int sec;
      unsigned int usec;
      rvs::lp::get_ticks(&sec, &usec);
      void *json_root_node = rvs::lp::LogRecordCreate(MODULE_NAME,
          action_name.c_str(), rvs::logerror, sec, usec, true);
      if (!json_root_node) {
        string emsg = std::string(JSON_CREATE_NODE_ERROR);
        rvs::lp::Err(emsg, MODULE_NAME_CAPS, action_name);
        return -1;
      }
      rvs::lp::AddString(json_root_node, "ERROR",
          "No AMD compatible GPU found!");
      rvs::lp::LogRecordFlush(json_root_node, rvs::logerror);
    }
    return -1;
  }

  int res = 0;
  if(do_pulse_test(pulse_gpus_device_index, mcm_type))
    res = 0;
  else
    res = -1;
  return res;
}

int pulse_action::run(void) {
  string msg;
  rvs::action_result_t action_result;

  rvs::lp::Log(std::string(kPulseBetaBanner), rvs::logresults);

  if (!get_all_common_config_keys())
    return -1;

  if (!get_all_pulse_config_keys())
    return -1;

  if(bjson){
    json_add_primary_fields(std::string(MODULE_NAME), action_name);
  }

  auto res = get_all_selected_gpus();
  if(bjson){
    rvs::lp::JsonActionEndNodeCreate();
  }

  action_result.state = rvs::actionstate::ACTION_COMPLETED;
  action_result.status = (!res) ? rvs::actionstatus::ACTION_SUCCESS
                                : rvs::actionstatus::ACTION_FAILED;
  action_result.output = "PULSE Module action " + action_name + " completed";
  action_callback(&action_result);

  return res;
}
