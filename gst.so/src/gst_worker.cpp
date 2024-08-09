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
#include "include/gst_worker.h"

#include <unistd.h>
#include <string>
#include <memory>
#include <iostream>
#include <iomanip>
#include "include/rvs_blas.h"
#include "include/rvs_module.h"
#include "include/rvsloglp.h"
#include "include/rvs_util.h"

#define MODULE_NAME                             "gst"

#define GST_MEM_ALLOC_ERROR                     "memory allocation error!"
#define GST_BLAS_ERROR                          "blas error !!!"
#define GST_BLAS_MEMCPY_ERROR                   "HostToDevice mem copy error!"

#define GST_MAX_GFLOPS_OUTPUT_KEY               "Gflop"
#define GST_FLOPS_PER_OP_OUTPUT_KEY             "flops_per_op"
#define GST_BYTES_COPIED_PER_OP_OUTPUT_KEY      "bytes_copied_per_op"
#define GST_TRY_OPS_PER_SEC_OUTPUT_KEY          "try_ops_per_sec"

#define GST_LOG_SELF_CHECK_ERROR_KEY            "self-check error"
#define GST_LOG_ACCU_CHECK_ERROR_KEY            "accu-check error"
#define GST_LOG_GFLOPS_INTERVAL_KEY             "GFLOPS"
#define GST_JSON_LOG_GPU_ID_KEY                 "gpu_id"

#define PROC_DEC_INC_SGEMM_FREQ_DELAY           10

#define NMAX_MS_GPU_RUN_PEAK_PERFORMANCE        1000
#define NMAX_MS_SGEMM_OPS_RAMP_SUB_INTERVAL     1000
#define USLEEP_MAX_VAL                          (1000000 - 1)

#define GST_COPY_MATRIX_MSG                     "copy matrix"
#define GST_START_MSG                           "start"
#define GST_PASS_KEY                            "pass"
#define GST_RAMP_EXCEEDED_MSG                   "ramp time exceeded"
#define GST_TARGET_ACHIEVED_MSG                 "target achieved"
#define GST_STRESS_VIOLATION_MSG                "stress violation"

using std::string;

bool GSTWorker::bjson = false;

GSTWorker::GSTWorker() {}
GSTWorker::~GSTWorker() {}

/**
 * @brief performs the rvsBlas setup
 * @param error pointer to a memory location where the error code will be stored
 * @param err_description stores the error description if any
 */
void GSTWorker::setup_blas(int *error, string *err_description) {
  *error = 0;
  // setup rvsBlas
  gpu_blas = std::unique_ptr<rvs_blas>(
      new rvs_blas(gpu_device_index, matrix_size_a, matrix_size_b,
        matrix_size_c, matrix_init, gst_trans_a, gst_trans_b,
        gst_alpha_val, gst_beta_val,
        gst_lda_offset, gst_ldb_offset, gst_ldc_offset, gst_ldd_offset, gst_ops_type, gst_data_type,
        gemm_mode, batch_size, stride_a, stride_b, stride_c, stride_d));

  if (!gpu_blas) {
    *error = 1;
    *err_description = GST_MEM_ALLOC_ERROR;
    return;
  }

  if (gpu_blas->error()) {
    *error = 1;
    *err_description = GST_MEM_ALLOC_ERROR;
    return;
  }

  // generate random matrix & copy it to the GPU
  gpu_blas->generate_random_matrix_data();
  if (!copy_matrix) {
    // copy matrix only once
    if (!gpu_blas->copy_data_to_gpu()) {
      *error = 1;
      *err_description = GST_BLAS_MEMCPY_ERROR;
    }
  }
}

/**
 * @brief attempts to hit the maximum Gflops value
 * @param error pointer to a memory location where the error code will be stored
 * @param err_description stores the error description if any
 */
void GSTWorker::hit_max_gflops(int *error, string *err_description) {
  std::chrono::time_point<std::chrono::system_clock> gst_start_time,
    gst_end_time,
    gst_log_interval_time;
  double seconds_elapsed = 0, curr_gflops;
  uint16_t num_sgemm_ops_log_interval = 0;
  uint64_t millis_sgemm_ops;
  string msg;

  *error = 0;
  gst_start_time = std::chrono::system_clock::now();
  gst_log_interval_time = std::chrono::system_clock::now();

  for (;;) {
    // check if stop signal was received
    if (rvs::lp::Stopping())
      break;

    gst_end_time = std::chrono::system_clock::now();
    if (time_diff(gst_end_time, gst_start_time) >=
        NMAX_MS_GPU_RUN_PEAK_PERFORMANCE)
      break;

    if (copy_matrix) {
      // copy matrix before each GEMM
      if (!gpu_blas->copy_data_to_gpu()) {
        *error = 1;
        *err_description = GST_BLAS_MEMCPY_ERROR;
        return;
      }
    }

    // run GEMM operation
    if (!gpu_blas->run_blas_gemm())
      continue;  // failed to run the GEMM operation

    // Waits for GEMM operation to complete
    if(!gpu_blas->is_gemm_op_complete())
      continue;  // failed to run the GEMM operation

    num_sgemm_ops_log_interval++;

    gst_end_time = std::chrono::system_clock::now();
    millis_sgemm_ops = time_diff(gst_end_time, gst_log_interval_time);
    if (millis_sgemm_ops >= log_interval) {
      // compute the GFLOPS
      seconds_elapsed = static_cast<double> (millis_sgemm_ops) / 1000;
      if (seconds_elapsed != 0) {
        curr_gflops = static_cast<double>(gpu_blas->gemm_gflop_count() *
            num_sgemm_ops_log_interval) / seconds_elapsed;
        log_interval_gflops(curr_gflops);
      }

      num_sgemm_ops_log_interval = 0;
      gst_log_interval_time = std::chrono::system_clock::now();
    }
  }
}

/**
 * @brief performs the ramp-up on the given GPU (attempts to reach the given 
 * target stress Gflops)
 * @param error pointer to a memory location where the error code will be stored
 * @param err_description stores the error description if any
 * @return true if target stress is achieved within the ramp_interval,
 * false otherwise
 */
bool GSTWorker::do_gst_ramp(int *error, string *err_description) {
  std::chrono::time_point<std::chrono::system_clock> gst_start_time,
    gst_end_time,
    gst_log_interval_time,
    gst_start_gflops_time,
    gst_last_sgemm_start_time,
    gst_last_sgemm_end_time;
  double seconds_elapsed, curr_gflops, dyn_delay_target_stress;
  uint16_t num_sgemm_ops = 0, num_sgemm_ops_log_interval = 0;
  uint64_t millis_sgemm_ops, millis_last_sgemm;
  uint16_t proc_delay = 0;
  uint64_t start_time, end_time;
  double timetakenforoneiteration, gflops_interval;
  string msg;

  // make sure that the ramp_interval & duration are not less than
  // NMAX_MS_GPU_RUN_PEAK_PERFORMANCE (e.g.: 1000)
  if (run_duration_ms < NMAX_MS_GPU_RUN_PEAK_PERFORMANCE)
    run_duration_ms += NMAX_MS_GPU_RUN_PEAK_PERFORMANCE;

  if (ramp_interval < NMAX_MS_GPU_RUN_PEAK_PERFORMANCE)
    ramp_interval += NMAX_MS_GPU_RUN_PEAK_PERFORMANCE;

  // stage 1. setup rvs blas
  setup_blas(error, err_description);
  if (*error)
    return false;

  // check if stop signal was received
  if (rvs::lp::Stopping())
    return false;

  // stage 3. reduce the SGEMM frequency and try to achieve the desired Gflops
  // the delay which gives the SGEMM frequency will be dynamically computed
  delay_target_stress = 0;

  gst_start_time = std::chrono::system_clock::now();
  gst_log_interval_time = std::chrono::system_clock::now();
  gst_start_gflops_time = std::chrono::system_clock::now();

  for (;;) {
    // check if stop signal was received
    if (rvs::lp::Stopping())
      return false;

    gst_end_time = std::chrono::system_clock::now();
    if (time_diff(gst_end_time,  gst_start_time) >
        ramp_interval - NMAX_MS_GPU_RUN_PEAK_PERFORMANCE)
      return false;

    gst_last_sgemm_start_time = std::chrono::system_clock::now();

    if (copy_matrix) {
      // Generate random matrix data
      gpu_blas->generate_random_matrix_data();
      // copy matrix before each GEMM
      if (!gpu_blas->copy_data_to_gpu()) {
        *error = 1;
        *err_description = GST_BLAS_MEMCPY_ERROR;
        return false;
      }
    }

    //Start the timer
    start_time = gpu_blas->get_time_us();

    // run GEMM operation
    if(!gpu_blas->run_blas_gemm()) {

      *err_description = GST_BLAS_ERROR;
      *error = 1;
      return false;
    }

    // Wait for GEMM operation to complete
    if(!gpu_blas->is_gemm_op_complete()) {

      *err_description = GST_BLAS_ERROR;
      *error = 1;
      return false;
    }

    //End the timer
    end_time = gpu_blas->get_time_us();

    //Converting microseconds to seconds
    timetakenforoneiteration = (end_time - start_time)/1e6;

    gflops_interval = gpu_blas->gemm_gflop_count()/timetakenforoneiteration;

    gst_last_sgemm_end_time = std::chrono::system_clock::now();
    millis_last_sgemm =
      time_diff(gst_last_sgemm_end_time, gst_last_sgemm_start_time);
    if (static_cast<double>(
          (1000 * gpu_blas->gemm_gflop_count()) /
          target_stress) <
        millis_last_sgemm) {
      // last SGEMM timed-out (it took more than it should)
      dyn_delay_target_stress = 1;
    }


    num_sgemm_ops++;
    num_sgemm_ops_log_interval++;

    gst_end_time = std::chrono::system_clock::now();
    millis_sgemm_ops =
      time_diff(gst_end_time, gst_start_gflops_time);
    if (millis_sgemm_ops >= NMAX_MS_SGEMM_OPS_RAMP_SUB_INTERVAL) {
      // compute the GFLOPS
      seconds_elapsed = static_cast<double>
        (millis_sgemm_ops) / 1000;
      if (seconds_elapsed > 0) {
        curr_gflops = static_cast<double>(
            gpu_blas->gemm_gflop_count() *
            num_sgemm_ops) / seconds_elapsed;
        if (curr_gflops >= target_stress && curr_gflops <
            target_stress + target_stress * tolerance/2) {
          ramp_actual_time =
            time_diff(gst_end_time,  gst_start_time) +
            NMAX_MS_GPU_RUN_PEAK_PERFORMANCE;
          delay_target_stress /= num_sgemm_ops;
          return true;
        }
      }
      proc_delay +=
        (delay_target_stress * PROC_DEC_INC_SGEMM_FREQ_DELAY) / 100;
      num_sgemm_ops = 0;
      delay_target_stress = 0;
      gst_start_gflops_time = std::chrono::system_clock::now();
    }

    millis_sgemm_ops =
      time_diff(gst_end_time, gst_log_interval_time);
    if (millis_sgemm_ops >= log_interval) {
      // compute the GFLOPS
      seconds_elapsed = static_cast<double>
        (millis_sgemm_ops) / 1000;

      if (seconds_elapsed > 0) {
        curr_gflops = static_cast<double>(
            gpu_blas->gemm_gflop_count() *
            num_sgemm_ops_log_interval) / seconds_elapsed;
        log_interval_gflops(gflops_interval);
      }

      num_sgemm_ops_log_interval = 0;
      gst_log_interval_time = std::chrono::system_clock::now();
    }
  }

  return false;
}

/**
 * @brief logs the Gflops computed over the last log_interval period 
 * @param gflops_interval the Gflops that the GPU achieved
 */
void GSTWorker::check_target_stress(double gflops_interval) {
  string msg;
  bool result;
  rvs::action_result_t action_result;

  if(gflops_interval >= target_stress){
    result = true;
  }else{
    result = false;
  }

  msg = "[" + action_name + "] " + "[GPU:: " + std::to_string(gpu_id) + "] " +
    GST_LOG_GFLOPS_INTERVAL_KEY + " " + std::to_string(static_cast<uint64_t>(gflops_interval)) + " " +
    "Target GFLOPS:" + " " + std::to_string(static_cast<uint64_t>(target_stress)) +
    " met: " + (result ? "TRUE" : "FALSE");
  rvs::lp::Log(msg, rvs::logresults);

  action_result.state = rvs::actionstate::ACTION_RUNNING;
  action_result.status = (true == result) ? rvs::actionstatus::ACTION_SUCCESS : rvs::actionstatus::ACTION_FAILED;
  action_result.output = msg.c_str();
  action.action_callback(&action_result);

  log_to_json(GST_LOG_GFLOPS_INTERVAL_KEY, std::to_string(static_cast<uint64_t>(gflops_interval)),
      rvs::loginfo);
}

/**
 * @brief logs the Gflops computed over the last log_interval period 
 * @param gflops_interval the Gflops that the GPU achieved
 */
void GSTWorker::log_interval_gflops(double gflops_interval) {
  string msg;
  rvs::action_result_t action_result;

  msg = "[" + action_name + "] " + "[GPU:: " + std::to_string(gpu_id) + "] " +
    GST_LOG_GFLOPS_INTERVAL_KEY + " " + std::to_string(static_cast<uint64_t>(gflops_interval));
  rvs::lp::Log(msg, rvs::logresults);

  action_result.state = rvs::actionstate::ACTION_RUNNING;
  action_result.status = rvs::actionstatus::ACTION_SUCCESS;
  action_result.output = msg.c_str();
  action.action_callback(&action_result);

  log_to_json(GST_LOG_GFLOPS_INTERVAL_KEY, std::to_string(static_cast<uint64_t>(gflops_interval)),
      rvs::loginfo);
}

/**
 * @brief checks for Gflops violation 
 * @param gflops_interval the Gflops that the GPU achieved over the last
 * log_interval period
 * @return true if this gflops violates the bounds, false otherwise
 */
bool GSTWorker::check_gflops_violation(double gflops_interval) {
  string msg;

  if (!(gflops_interval > target_stress - target_stress * tolerance &&
        gflops_interval < target_stress + target_stress * tolerance)) {
    msg = "[" + action_name + "] " + MODULE_NAME + " " +
      std::to_string(gpu_id) + " " + GST_STRESS_VIOLATION_MSG + " " +
      std::to_string(gflops_interval);
    //        rvs::lp::Log(msg, rvs::loginfo);

    //log_to_json(GST_STRESS_VIOLATION_MSG, std::to_string(gflops_interval),
    //           rvs::loginfo);
    return true;
  }


  return false;
}

/**
 * @brief performs the stress test on the given GPU
 * @param error pointer to a memory location where the error code will be stored
 * @param err_description stores the error description if any
 * @return true if stress violations is less than max_violations, false otherwise
 */
bool GSTWorker::do_gst_stress_test(int *error, std::string *err_description) {

  uint32_t num_gemm_ops = 0;
  uint64_t total_milliseconds, log_interval_milliseconds;
  double start_time, end_time;
  double seconds_elapsed, gflops_interval;
  double timetakenforoneiteration;
  double timetakenforniterations;
  string msg;
  std::chrono::time_point<std::chrono::system_clock> gst_start_time,
    gst_end_time, gst_log_interval_time;

  *error = 0;
  max_gflops = 0;
  num_gemm_ops = 0;
  start_time = 0;
  end_time = 0;

  gst_start_time = std::chrono::system_clock::now();
  gst_log_interval_time = std::chrono::system_clock::now();

  for (;;) {

    // check if stop signal was received
    if (rvs::lp::Stopping())
      return false;

    if (copy_matrix) {
      // copy matrix before each GEMM
      if (!gpu_blas->copy_data_to_gpu()) {
        *error = 1;
        *err_description = GST_BLAS_MEMCPY_ERROR;
        return false;
      }
    }

    //Start the timer
    start_time = gpu_blas->get_time_us();

    for (uint64_t i = 0; i < gst_hot_calls; i++) {

      // launch GEMM operation
      if(!gpu_blas->run_blas_gemm()) {

        *err_description = GST_BLAS_ERROR;
        *error = 1;
        return false;
      }
    }

    // Wait for all the GEMM operations to complete
    if(!gpu_blas->is_gemm_op_complete()) {

      *err_description = GST_BLAS_ERROR;
      *error = 1;
      return false;
    }

    //End the timer
    end_time = gpu_blas->get_time_us();

    timetakenforniterations += (end_time - start_time);

    num_gemm_ops += gst_hot_calls;

    gst_end_time = std::chrono::system_clock::now();
    total_milliseconds = time_diff(gst_end_time, gst_start_time);

    log_interval_milliseconds = time_diff(gst_end_time,
        gst_log_interval_time);

    if (log_interval_milliseconds >= log_interval && num_gemm_ops > 0) {

      seconds_elapsed = static_cast<double> (log_interval_milliseconds) / 1000;

      if (seconds_elapsed != 0) {

        timetakenforoneiteration = timetakenforniterations / num_gemm_ops;

        gflops_interval = gpu_blas->gemm_gflop_count()/timetakenforoneiteration * 1e6;

        if (gflops_interval > max_gflops)
          max_gflops = gflops_interval;

        log_interval_gflops(gflops_interval);

        // reset time & gflops related data
        num_gemm_ops = 0;
        timetakenforniterations = 0;

        gst_log_interval_time = std::chrono::system_clock::now();
      }
    }

    if (self_check || accu_check) {

      if (error_inject) {
        gpu_blas->set_gemm_error(error_freq, error_count);
      }

      double self_error = 0.0;
      double accu_error = 0.0;

      gpu_blas->validate_gemm(self_check, accu_check, self_error, accu_error);

      if(self_error > 0) {

        std::ostringstream oss;
        oss << std::setprecision(10) << std::noshowpoint << std::fixed << self_error;

        msg = "[" + action_name + "] " + "[GPU:: " + std::to_string(gpu_id) + "] " +
          GST_LOG_SELF_CHECK_ERROR_KEY + " " + oss.str();
        rvs::lp::Log(msg, rvs::logresults);
      }

      if(accu_error > 0) {

        std::ostringstream oss;
        oss << std::setprecision(10) << std::noshowpoint << std::fixed << accu_error;

        msg = "[" + action_name + "] " + "[GPU:: " + std::to_string(gpu_id) + "] " +
          GST_LOG_ACCU_CHECK_ERROR_KEY + " " + oss.str();
        rvs::lp::Log(msg, rvs::logresults);
      }
    }

    msg = "[" + action_name + "] " + MODULE_NAME + " " +
      std::to_string(gpu_id) + " " + GST_START_MSG + " " +
      " Execution time in milliseconds :" + std::to_string(total_milliseconds) +
      " run_duration_ms :" + std::to_string(run_duration_ms);
    rvs::lp::Log(msg, rvs::logtrace);

    if (total_milliseconds >= run_duration_ms)
      break;
  }

  return true;
}

/**
 * @brief performs the stress test on the given GPU
 */
void GSTWorker::run() {
  string msg, err_description;
  int error = 0;
  bool gst_test_passed = true;
  rvs::action_result_t action_result;

  max_gflops = 0;

  // log GST stress test - start message
  msg = "[" + action_name + "] " + MODULE_NAME + " " +
    std::to_string(gpu_id) + " " + GST_START_MSG + " " +
    " Starting the GST stress test ";
  rvs::lp::Log(msg, rvs::logtrace);

  // log GST ramp up - start message
  msg = "[" + action_name + "] " + "[GPU:: " + std::to_string(gpu_id) + "] " +
    "Start of GPU ramp up";
  rvs::lp::Log(msg, rvs::logresults);

  // let the GPU ramp-up and check the result
  bool ramp_up_success = do_gst_ramp(&error, &err_description);

  // log GST ramp up - end message
  msg = "[" + action_name + "] " + "[GPU:: " + std::to_string(gpu_id) + "] " +
    "End of GPU ramp up";
  rvs::lp::Log(msg, rvs::logresults);

  // GPU was not able to do the processing (HIP/rocBlas error(s) occurred)
  if (error) {
    string msg = "[" + action_name + "] " + MODULE_NAME + " "
      + std::to_string(gpu_id) + " " + err_description;
    rvs::lp::Log(msg, rvs::logerror);
    log_to_json("err", err_description, rvs::logerror);

    action_result.state = rvs::actionstate::ACTION_COMPLETED;
    action_result.status = rvs::actionstatus::ACTION_FAILED;
    action_result.output = msg.c_str();
    action.action_callback(&action_result);

    return;
  }

  // the GPU succeeded to achieve the target_stress GFLOPS
  // continue with the same workload for the rest of the test duration
  msg = "[" + action_name + "] " + MODULE_NAME + " " +
    std::to_string(gpu_id) + " " + " GST ramp completed for interval :" + " " +
    std::to_string(ramp_interval);
  rvs::lp::Log(msg, rvs::loginfo);
  //log_to_json(GST_TARGET_ACHIEVED_MSG, std::to_string(target_stress),
  //                rvs::loginfo);
  if (run_duration_ms > 0) {
    gst_test_passed = do_gst_stress_test(&error, &err_description);
    // check if stop signal was received
    if (rvs::lp::Stopping())
      return;

    if (error) {
      // GPU didn't complete the test (HIP/rocBlas error(s) occurred)
      string msg = "[" + action_name + "] " + MODULE_NAME + " " +
        std::to_string(gpu_id) + " " + err_description;
      rvs::lp::Log(msg, rvs::logerror);
      log_to_json("err", err_description, rvs::logerror);

      action_result.state = rvs::actionstate::ACTION_COMPLETED;
      action_result.status = rvs::actionstatus::ACTION_FAILED;
      action_result.output = msg.c_str();
      action.action_callback(&action_result);

      return;
    }
  }

  log_interval_gflops(max_gflops);
  check_target_stress(max_gflops);
}

/**
 * @brief logs the GST test result
 * @param gst_test_passed true if test succeeded, false otherwise
 */
void GSTWorker::log_gst_test_result(bool gst_test_passed) {
  string msg;

  double flops_per_op = (2 * (static_cast<double>(gpu_blas->get_m())/1000) *
      (static_cast<double>(gpu_blas->get_n())/1000) *
      (static_cast<double>(gpu_blas->get_k())/1000));
  msg = "[" + action_name + "] " + MODULE_NAME + " " +
    std::to_string(gpu_id) + " " + GST_MAX_GFLOPS_OUTPUT_KEY + ": " +
    std::to_string(max_gflops) + " " + GST_FLOPS_PER_OP_OUTPUT_KEY + ": " +
    std::to_string(flops_per_op) + "x1e9" + " " +
    GST_BYTES_COPIED_PER_OP_OUTPUT_KEY + ": " +
    std::to_string(gpu_blas->get_bytes_copied_per_op()) +
    " " + GST_TRY_OPS_PER_SEC_OUTPUT_KEY + ": "+
    std::to_string(target_stress / gpu_blas->gemm_gflop_count()) +
    " "  ;
  rvs::lp::Log(msg, rvs::logresults);

  log_to_json(GST_MAX_GFLOPS_OUTPUT_KEY, std::to_string(max_gflops),
      rvs::loginfo);
  log_to_json(GST_FLOPS_PER_OP_OUTPUT_KEY, std::to_string(flops_per_op) +
      "x1e9", rvs::loginfo);
  log_to_json(GST_BYTES_COPIED_PER_OP_OUTPUT_KEY,
      std::to_string(gpu_blas->get_bytes_copied_per_op()),
      rvs::loginfo);
  log_to_json(GST_TRY_OPS_PER_SEC_OUTPUT_KEY,
      std::to_string(target_stress / gpu_blas->gemm_gflop_count()),
      rvs::loginfo);
  log_to_json(GST_PASS_KEY, (gst_test_passed ?
        GST_RESULT_PASS_MESSAGE : GST_RESULT_FAIL_MESSAGE),
      rvs::logresults);
}

/**
 * @brief computes the difference (in milliseconds) between 2 points in time
 * @param t_end second point in time
 * @param t_start first point in time
 * @return time difference in milliseconds
 */
uint64_t GSTWorker::time_diff(
    std::chrono::time_point<std::chrono::system_clock> t_end,
    std::chrono::time_point<std::chrono::system_clock> t_start) {
  auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
      t_end - t_start);
  return milliseconds.count();
}


/**
 * @brief logs a message to JSON
 * @param key info type
 * @param value message to log
 * @param log_level the level of log (e.g.: info, results, error)
 */
void GSTWorker::log_to_json(const std::string &key, const std::string &value,
    int log_level) {
  if (GSTWorker::bjson) {
    void *json_node = json_node_create(std::string(MODULE_NAME),
        action_name.c_str(), log_level);
    if (json_node) {
      rvs::lp::AddString(json_node, GST_JSON_LOG_GPU_ID_KEY,
          std::to_string(gpu_id));
      rvs::lp::AddString(json_node, key, value);
      rvs::lp::LogRecordFlush(json_node, log_level);
    }
  }
}


/**
 * @brief extends the usleep for more than 1000000us
 * @param microseconds us to sleep
 */
void GSTWorker::usleep_ex(uint64_t microseconds) {
  uint64_t total_microseconds = microseconds;
  for (;;) {
    if (total_microseconds > USLEEP_MAX_VAL) {
      usleep(USLEEP_MAX_VAL);
      total_microseconds -= USLEEP_MAX_VAL;
    } else {
      usleep(total_microseconds);
      return;
    }
  }
}

