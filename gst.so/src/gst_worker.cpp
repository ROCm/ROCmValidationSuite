/********************************************************************************
 *
 * Copyright (c) 2018-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <chrono>
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
#define GST_LOG_CRC_CHECK_ERROR_KEY             "crc-check error"
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
const std::string TARGET_KEY{"target"};
const std::string DTYPE_KEY{"dtype"};

using std::string;

bool GSTWorker::bjson = false;

// ---------------------------------------------------------------------------
// GstCrcSync — per-iteration cross-GPU CRC barrier
// ---------------------------------------------------------------------------

void GstCrcSync::sync_and_compare(size_t slot, uint16_t gpu_id, uint32_t crc) {
  std::unique_lock<std::mutex> lock(mtx);

  crcs[slot]    = crc;
  gpu_ids[slot] = gpu_id;
  ++arrived;
  uint64_t my_gen = generation;

  if (arrived == total_workers) {
    // Last worker to arrive: collect mismatches, then release all waiters.
    //
    // Iterate slot indices and skip detached slots.  total_workers is a
    // running count, not a slot bound: using it here would skip live
    // high-numbered slots and treat a detached slot 0 as the reference.
    std::vector<std::string> msgs;
    size_t ref = active.size();          // sentinel: no active slot found
    for (size_t i = 0; i < active.size(); ++i) {
      if (active[i]) {
        ref = i;
        break;
      }
    }

    // When ref is the sentinel this loop starts past the end and does nothing.
    for (size_t i = ref + 1; i < active.size(); ++i) {
      if (!active[i])
        continue;
      if (crcs[i] != crcs[ref]) {
        std::ostringstream oss;
        oss << "[" << action_name << "] "
            << "cross-GPU crc-check error: "
            << "GPU " << gpu_ids[i]
            << " CRC 0x" << std::hex << std::setw(8) << std::setfill('0') << crcs[i]
            << " != GPU " << std::dec << gpu_ids[ref]
            << " CRC 0x" << std::hex << std::setw(8) << std::setfill('0') << crcs[ref]
            << " (SDC candidate)";
        msgs.push_back(oss.str());
      }
    }
    arrived = 0;
    ++generation;
    cv.notify_all();
    lock.unlock();
    for (const auto& m : msgs)
      rvs::lp::Log(m, rvs::logresults);
  } else {
    // Wait until the last worker releases this round (or the barrier aborts).
    cv.wait(lock, [this, my_gen]() {
      return generation != my_gen || aborted;
    });
  }
}

void GstCrcSync::detach(size_t slot) {
  std::unique_lock<std::mutex> lock(mtx);
  // Idempotent per slot: a worker may reach both the explicit detach on the
  // compute_output_crc() error path and its RAII guard on function exit.
  if (slot >= active.size() || !active[slot])
    return;
  active[slot] = false;
  --total_workers;
  // If the remaining workers already arrived, act as if this was the last one.
  if (total_workers > 0 && arrived >= total_workers) {
    arrived = 0;
    ++generation;
    cv.notify_all();
  } else if (total_workers == 0) {
    aborted = true;
    cv.notify_all();
  }
}

GSTWorker::GSTWorker()
  : crc_check(false), crc_computed(false), crc_matrix_seed(0)
  , crc_sync(nullptr), crc_sync_slot(0) {}
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
        gemm_mode, batch_size, stride_a, stride_b, stride_c, stride_d, blas_source, compute_type,
        gst_out_data_type, gst_scale_a, gst_scale_b, gst_rotating, gst_hot_calls));

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

  // Apply shared seed before matrix generation so all GPU workers receive
  // identical inputs (prerequisite for a valid cross-GPU CRC comparison).
  if (crc_matrix_seed != 0)
    gpu_blas->set_matrix_seed(crc_matrix_seed);

  // generate random matrix & copy it to the GPU
  if (shared_matrices) {
    std::unique_lock<std::mutex> lk(shared_matrices->mtx);
    if (!shared_matrices->ready) {
      // First worker: generate and snapshot into the shared pool.
      gpu_blas->generate_random_matrix_data();
      gpu_blas->get_host_matrix_bytes(shared_matrices->a,
                                      shared_matrices->b,
                                      shared_matrices->c);
      gpu_blas->get_host_scale_bytes(shared_matrices->sa,
                                     shared_matrices->sb);
      shared_matrices->ready = true;
    } else {
      // Subsequent workers: copy from shared pool — bytes are identical by
      // construction (s_host_matrix_byte_sizes mirrors allocate_host_matrix_mem).
      gpu_blas->inject_host_matrix_data(shared_matrices->a,
                                        shared_matrices->b,
                                        shared_matrices->c);
      gpu_blas->inject_host_scale_data(shared_matrices->sa,
                                       shared_matrices->sb);
    }
  } else {
    gpu_blas->generate_random_matrix_data();
  }

  // Log per-GPU input matrix CRCs so the caller can verify all GPUs received
  // byte-identical data.  Only emitted when cross-GPU CRC checking is active.
  if (shared_matrices) {
    uint32_t crc_a, crc_b, crc_c;
    gpu_blas->get_host_matrix_input_crcs(crc_a, crc_b, crc_c);

    char gpuid_buff[12];
    snprintf(gpuid_buff, sizeof(gpuid_buff), "%5d", gpu_id);

    std::ostringstream oss;
    oss << "[" << action_name << "] "
        << "[GPU:: " << gpuid_buff << "] "
        << "input-matrix crc32 "
        << "A=0x" << std::hex << std::setw(8) << std::setfill('0') << crc_a
        << " B=0x" << std::setw(8) << std::setfill('0') << crc_b
        << " C=0x" << std::setw(8) << std::setfill('0') << crc_c;
    rvs::lp::Log(oss.str(), rvs::loginfo);

  }

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
  uint64_t micros_sgemm_ops;
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
        NMAX_MS_GPU_RUN_PEAK_PERFORMANCE * 1000u)
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
    if (!gpu_blas->run_blas_gemm(1))
      continue;  // failed to run the GEMM operation

    // Waits for GEMM operation to complete
    if(!gpu_blas->is_gemm_op_complete())
      continue;  // failed to run the GEMM operation

    num_sgemm_ops_log_interval++;

    gst_end_time = std::chrono::system_clock::now();
    micros_sgemm_ops = time_diff(gst_end_time, gst_log_interval_time);
    if (micros_sgemm_ops >= log_interval * 1000u) {
      // compute the GFLOPS
      seconds_elapsed = static_cast<double> (micros_sgemm_ops) / 1000000;
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
  uint64_t micros_sgemm_ops, micros_last_sgemm;
  uint16_t proc_delay = 0;
  uint64_t start_time, end_time;
  double timetakenforoneiteration, gflops_interval;
  string msg;

  // make sure that the ramp_interval & duration are not less than
  // NMAX_MS_GPU_RUN_PEAK_PERFORMANCE (e.g.: 1000)
  if (run_duration_ms > 0 && run_duration_ms < NMAX_MS_GPU_RUN_PEAK_PERFORMANCE)
    run_duration_ms += NMAX_MS_GPU_RUN_PEAK_PERFORMANCE;

  if (ramp_interval > 0 && ramp_interval < NMAX_MS_GPU_RUN_PEAK_PERFORMANCE)
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

  bool ramp_single_shot = (ramp_interval == 0);

  gst_start_time = std::chrono::system_clock::now();
  gst_log_interval_time = std::chrono::system_clock::now();
  gst_start_gflops_time = std::chrono::system_clock::now();

  for (;;) {
    // check if stop signal was received
    if (rvs::lp::Stopping())
      return false;

    if (!ramp_single_shot) {
      gst_end_time = std::chrono::system_clock::now();
      if (time_diff(gst_end_time,  gst_start_time) >
          (ramp_interval - NMAX_MS_GPU_RUN_PEAK_PERFORMANCE) * 1000u) {
        return false;
      }
    }

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
    if(!gpu_blas->run_blas_gemm(gst_warm_calls)) {

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

    gflops_interval = gpu_blas->gemm_gflop_count() * gst_warm_calls / timetakenforoneiteration;

    gst_last_sgemm_end_time = std::chrono::system_clock::now();
    micros_last_sgemm =
      time_diff(gst_last_sgemm_end_time, gst_last_sgemm_start_time);
    if (static_cast<double>(
          (1000000 * gpu_blas->gemm_gflop_count()) /
          target_stress) <
        micros_last_sgemm) {
      // last SGEMM timed-out (it took more than it should)
      dyn_delay_target_stress = 1;
    }


    num_sgemm_ops += gst_warm_calls;
    num_sgemm_ops_log_interval += gst_warm_calls;

    gst_end_time = std::chrono::system_clock::now();
    micros_sgemm_ops =
      time_diff(gst_end_time, gst_start_gflops_time);
    if (micros_sgemm_ops >= NMAX_MS_SGEMM_OPS_RAMP_SUB_INTERVAL * 1000u) {
      // compute the GFLOPS
      seconds_elapsed = static_cast<double>
        (micros_sgemm_ops) / 1000000;
      if (seconds_elapsed > 0) {
        curr_gflops = static_cast<double>(
            gpu_blas->gemm_gflop_count() *
            num_sgemm_ops) / seconds_elapsed;
        if (curr_gflops >= target_stress && curr_gflops <
            target_stress + target_stress * tolerance/2) {
          ramp_actual_time =
            time_diff(gst_end_time,  gst_start_time) +
            NMAX_MS_GPU_RUN_PEAK_PERFORMANCE * 1000u;
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

    micros_sgemm_ops =
      time_diff(gst_end_time, gst_log_interval_time);
    if (micros_sgemm_ops >= log_interval * 1000u) {
      // compute the GFLOPS
      seconds_elapsed = static_cast<double>
        (micros_sgemm_ops) / 1000000;

      if (seconds_elapsed > 0) {
        curr_gflops = static_cast<double>(
            gpu_blas->gemm_gflop_count() *
            num_sgemm_ops_log_interval) / seconds_elapsed;
        log_interval_gflops(gflops_interval);
      }

      num_sgemm_ops_log_interval = 0;
      gst_log_interval_time = std::chrono::system_clock::now();
    }

    if (ramp_single_shot) {
      return gflops_interval >= target_stress * (1.0 - tolerance);
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
  rvs::action_result_t action_result;
  char gpuid_buff[12];
  auto desc = action_descriptor{action_name, MODULE_NAME,gpu_id};
  snprintf(gpuid_buff, sizeof(gpuid_buff), "%5d", gpu_id);

  if(gflops_interval >= (target_stress- (target_stress * tolerance))){
    result = true;
  }else{
    result = false;
  }

  msg = "[" + action_name + "] " + "[GPU:: " + gpuid_buff + "] " +
    GST_LOG_GFLOPS_INTERVAL_KEY + " " + std::to_string(static_cast<uint64_t>(gflops_interval)) + " " +
    "Target GFLOPS:" + " " + std::to_string(static_cast<uint64_t>(target_stress)) +
    " met: " + (result ? "TRUE" : "FALSE");
  rvs::lp::Log(msg, rvs::logresults); 

  action_result.state = rvs::actionstate::ACTION_RUNNING;
  action_result.status = (true == result) ? rvs::actionstatus::ACTION_SUCCESS : rvs::actionstatus::ACTION_FAILED;
  action_result.output = msg.c_str();
  action.action_callback(&action_result);
  if (bjson)
      log_to_json(desc ,rvs::logresults,
		      TARGET_KEY, std::to_string(static_cast<uint64_t>(target_stress)),
		      DTYPE_KEY, !gst_data_type.empty() ? gst_data_type : gst_ops_type,
		      "gflops", std::to_string(static_cast<uint64_t>(gflops_interval)),
		      "pass", result ? "true" : "false");
}

/**
 * @brief logs the Gflops computed over the last log_interval period 
 * @param gflops_interval the Gflops that the GPU achieved
 */
void GSTWorker::log_interval_gflops(double gflops_interval) {
  string msg;
  rvs::action_result_t action_result;
  char gpuid_buff[12];

  snprintf(gpuid_buff, sizeof(gpuid_buff), "%5d", gpu_id);

  msg = "[" + action_name + "] " + "[GPU:: " + gpuid_buff + "] " +
    GST_LOG_GFLOPS_INTERVAL_KEY + " " + std::to_string(static_cast<uint64_t>(gflops_interval));
  rvs::lp::Log(msg, rvs::logresults);

  action_result.state = rvs::actionstate::ACTION_RUNNING;
  action_result.status = rvs::actionstatus::ACTION_SUCCESS;
  action_result.output = msg.c_str();
  action.action_callback(&action_result);

  //log_to_json(GST_LOG_GFLOPS_INTERVAL_KEY, std::to_string(static_cast<uint64_t>(gflops_interval)),
    //  rvs::loginfo);
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
  auto desc = action_descriptor{action_name, MODULE_NAME, gpu_id};
  uint64_t total_microseconds, log_interval_microseconds;
  double start_time, end_time;
  double seconds_elapsed, gflops_interval;
  double timetakenforoneiteration;
  double timetakenforniterations = 0;
  string msg;
  std::chrono::time_point<std::chrono::system_clock> gst_start_time,
    gst_end_time, gst_log_interval_time;

  *error = 0;
  max_gflops = 0;
  num_gemm_ops = 0;
  start_time = 0;
  end_time = 0;

  // RAII guard: calls crc_sync->detach() on any exit path (normal or early)
  // so peer workers are never left blocking at the barrier.
  struct CrcSyncGuard {
    GstCrcSync* s;
    size_t      slot;
    ~CrcSyncGuard() { if (s) s->detach(slot); }
  } crc_guard{crc_sync, crc_sync_slot};

  // Warn when crc_check is enabled with a configuration that produces
  // non-repeatable outputs:
  //   sgemm / dgemm + copy_matrix:false + beta != 0
  // In this case the C matrix is the GEMM output buffer and gets overwritten
  // each iteration, so inputs change and CRC mismatches are expected even on
  // a healthy GPU (false positives).
  if (crc_check &&
      !copy_matrix &&
      (gst_ops_type == "sgemm" || gst_ops_type == "dgemm" || gst_ops_type == "hgemm") &&
      gst_beta_val != 0.0f) {
    msg = "[" + action_name + "] " + "[GPU:: " + std::to_string(gpu_id) + "] " +
      "WARNING: crc_check with copy_matrix:false and beta != 0 on " + gst_ops_type +
      " will produce false-positive CRC errors (C matrix changes each iteration)."
      " Set beta:0 or copy_matrix:true for reliable CRC detection.";
    rvs::lp::Log(msg, rvs::logresults);
  }

  // Warn when crc_check is enabled for fp4/fp6/mxfp8 without out_data_type
  // set to fp32_r.  These types output float32 from the GEMM, but if
  // out_data_type is left empty the output layout defaults to the sub-byte
  // input type, making the CRC byte-count wrong.  compute_output_crc() will
  // return -1 in that case; raise a clear message here so users don't have
  // to hunt for the cause.
  if (crc_check &&
      (gst_data_type == "fp4_r"        ||
       gst_data_type == "fp6_e3m2_r"   ||
       gst_data_type == "fp6_e2m3_r"   ||
       gst_data_type == "mxfp8_e4m3_r" ||
       gst_data_type == "mxfp8_e5m2_r") &&
      gst_out_data_type != "fp16_r" &&
      gst_out_data_type != "bf16_r" &&
      gst_out_data_type != "fp32_r") {
    msg = "[" + action_name + "] " + "[GPU:: " + std::to_string(gpu_id) + "] " +
      "WARNING: crc_check on " + gst_data_type +
      " requires out_data_type: fp16_r (or bf16_r/fp32_r) — CRC will be skipped "
      "each iteration. Add 'out_data_type: fp16_r' to the config.";
    rvs::lp::Log(msg, rvs::logresults);
  }

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

    // launch GEMM operation
    if(!gpu_blas->run_blas_gemm(gst_hot_calls)) {

      *err_description = GST_BLAS_ERROR;
      *error = 1;
      return false;
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
    total_microseconds = time_diff(gst_end_time, gst_start_time);

    log_interval_microseconds = time_diff(gst_end_time,
        gst_log_interval_time);

    if ((log_interval_microseconds >= log_interval * 1000u || 0 == run_duration_ms) && num_gemm_ops > 0) {

      seconds_elapsed = static_cast<double> (log_interval_microseconds) / 1000000;

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

    if (crc_check) {
      int crc_result = gpu_blas->compute_output_crc();
      if (crc_result >= 0) {
        crc_computed = true;   // at least one CRC value is now available

        // Log the CRC value at trace level so it is visible with -d 5 or
        // equivalent verbose logging, without flooding normal output.
        std::ostringstream crc_oss;
        crc_oss << "[" << action_name << "] [GPU:: " << gpu_id << "] "
                << "crc-check: 0x"
                << std::hex << std::setw(8) << std::setfill('0')
                << gpu_blas->get_last_output_crc()
                << (crc_result == 1 ? " [MISMATCH]" : " [OK]");
        rvs::lp::Log(crc_oss.str(), crc_result == 1 ? rvs::logresults : rvs::loginfo);
      }

      if (crc_result == 1) {
        msg = "[" + action_name + "] " + "[GPU:: " + std::to_string(gpu_id) + "] " +
          GST_LOG_CRC_CHECK_ERROR_KEY + " CRC-32 mismatch detected (SDC candidate)";
        rvs::lp::Log(msg, rvs::logresults);

        uint32_t damaged_bytes   = 0;
        uint32_t total_bytes_out = 0;
        double   fnorm           = -1.0;
        gpu_blas->compute_mismatch_magnitude(damaged_bytes, total_bytes_out, fnorm);

        std::ostringstream mag_oss;
        mag_oss << "[" << action_name << "] [GPU:: " << gpu_id << "] "
                << GST_LOG_CRC_CHECK_ERROR_KEY
                << " mismatch magnitude:"
                << " damaged_bytes=" << damaged_bytes
                << "/" << total_bytes_out;
        if (fnorm >= 0.0)
          mag_oss << " frobenius_norm=" << std::scientific << std::setprecision(6) << fnorm;
        else
          mag_oss << " frobenius_norm=n/a";
        rvs::lp::Log(mag_oss.str(), rvs::logresults);
      } else if (crc_result == -1) {
        msg = "[" + action_name + "] " + "[GPU:: " + std::to_string(gpu_id) + "] " +
          GST_LOG_CRC_CHECK_ERROR_KEY + " internal error (unsupported type or copy failed)";
        rvs::lp::Log(msg, rvs::logresults);
        // Remove this worker from the cross-GPU barrier immediately so peers
        // are not left waiting for the remainder of the stress-test duration.
        // CrcSyncGuard will see crc_sync == nullptr and skip the redundant
        // detach() call on function exit.
        if (crc_sync) {
          crc_sync->detach(crc_sync_slot);
          crc_sync        = nullptr;
          crc_guard.s     = nullptr;
        }
      }

      // Per-iteration cross-GPU barrier: all workers rendezvous here,
      // the last arrival compares all CRCs and logs any mismatch.
      if (crc_sync && crc_result >= 0)
        crc_sync->sync_and_compare(crc_sync_slot, gpu_id,
                                   gpu_blas->get_last_output_crc());
    }

    msg = "[" + action_name + "] " + MODULE_NAME + " " +
      std::to_string(gpu_id) + " " + GST_START_MSG + " " +
      " Execution time in microseconds :" + std::to_string(total_microseconds) +
      " run_duration_ms :" + std::to_string(run_duration_ms);
    rvs::lp::Log(msg, rvs::logtrace);

    if (0 == run_duration_ms || total_microseconds >= run_duration_ms * 1000u)
      break;
  }

  // Log the final run-level CRC digest for this GPU.  The digest is a single
  // CRC-32 value built by chaining every per-iteration output CRC, so it
  // reflects the entire run history — not just the last iteration.
  if (crc_check && crc_computed) {
    std::ostringstream oss;
    oss << "[" << action_name << "] [GPU:: " << gpu_id << "] "
        << "crc-check run-digest: 0x"
        << std::hex << std::setw(8) << std::setfill('0')
        << gpu_blas->get_run_crc_digest();
    rvs::lp::Log(oss.str(), rvs::loginfo);
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
  char gpuid_buff[12];
  auto desc = action_descriptor{action_name, MODULE_NAME, gpu_id};
  max_gflops = 0;

  snprintf(gpuid_buff, sizeof(gpuid_buff), "%5d", gpu_id);

  // Guard covering ramp-failure early-exit paths.  If the GPU fails ramp and
  // run() returns before do_gst_stress_test() is entered, this ensures
  // crc_sync->detach() is called so peer workers are not left deadlocked at
  // the per-iteration CRC barrier.  Disarmed (s = nullptr) just before
  // do_gst_stress_test() is called because that function has its own
  // CrcSyncGuard and must not be double-detached.
  struct CrcSyncGuard {
    GstCrcSync* s;
    size_t      slot;
    ~CrcSyncGuard() { if (s) s->detach(slot); }
  } ramp_guard{crc_sync, crc_sync_slot};

  // log GST stress test - start message
  msg = "[" + action_name + "] " + MODULE_NAME + " " +
    "[GPU:: " + gpuid_buff + "] "  + " " + GST_START_MSG + " " +
    " Starting the GST stress test ";
  rvs::lp::Log(msg, rvs::logtrace);

  // log GST ramp up - start message
  msg = "[" + action_name + "] " + "[GPU:: " + gpuid_buff + "] " +
    "Start of GPU ramp up";
  rvs::lp::Log(msg, rvs::logresults);

  // let the GPU ramp-up and check the result
  bool ramp_up_success = do_gst_ramp(&error, &err_description);

  // log GST ramp up - end message
  msg = "[" + action_name + "] " + "[GPU:: " + gpuid_buff + "] " +
    "End of GPU ramp up";
  rvs::lp::Log(msg, rvs::logresults);

  // GPU was not able to do the processing (HIP/rocBlas error(s) occurred)
  if (error) {
    string msg = "[" + action_name + "] " + MODULE_NAME + " "
      + std::to_string(gpu_id) + " " + err_description;
    rvs::lp::Log(msg, rvs::logerror);
    if (bjson)
        log_to_json(desc ,rvs::logerror,"err", err_description);

    action_result.state = rvs::actionstate::ACTION_COMPLETED;
    action_result.status = rvs::actionstatus::ACTION_FAILED;
    action_result.output = msg.c_str();
    action.action_callback(&action_result);

    return;   // ramp_guard destructs here → detach() called
  }

  // Ramp succeeded — do_gst_stress_test() has its own CrcSyncGuard.
  // Disarm the ramp guard to prevent a double-detach.
  ramp_guard.s = nullptr;

  // the GPU succeeded to achieve the target_stress GFLOPS
  // continue with the same workload for the rest of the test duration
  msg = "[" + action_name + "] " + MODULE_NAME + " " +
    std::to_string(gpu_id) + " " + " GST ramp completed for interval :" + " " +
    std::to_string(ramp_interval);
  rvs::lp::Log(msg, rvs::loginfo);

  gst_test_passed = do_gst_stress_test(&error, &err_description);
  // check if stop signal was received
  if (rvs::lp::Stopping())
    return;

  if (error) {
    // GPU didn't complete the test (HIP/rocBlas error(s) occurred)
    string msg = "[" + action_name + "] " + MODULE_NAME + " " +
      std::to_string(gpu_id) + " " + err_description;
    rvs::lp::Log(msg, rvs::logerror);
    if (bjson)
      log_to_json(desc, rvs::logerror,"err", err_description);

    action_result.state = rvs::actionstate::ACTION_COMPLETED;
    action_result.status = rvs::actionstatus::ACTION_FAILED;
    action_result.output = msg.c_str();
    action.action_callback(&action_result);

    return;
  }

  check_target_stress(max_gflops);
}

/**
 * @brief computes the difference (in microseconds) between 2 points in time
 * @param t_end second point in time
 * @param t_start first point in time
 * @return time difference in microseconds
 */
uint64_t GSTWorker::time_diff(
    std::chrono::time_point<std::chrono::system_clock> t_end,
    std::chrono::time_point<std::chrono::system_clock> t_start) {
  auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(
      t_end - t_start);
  return microseconds.count();
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

