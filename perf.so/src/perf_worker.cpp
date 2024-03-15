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
#include "include/perf_worker.h"

#include <unistd.h>
#include <string>
#include <memory>
#include <iostream>

#include "include/rvs_blas.h"
#include "include/rvs_module.h"
#include "include/rvsloglp.h"

#define MODULE_NAME                             "perf"

#define PERF_MEM_ALLOC_ERROR                     "memory allocation error!"
#define PERF_BLAS_ERROR                          "memory/blas error!"
#define PERF_BLAS_MEMCPY_ERROR                   "HostToDevice mem copy error!"

#define PERF_MAX_GFLOPS_OUTPUT_KEY               "Gflop"
#define PERF_FLOPS_PER_OP_OUTPUT_KEY             "flops_per_op"
#define PERF_BYTES_COPIED_PER_OP_OUTPUT_KEY      "bytes_copied_per_op"
#define PERF_TRY_OPS_PER_SEC_OUTPUT_KEY          "try_ops_per_sec"

#define PERF_LOG_GFLOPS_INTERVAL_KEY             "Gflops"
#define PERF_JSON_LOG_GPU_ID_KEY                 "gpu_id"

#define PROC_DEC_INC_SGEMM_FREQ_DELAY           10

#define NMAX_MS_GPU_RUN_PEAK_PERFORMANCE        1000
#define NMAX_MS_SGEMM_OPS_RAMP_SUB_INTERVAL     1000
#define USLEEP_MAX_VAL                          (1000000 - 1)

#define PERF_COPY_MATRIX_MSG                     "copy matrix"
#define PERF_START_MSG                           "start"
#define PERF_PASS_KEY                            "pass"
#define PERF_RAMP_EXCEEDED_MSG                   "ramp time exceeded"
#define PERF_TARGET_ACHIEVED_MSG                 "target achieved"
#define PERF_STRESS_VIOLATION_MSG                "stress violation"

using std::string;

bool PERFWorker::bjson = false;

PERFWorker::PERFWorker() {}
PERFWorker::~PERFWorker() {}

/**
 * @brief performs the rvsBlas setup
 * @param error pointer to a memory location where the error code will be stored
 * @param err_description stores the error description if any
 */
void PERFWorker::setup_blas(int *error, string *err_description) {
    *error = 0;
    // setup rvsBlas
    gpu_blas = std::unique_ptr<rvs_blas>(
        new rvs_blas(gpu_device_index, matrix_size_a, matrix_size_b,
                        matrix_size_c, perf_trans_a, perf_trans_b,
                        perf_alpha_val, perf_beta_val, 
                        perf_lda_offset, perf_ldb_offset, perf_ldc_offset, perf_ops_type, ""));

    if (!gpu_blas) {
        *error = 1;
        *err_description = PERF_MEM_ALLOC_ERROR;
        return;
    }

    if (gpu_blas->error()) {
        *error = 1;
        *err_description = PERF_MEM_ALLOC_ERROR;
        return;
    }

    // generate random matrix & copy it to the GPU
    gpu_blas->generate_random_matrix_data();
    if (!copy_matrix) {
        // copy matrix only once
        if (!gpu_blas->copy_data_to_gpu(perf_ops_type)) {
            *error = 1;
            *err_description = PERF_BLAS_MEMCPY_ERROR;
        }
    }
}


/**
 * @brief logs the Gflops computed over the last log_interval period 
 * @param gflops_interval the Gflops that the GPU achieved
 */
void PERFWorker::check_target_stress(double gflops_interval) {
    string msg;
    bool result;
    rvs::action_result_t action_result;

    if(gflops_interval >= target_stress){
           result = true;
    }else{
           result = false;
    }

    msg = "[" + action_name + "] " + MODULE_NAME + " " +
              std::to_string(gpu_id) + " " + PERF_LOG_GFLOPS_INTERVAL_KEY + " " + std::to_string(gflops_interval) + " " +
              "Target stress :" + " " + std::to_string(target_stress) + " met :" + (result ? "TRUE" : "FALSE");
    rvs::lp::Log(msg, rvs::logresults);

    action_result.state = rvs::actionstate::ACTION_RUNNING;
    action_result.status = (true == result) ? rvs::actionstatus::ACTION_SUCCESS : rvs::actionstatus::ACTION_FAILED;
    action_result.output = msg.c_str();
    action.action_callback(&action_result);

    log_to_json(PERF_LOG_GFLOPS_INTERVAL_KEY, std::to_string(gflops_interval),
                rvs::loginfo);
}

/**
 * @brief logs the Gflops computed over the last log_interval period 
 * @param gflops_interval the Gflops that the GPU achieved
 */
void PERFWorker::log_interval_gflops(double gflops_interval) {
    string msg;
    rvs::action_result_t action_result;

    msg = "[" + action_name + "] " + MODULE_NAME + " " +
            std::to_string(gpu_id) + " " + PERF_LOG_GFLOPS_INTERVAL_KEY + " " +
            std::to_string(gflops_interval);
    rvs::lp::Log(msg, rvs::logresults);

    action_result.state = rvs::actionstate::ACTION_RUNNING;
    action_result.status = rvs::actionstatus::ACTION_SUCCESS;
    action_result.output = msg.c_str();
    action.action_callback(&action_result);

    log_to_json(PERF_LOG_GFLOPS_INTERVAL_KEY, std::to_string(gflops_interval),
                rvs::loginfo);
}


/**
 * @brief performs the stress test on the given GPU
 * @param error pointer to a memory location where the error code will be stored
 * @param err_description stores the error description if any
 * @return true if stress violations is less than max_violations, false otherwise
 */
bool PERFWorker::do_perf_stress_test(int *error, std::string *err_description) {
    uint16_t num_gemm_ops = 0;
    double start_time, end_time;
    double timetaken;
    string msg;

    *error = 0;
    max_gflops = 0;
    num_gemm_ops = 0;
    start_time = 0;
    end_time = 0;

    //Start the timer
    start_time = gpu_blas->get_time_us();

    while(num_gemm_ops++ <= perf_hot_calls) { 
        // run GEMM & wait for completion
        gpu_blas->run_blass_gemm(perf_ops_type);
    }

    //End the timer
    end_time = gpu_blas->get_time_us();

    //Converting microseconds to seconds
    timetaken = (end_time - start_time)/1e6;

    max_gflops =  static_cast<double> ((gpu_blas->gemm_gflop_count() * perf_hot_calls)/timetaken) ;

    log_interval_gflops(max_gflops);

    return true;
}

/**
 * @brief performs the stress test on the given GPU
 */
void PERFWorker::run() {
    string msg, err_description;
    int error = 0;
    bool perf_test_passed = true;

    max_gflops = 0;

    // log PERF stress test - start message
    msg = "[" + action_name + "] " + MODULE_NAME + " " +
            std::to_string(gpu_id) + " " + PERF_START_MSG + " " +
            " Starting the PERF stress test "; 
    rvs::lp::Log(msg, rvs::logtrace);

    log_to_json(PERF_START_MSG, std::to_string(target_stress), rvs::loginfo);
    log_to_json(PERF_COPY_MATRIX_MSG, (copy_matrix ? "true":"false"),
                rvs::loginfo);

    // stage 1. setup rvs blas
    setup_blas(&error, &err_description);
    if (error)
        return;

    if (run_duration_ms > 0) {
            perf_test_passed = do_perf_stress_test(&error, &err_description);
            // check if stop signal was received
            if (rvs::lp::Stopping())
                return;

            if (error) {
                // GPU didn't complete the test (HIP/rocBlas error(s) occurred)
                string msg = "[" + action_name + "] " + MODULE_NAME + " " +
                                std::to_string(gpu_id) + " " + err_description;
                rvs::lp::Log(msg, rvs::logerror);
                log_to_json("err", err_description, rvs::logerror);
                return;
            }
    }

    log_interval_gflops(max_gflops);
    check_target_stress(max_gflops);
}


/**
 * @brief computes the difference (in milliseconds) between 2 points in time
 * @param t_end second point in time
 * @param t_start first point in time
 * @return time difference in milliseconds
 */
uint64_t PERFWorker::time_diff(
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
void PERFWorker::log_to_json(const std::string &key, const std::string &value,
                     int log_level) {
    if (PERFWorker::bjson) {
        unsigned int sec;
        unsigned int usec;

        rvs::lp::get_ticks(&sec, &usec);
        void *json_node = rvs::lp::LogRecordCreate(MODULE_NAME,
                            action_name.c_str(), log_level, sec, usec);
        if (json_node) {
            rvs::lp::AddString(json_node, PERF_JSON_LOG_GPU_ID_KEY,
                            std::to_string(gpu_id));
            rvs::lp::AddString(json_node, key, value);
            rvs::lp::LogRecordFlush(json_node);
        }
    }
}

/**
 * @brief extends the usleep for more than 1000000us
 * @param microseconds us to sleep
 */
void PERFWorker::usleep_ex(uint64_t microseconds) {
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
