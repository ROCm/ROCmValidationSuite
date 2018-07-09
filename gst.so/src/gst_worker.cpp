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
#include "gst_worker.h"

#include <chrono>
#include <map>
#include <string>
#include <algorithm>
#include <iostream>

#include "rvs_blas.h"
#include "rvsliblogger.h"
#include "rvs_module.h"
#include "rvsloglp.h"

#define MODULE_NAME                                 "gst"

#define GST_GPU_RUN_ERROR                           "stress test cannot run "\
                                                    "at the moment! Please "\
                                                    "try again later!"

#define GST_MAX_GFLOPS_OUTPUT_KEY                   "Gflops"
#define GST_FLOPS_PER_OP_OUTPUT_KEY                 "flops_per_op"
#define GST_BYTES_COPIED_PER_OP_OUTPUT_KEY          "bytes_copied_per_op"
#define GST_TRY_OPS_PER_SEC_OUTPUT_KEY              "try_ops_per_sec"

using std::string;

GSTWorker::GSTWorker() {}
GSTWorker::~GSTWorker() {}

/**
 * @brief logs an error & the GST module's result as FALSE
 * @param error the actual error message
 */
void GSTWorker::log_module_error(const string &error) {
    string msg = action_name + " " + MODULE_NAME + " "
                    + std::to_string(gpu_id) + " "
                    + error;
    log(msg.c_str(), rvs::logerror);

    // log the module's result (FALSE) and abort GST
    // in case of error,the Gflops/flops_per_op/bytes_copied_per_op
    // and try_ops_per_sec won't be sent to the log
    msg = action_name + " " + MODULE_NAME + " " + std::to_string(gpu_id)
            + " pass: " + GST_RESULT_FAIL_MESSAGE;
    log(msg.c_str(), rvs::logresults);
}

/**
 * @brief performs the rampup stress test on the given GPU
 * @param error pointer to a memory location where the error code will be stored
 * @return true if target_stress is achieved within 
 * ramp_interval, false otherwise
 */
bool GSTWorker::do_gst_ramp(int *error) {
    bool gpu_rampup_finished = false;
    uint64_t num_sgemm_ops = 0;
    bool ramp_up_success = true;  // TODO(Tudor) put it to false
    std::chrono::time_point<std::chrono::system_clock> gst_start_time,
                                    gst_end_time, gst_log_interval_time;

    *error = 0;
    rvs_blas gpu_blas(gpu_device_index, 6000, 6000, 6000);
    if (!gpu_blas.error()) {
        gpu_blas.generate_random_matrix_data();
        if (!copy_matrix)  // copy matrix only once
            gpu_blas.copy_data_to_gpu();

        gst_start_time = std::chrono::system_clock::now();

        while (!gpu_rampup_finished) {
            if (copy_matrix)  // copy matrix before each GEMM
                gpu_blas.copy_data_to_gpu();

            // run GEMM & wait for completion
            gpu_blas.run_blass_gemm();
            // TODO(Tudor) add guard
            while (!gpu_blas.is_gemm_op_complete()) {}
            if (!gpu_blas.error())
                num_sgemm_ops++;

            gst_end_time = std::chrono::system_clock::now();
            auto milliseconds =
                    std::chrono::duration_cast<std::chrono::milliseconds>
                        (gst_end_time - gst_start_time);
            if (milliseconds.count() >= ramp_interval)
                gpu_rampup_finished = true;
        }

        return ramp_up_success;
    } else {
        log_module_error(GST_GPU_RUN_ERROR);
        *error = 1;
        return false;
    }
}

/**
 * @brief performs the stress test on the given GPU
 * @param error pointer to a memory location where the error code will be stored
 * @return true if stress violations is less than max_violations, false otherwise
 */
bool GSTWorker::do_gst_stress_test(int *error) {
    bool gpu_stress_test_finished = false;
    uint64_t num_sgemm_ops = 0, num_gflops_violations = 0;
    uint64_t total_milliseconds = 0, log_interval_milliseconds = 0;
    double seconds_elapsed = 0, gflops_interval = 0, max_gflops = 0;
    string msg;
    std::chrono::time_point<std::chrono::system_clock> gst_start_time,
                                    gst_end_time, gst_log_interval_time;

    *error = 0;
    rvs_blas gpu_blas(gpu_device_index, 6000, 6000, 6000);
    if (!gpu_blas.error()) {
        gpu_blas.generate_random_matrix_data();
        if (!copy_matrix)  // copy matrix only once
            gpu_blas.copy_data_to_gpu();

        // continue with the same workload for the rest of the duration
        num_sgemm_ops = 0;
        gst_start_time = std::chrono::system_clock::now();
        gst_log_interval_time = std::chrono::system_clock::now();
        while (!gpu_stress_test_finished) {
            if (copy_matrix)  // copy matrix before each GEMM
                gpu_blas.copy_data_to_gpu();
            // run GEMM & wait for completion
            gpu_blas.run_blass_gemm();
            while (!gpu_blas.is_gemm_op_complete()) {}
            if (!gpu_blas.error())
                num_sgemm_ops++;

            gst_end_time = std::chrono::system_clock::now();
            auto total_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>
                    (gst_end_time - gst_start_time);
            total_milliseconds = total_ms.count();

            auto interval_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>
                    (gst_end_time - gst_log_interval_time);
            log_interval_milliseconds = interval_ms.count();

            if (log_interval_milliseconds >= log_interval ||
                            total_milliseconds >
                                run_duration_ms - ramp_interval) {
                seconds_elapsed =
                        static_cast<double>
                            (log_interval_milliseconds) / 1000;

                if (seconds_elapsed != 0) {
                    gflops_interval =
                            static_cast<double>(gpu_blas.gemm_gflop_count()
                                * num_sgemm_ops) / seconds_elapsed;

                    // log gflops for this interval
                    msg = action_name + " " + MODULE_NAME + " " +
                            std::to_string(gpu_id) + " GFlops " +
                            std::to_string(gflops_interval);
                    log(msg.c_str(), rvs::loginfo);

                    // check for gflops tolerance violation
                    if (gflops_interval <
                            target_stress - target_stress * tolerance) {
                        num_gflops_violations++;

                        msg = action_name + " " + MODULE_NAME + " " +
                                std::to_string(gpu_id) +
                                " stress violation " +
                                std::to_string(gflops_interval);
                        log(msg.c_str(), rvs::loginfo);
                    }

                    // reset time & gflops related data
                    num_sgemm_ops = 0;
                    gst_log_interval_time =
                                std::chrono::system_clock::now();
                }
            }

            if (total_milliseconds >=
                    run_duration_ms - ramp_interval)
                gpu_stress_test_finished = true;
        }

        if (num_gflops_violations > max_violations)
            return false;

        return true;
    } else {
        log_module_error(GST_GPU_RUN_ERROR);
        *error = 1;
        return false;
    }
}


/**
 * @brief performs the stress test on the given GPU
 */
void GSTWorker::run() {
    string msg;
    int error;
    bool gst_test_passed;

    // log GST stress test - start
    msg = action_name + " " + MODULE_NAME + " " +
            std::to_string(gpu_id) + " start " +
            std::to_string(target_stress) +
            " copy matrix:" + (copy_matrix ? "true":"false");
    log(msg.c_str(), rvs::loginfo);

    // let the GPU ramp-up and check the result
    bool ramp_up_success = do_gst_ramp(&error);
    // GPU was not able to do the processing (HIP/rocBlas error(s) occurred)
    if (error)
        return;

    if (!ramp_up_success) {
        // the selected GPU was not able to achieve the target_stress GFLOPS
        msg = action_name + " " + MODULE_NAME + " " +
        std::to_string(gpu_id) + " ramp time exceeded " +
        std::to_string(ramp_interval);
        log(msg.c_str(), rvs::loginfo);
        gst_test_passed = false;
    } else {
        // the GPU succeeded to achieve the target_stress GFLOPS
        // continue with the same workload for the rest of the duration
        // let the GPU ramp-up and check the result
        // the selected GPU was not able to achieve the target_stress GFLOPS
        msg = action_name + " " + MODULE_NAME + " " +
        std::to_string(gpu_id) + " target achieved " +
        std::to_string(target_stress);
        log(msg.c_str(), rvs::loginfo);

        gst_test_passed = do_gst_stress_test(&error);
        // GPU was not able to do the processing (HIP/rocBlas error(s) occurred)
        if (error)
            return;
    }

    // log the GST test result (as of now no Gflops/flops_per_op/
    // bytes_copied_per_op and try_ops_per_sec) ... will add them as soon
    // as we get some answers to our questions
    msg = action_name + " " + MODULE_NAME + " " + std::to_string(gpu_id)
        + " pass: " +
        (gst_test_passed ? GST_RESULT_PASS_MESSAGE : GST_RESULT_FAIL_MESSAGE);
    log(msg.c_str(), rvs::logresults);
}
