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

#include <unistd.h>
#include <chrono>
#include <map>
#include <string>
#include <algorithm>
#include <iostream>
#include <memory>

#include "rvs_blas.h"
#include "rvsliblogger.h"
#include "rvs_module.h"
#include "rvsloglp.h"

#define MODULE_NAME                             "gst"

#define GST_MEM_ALLOC_ERROR                     "memory allocation error!"
#define GST_BLAS_ERROR                          "memory/blas error!"
#define GST_BLAS_MEMCPY_ERROR                   "HostToDevice mem copy error!"

#define GST_MAX_GFLOPS_OUTPUT_KEY               "Gflop"
#define GST_FLOPS_PER_OP_OUTPUT_KEY             "flops_per_op"
#define GST_BYTES_COPIED_PER_OP_OUTPUT_KEY      "bytes_copied_per_op"
#define GST_TRY_OPS_PER_SEC_OUTPUT_KEY          "try_ops_per_sec"

#define GST_LOG_GFLOPS_INTERVAL_KEY             "Gflops"
#define GST_JSON_LOG_GPU_ID_KEY                 "gpu_id"

#define NMAX_MS_GPU_RUN_PEAK_PERFORMANCE        500
#define NMAX_MS_SGEMM_OPS_RAMP_SUB_INTERVAL     500
#define NUM_US_SECONDS_DELAY_INC_DEC            500

using std::string;

bool GSTWorker::bjson = false;


GSTWorker::GSTWorker() {}
GSTWorker::~GSTWorker() {}

/**
 * @brief performs the rvsBlas setup
 * @param error pointer to a memory location where the error code will be stored
 * @param err_description will store the error description if any
 */
void GSTWorker::setup_blas(int *error, string *err_description) {
    *error = 0;
    // setup rvsBlas
    gpu_blas = std::unique_ptr<rvs_blas>(
        new rvs_blas(gpu_device_index, matrix_size, matrix_size, matrix_size));

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
 * @brief hits the maximum Gflops value
 * @param error pointer to a memory location where the error code will be stored
 * @param err_description will store the error description if any
 */
void GSTWorker::hit_max_gflops(int *error, string *err_description) {
    std::chrono::time_point<std::chrono::system_clock>
                                    gst_start_time,
                                    gst_end_time,
                                    gst_log_interval_time;

    double seconds_elapsed = 0, curr_gflops;
    uint16_t num_sgemm_ops = 0, num_sgemm_ops_log_interval = 0;
    uint64_t millis_sgemm_ops;
    bool sgemm_ok;
    string msg;

    *error = 0;
    max_gflops = 0;
    gst_start_time = std::chrono::system_clock::now();
    gst_log_interval_time = std::chrono::system_clock::now();
    for (;;) {
        // useful guard in case gpu_blas->run_blass_gemm() keeps failing
        gst_end_time = std::chrono::system_clock::now();
        if (time_diff(gst_end_time,  gst_start_time) >
                            NMAX_MS_GPU_RUN_PEAK_PERFORMANCE) {
            break;
        }

        if (copy_matrix) {
            // copy matrix before each GEMM
            if (!gpu_blas->copy_data_to_gpu()) {
                *error = 1;
                *err_description = GST_BLAS_MEMCPY_ERROR;
                return;
            }
        }

        // run GEMM & wait for completion
        if (!gpu_blas->run_blass_gemm())
            continue;  // failed to run the current SGEMM

        sgemm_ok = true;
        while (!gpu_blas->is_gemm_op_complete()) {
            gst_end_time = std::chrono::system_clock::now();
            if (time_diff(gst_end_time, gst_start_time) >
                            NMAX_MS_GPU_RUN_PEAK_PERFORMANCE) {
                sgemm_ok = false;
                break;
            }
        }

        if (!sgemm_ok)
            break;

        num_sgemm_ops++;
        num_sgemm_ops_log_interval++;

        gst_end_time = std::chrono::system_clock::now();
        millis_sgemm_ops =
                    time_diff(gst_end_time, gst_log_interval_time);

        if (millis_sgemm_ops >= log_interval) {
            // compute the GFLOPS
            seconds_elapsed = static_cast<double>
                                (millis_sgemm_ops) / 1000;
            curr_gflops = static_cast<double>(gpu_blas->gemm_gflop_count() *
                                num_sgemm_ops_log_interval) / seconds_elapsed;
            if (curr_gflops > max_gflops)
                max_gflops = curr_gflops;

            // log gflops for this interval
            msg = action_name + " " + MODULE_NAME + " " +
                    std::to_string(gpu_id) + " " +
                    GST_LOG_GFLOPS_INTERVAL_KEY + " " +
                    std::to_string(curr_gflops);
            log(msg.c_str(), rvs::loginfo);

            log_to_json(GST_LOG_GFLOPS_INTERVAL_KEY,
                        std::to_string(curr_gflops), rvs::loginfo);
            num_sgemm_ops_log_interval = 0;
            gst_log_interval_time = std::chrono::system_clock::now();
        }
    }

    // compute the Gflops for the NMAX_MS_GPU_RUN_PEAK_PERFORMANCE ms period
    gst_end_time = std::chrono::system_clock::now();
    seconds_elapsed = static_cast<double>
                                    (NMAX_MS_GPU_RUN_PEAK_PERFORMANCE) / 1000;
    curr_gflops = static_cast<double>(gpu_blas->gemm_gflop_count() *
                                    num_sgemm_ops) / seconds_elapsed;
    if (curr_gflops > max_gflops) {
        max_gflops = curr_gflops;
    }
}

/**
 * @brief performs the rampup stress test on the given GPU
 * @param error pointer to a memory location where the error code will be stored
 * @param err_description will store the error description if any
 * @return true if target_stress is achieved within ramp_interval, false otherwise
 */
bool GSTWorker::do_gst_ramp(int *error, string *err_description) {
    std::chrono::time_point<std::chrono::system_clock>
                                    gst_start_time,
                                    gst_end_time,
                                    gst_log_interval_time,
                                    gst_start_gflops_time;

    double seconds_elapsed = 0, curr_gflops;
    uint16_t num_sgemm_ops = 0, num_sgemm_ops_log_interval = 0;
    uint64_t millis_sgemm_ops;
    string msg;

    delay_target_stress = 0;

    // make sure that the ramp_interval & duration are not less than
    // NMAX_MS_GPU_RUN_PEAK_PERFORMANCE (e.g.: 500)
    if (run_duration_ms < NMAX_MS_GPU_RUN_PEAK_PERFORMANCE)
        run_duration_ms += NMAX_MS_GPU_RUN_PEAK_PERFORMANCE;
    if (ramp_interval < NMAX_MS_GPU_RUN_PEAK_PERFORMANCE)
        ramp_interval += NMAX_MS_GPU_RUN_PEAK_PERFORMANCE;

    // stage 1.
    // setup rvs blas
    setup_blas(error, err_description);
    if (*error)
        return false;

    // stage 2.
    // run the SGEMM with the given matrix_size for about
    // NMAX_MS_GPU_RUN_PEAK_PERFORMANCE ms (e.g.: 500) in order to hit a
    // maximum Gflops value
    hit_max_gflops(error, err_description);
    if (*error)
        return false;

    // stage 3.
    // reduce the SGEMM frequency and try to achieve the desired Gflops
    if (target_stress > max_gflops) {
        // target stress is bigger than the peak performance
        return false;
    }

    // compute frequency to reach the given target stress
    double ms_per_op_max_glops = (1000.0 /
                            (max_gflops / gpu_blas->gemm_gflop_count()));
    double ms_per_target_stress = ms_per_op_max_glops *
                            (max_gflops / target_stress);

    delay_target_stress = (ms_per_target_stress - ms_per_op_max_glops) * 1000;

    gst_start_time = std::chrono::system_clock::now();
    gst_log_interval_time = std::chrono::system_clock::now();
    gst_start_gflops_time = std::chrono::system_clock::now();

    for (;;) {
        // useful guard in case gpu_blas->run_blass_gemm() keeps failing
        gst_end_time = std::chrono::system_clock::now();
        if (time_diff(gst_end_time,  gst_start_time) >
                            ramp_interval - NMAX_MS_GPU_RUN_PEAK_PERFORMANCE) {
            return false;
        }

        if (copy_matrix) {
            // copy matrix before each GEMM
            if (!gpu_blas->copy_data_to_gpu()) {
                *error = 1;
                *err_description = GST_BLAS_MEMCPY_ERROR;
                return false;
            }
        }

        // run GEMM & wait for completion
        if (!gpu_blas->run_blass_gemm())
            continue;  // failed to run the current SGEMM

        while (!gpu_blas->is_gemm_op_complete()) {
            gst_end_time = std::chrono::system_clock::now();
            if (time_diff(gst_end_time, gst_start_time) >
                            ramp_interval - NMAX_MS_GPU_RUN_PEAK_PERFORMANCE) {
                return false;
            }
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
            curr_gflops = static_cast<double>(gpu_blas->gemm_gflop_count() *
                                num_sgemm_ops) / seconds_elapsed;

            if (curr_gflops > target_stress - target_stress * tolerance &&
                curr_gflops < target_stress + target_stress * tolerance) {
                ramp_actual_time = time_diff(gst_end_time,  gst_start_time) +
                                        NMAX_MS_GPU_RUN_PEAK_PERFORMANCE;
                return true;
            }

            if (curr_gflops > target_stress)
                delay_target_stress += NUM_US_SECONDS_DELAY_INC_DEC;
            else
                delay_target_stress -= NUM_US_SECONDS_DELAY_INC_DEC;

            num_sgemm_ops = 0;
            gst_start_gflops_time = std::chrono::system_clock::now();
            continue;
        }

        millis_sgemm_ops =
                    time_diff(gst_end_time, gst_log_interval_time);
        if (millis_sgemm_ops >= log_interval) {
            // compute the GFLOPS
            seconds_elapsed = static_cast<double>
                                (millis_sgemm_ops) / 1000;
            curr_gflops = static_cast<double>(gpu_blas->gemm_gflop_count() *
                                num_sgemm_ops_log_interval) / seconds_elapsed;
            if (curr_gflops > max_gflops)
                max_gflops = curr_gflops;

            // log gflops for this interval
            msg = action_name + " " + MODULE_NAME + " " +
                    std::to_string(gpu_id) + " " +
                    GST_LOG_GFLOPS_INTERVAL_KEY + " " +
                    std::to_string(curr_gflops);
            log(msg.c_str(), rvs::loginfo);

            log_to_json(GST_LOG_GFLOPS_INTERVAL_KEY,
                        std::to_string(curr_gflops), rvs::loginfo);
            num_sgemm_ops_log_interval = 0;
            gst_log_interval_time = std::chrono::system_clock::now();
            continue;
        }

        usleep(delay_target_stress);
    }

    return false;
}
/**
 * @brief performs the stress test on the given GPU
 * @param error pointer to a memory location where the error code will be stored
 * @param err_description will store the error description if any 
 * @return true if stress violations is less than max_violations, false otherwise
 */
bool GSTWorker::do_gst_stress_test(int *error, std::string *err_description) {
    bool gpu_stress_test_finished = false;
    uint64_t num_sgemm_ops = 0, num_gflops_violations = 0;
    uint64_t total_milliseconds = 0, log_interval_milliseconds = 0;
    double seconds_elapsed = 0, gflops_interval = 0;
    string msg;
    std::chrono::time_point<std::chrono::system_clock> gst_start_time,
                                    gst_end_time, gst_log_interval_time;

    *error = 0;

    // continue with the same workload for the rest of the duration
    num_sgemm_ops = 0;
    gst_start_time = std::chrono::system_clock::now();
    gst_log_interval_time = std::chrono::system_clock::now();

    while (!gpu_stress_test_finished) {
        if (copy_matrix) {
            // copy matrix before each GEMM
            if (!gpu_blas->copy_data_to_gpu()) {
                *error = 1;
                *err_description = GST_BLAS_MEMCPY_ERROR;
                return false;
            }
        }

        bool sgemm_success = true;
        // run GEMM & wait for completion
        if (gpu_blas->run_blass_gemm()) {
            while (!gpu_blas->is_gemm_op_complete()) {
                // add guard to avoid deadlocks
                gst_end_time = std::chrono::system_clock::now();
                total_milliseconds = time_diff(gst_end_time, gst_start_time);
                if (total_milliseconds > run_duration_ms - ramp_actual_time) {
                    sgemm_success = false;
                    break;
                }
            }
        } else {
            sgemm_success = false;
        }

        if (sgemm_success)
            num_sgemm_ops++;

        log_interval_milliseconds = time_diff(gst_end_time,
                                              gst_log_interval_time);

        if (log_interval_milliseconds >= log_interval && num_sgemm_ops > 0) {
            seconds_elapsed =
                    static_cast<double>
                        (log_interval_milliseconds) / 1000;

            if (seconds_elapsed != 0) {
                gflops_interval =
                        static_cast<double>(gpu_blas->gemm_gflop_count()
                            * num_sgemm_ops) / seconds_elapsed;
                if (gflops_interval > max_gflops)
                    max_gflops = gflops_interval;

                // log gflops for this interval
                msg = action_name + " " + MODULE_NAME + " " +
                        std::to_string(gpu_id) + " " +
                        GST_LOG_GFLOPS_INTERVAL_KEY + " " +
                        std::to_string(gflops_interval);
                log(msg.c_str(), rvs::loginfo);

                log_to_json(GST_LOG_GFLOPS_INTERVAL_KEY,
                            std::to_string(gflops_interval), rvs::loginfo);

                // check for gflops tolerance violation
                if (!(gflops_interval >
                            target_stress - target_stress * tolerance &&
                    gflops_interval <
                            target_stress + target_stress * tolerance)) {
                        num_gflops_violations++;

                    msg = action_name + " " + MODULE_NAME + " " +
                            std::to_string(gpu_id) +
                            " stress violation " +
                            std::to_string(gflops_interval);
                    log(msg.c_str(), rvs::loginfo);

                    log_to_json("stress violation",
                                std::to_string(gflops_interval), rvs::loginfo);
                }

                // reset time & gflops related data
                num_sgemm_ops = 0;
                gst_log_interval_time =
                            std::chrono::system_clock::now();
            }
        }

        if (total_milliseconds >= run_duration_ms - ramp_actual_time)
            gpu_stress_test_finished = true;
        else
            usleep(delay_target_stress);
    }

    if (num_gflops_violations > max_violations)
        return false;

    return true;
}

/**
 * @brief performs the stress test on the given GPU
 */
void GSTWorker::run() {
    string msg, err_description;
    int error = 0;
    bool gst_test_passed;

    max_gflops = 0;

    // log GST stress test - start
    msg = action_name + " " + MODULE_NAME + " " +
            std::to_string(gpu_id) + " start " +
            std::to_string(target_stress) +
            " copy matrix:" + (copy_matrix ? "true":"false");
    log(msg.c_str(), rvs::loginfo);

    log_to_json("copy matrix", (copy_matrix ? "true":"false"), rvs::loginfo);

    // let the GPU ramp-up and check the result
    bool ramp_up_success = do_gst_ramp(&error, &err_description);

    // GPU was not able to do the processing (HIP/rocBlas error(s) occurred)
    if (error) {
        string msg = action_name + " " + MODULE_NAME + " "
                        + std::to_string(gpu_id) + " "
                        + err_description;
        log(msg.c_str(), rvs::logerror);
        log_to_json("err", err_description, rvs::logerror);

        return;
    }

    if (!ramp_up_success) {
        // the selected GPU was not able to achieve the target_stress GFLOPS
        msg = action_name + " " + MODULE_NAME + " " +
        std::to_string(gpu_id) + " ramp time exceeded " +
        std::to_string(ramp_interval);
        log(msg.c_str(), rvs::loginfo);
        log_to_json("ramp time exceeded",
                    std::to_string(ramp_interval), rvs::loginfo);
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
        log_to_json("target achieved",
                    std::to_string(ramp_interval), rvs::loginfo);
        if (run_duration_ms > 0) {
            gst_test_passed = do_gst_stress_test(&error, &err_description);
            // GPU was not able to do the processing
            // (HIP/rocBlas error(s) occurred)
            if (error) {
                string msg = action_name + " " + MODULE_NAME + " "
                                + std::to_string(gpu_id) + " "
                                + err_description;
                log(msg.c_str(), rvs::logerror);
                log_to_json("err", err_description, rvs::logerror);
                return;
            }
        }
    }
    double flops_per_op = (2 * (static_cast<double>(gpu_blas->get_m())/1000) *
                                (static_cast<double>(gpu_blas->get_n())/1000) *
                                (static_cast<double>(gpu_blas->get_k())/1000));
    msg = action_name + " " + MODULE_NAME + " " + std::to_string(gpu_id)+
        " " + GST_MAX_GFLOPS_OUTPUT_KEY + ": "+ std::to_string(max_gflops) +
        " " + GST_FLOPS_PER_OP_OUTPUT_KEY + ": " +
        std::to_string(flops_per_op) + "x1e9" +
        " " + GST_BYTES_COPIED_PER_OP_OUTPUT_KEY + ": " +
        std::to_string(gpu_blas->get_bytes_copied_per_op()) +
        " " + GST_TRY_OPS_PER_SEC_OUTPUT_KEY + ": "+
        std::to_string(target_stress / gpu_blas->gemm_gflop_count()) +
        " pass: " +
        (gst_test_passed ? GST_RESULT_PASS_MESSAGE : GST_RESULT_FAIL_MESSAGE);
    log(msg.c_str(), rvs::logresults);

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
    log_to_json("pass", (gst_test_passed ?
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
        unsigned int sec;
        unsigned int usec;

        rvs::lp::get_ticks(sec, usec);
        void *json_node = rvs::lp::LogRecordCreate(MODULE_NAME,
                            action_name.c_str(), log_level,
                            sec, usec);
        if (json_node) {
            rvs::lp::AddString(json_node, GST_JSON_LOG_GPU_ID_KEY,
                            std::to_string(gpu_id));
            rvs::lp::AddString(json_node, key, value);
            rvs::lp::LogRecordFlush(json_node);
        }
    }
}
