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

// lazy gflops approach
#define GST_MIN_MATRIX_N_SIZE                   100
#define GST_MIN_MATRIX_M_SIZE                   100
#define GST_MIN_MATRIX_K_SIZE                   100
#define GST_MATRIX_SIZE_INCREMENT               500
#define NUM_MIN_SGEMM_OPS_PER_MATRIX_SIZE       15
#define NUM_MIN_MS_RAMP_SUSTAIN_TARGET_GFLOPS   1000

// greedy gflops approach
#define NUM_INIT_SGEMM_OPS_PER_MATRIX_SIZE      5
#define NUM_INC_SGEMM_OPS_PER_MATRIX_SIZE       3
#define NUM_MAX_TRY_SGEMM_OPS_DIFF_SIZE         10
#define GST_MATRIX_SIZE_PROCENT_DELTA           15
#define NUM_MS_CHECK_GFLOPS                     1000

using std::string;

bool GSTWorker::bjson = false;


GSTWorker::GSTWorker() {}
GSTWorker::~GSTWorker() {}


/**
 * @brief performs the rampup stress test (lazy approach) on the given GPU
 * @param error pointer to a memory location where the error code will be stored
 * @param err_description will store the error description if any 
 * @return true if target_stress is achieved within ramp_interval, false otherwise
 */
bool GSTWorker::do_gst_ramp(int *error, string *err_description) {
    string msg;
    uint64_t total_millis_gen_rand_matrix = 0;
    uint64_t num_sgemm_ops_per_matrix_size = 0, log_interval_milliseconds = 0;
    uint16_t num_sgemm_ops = 0, num_sgemm_ops_sustained_stage = 0;
    bool glops_sustained_started = false;
    double total_gflop = 0, gflops_interval = 0, seconds_elapsed = 0;
    std::chrono::time_point<std::chrono::system_clock> gst_start_time,
                    gst_end_time, gst_log_interval_time,
                    gst_start_gen_matrix,
                    gst_start_gflops_time, gst_end_gflops_time,
                    gst_start_gflops_sustained_time,
                    gst_end_gflops_sustained_time;
    uint64_t n = GST_MIN_MATRIX_N_SIZE,
                m = GST_MIN_MATRIX_M_SIZE,
                k = GST_MIN_MATRIX_K_SIZE;

    *error = 0;
    // record ramp start time and log interval start time
    gst_start_time = std::chrono::system_clock::now();
    gst_log_interval_time = std::chrono::system_clock::now();

    for (;;) {
        // useful guard in case gpu_blas->run_blass_gemm() keeps failing
        gst_end_time = std::chrono::system_clock::now();
        if (time_diff(gst_end_time,  gst_start_time) > ramp_interval) {
            // ramp up finished and the GPU did not reach
            // the target_stress FGlops
            return false;
        }

        if (num_sgemm_ops_per_matrix_size == 0) {
            gpu_blas = std::unique_ptr<rvs_blas>(
                new rvs_blas(gpu_device_index, m, n, k));
            if (!gpu_blas) {
                *error = 1;
                *err_description = GST_MEM_ALLOC_ERROR;
                return false;
            }

            if (!gpu_blas->error()) {
                // record matrix creation start time (within a single log
                // interval the GST might need to work with diff matrix sizes
                // which means that it has to generate new random
                // matrix/matrices which takes time). In order to provide
                // accurate Gflops results the total matrix creation time will
                // be subtracted from the total log interval
                gst_start_gen_matrix = std::chrono::system_clock::now();
                // generate the random matrix and copy it to the GPU
                gpu_blas->generate_random_matrix_data();
                if (!copy_matrix) {
                    // copy matrix only once
                    if (!gpu_blas->copy_data_to_gpu()) {
                        *error = 1;
                        *err_description = GST_BLAS_MEMCPY_ERROR;
                        return false;
                    }
                }

                // compute matrix creation time
                total_millis_gen_rand_matrix +=
                        time_diff(std::chrono::system_clock::now(),
                                    gst_start_gen_matrix);

                // record log interval - start time
                gst_start_gflops_time = std::chrono::system_clock::now();
            } else {
                // blas related error (finish GST session for the current GPU)
                *error = 1;
                *err_description = GST_BLAS_ERROR;
                return false;
            }
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
            continue;

        while (!gpu_blas->is_gemm_op_complete()) {
            gst_end_time = std::chrono::system_clock::now();
            if (time_diff(gst_end_time,  gst_start_time) > ramp_interval) {
                if (glops_sustained_started) {
                    // ramp finished while the GPU was in the
                    // Gflops sustained period ... we return true
                    ramp_actual_time = time_diff(gst_end_time,  gst_start_time);
                    return true;
                } else {
                    // ramp up finished and the GPU did not reach
                    // the target_stress FGlops
                    return false;
                }
            }
        }

        num_sgemm_ops++;
        // compute the gflop for the current log interval
        total_gflop += gpu_blas->gemm_gflop_count();
        // compute the amount of time elapsed since we started the
        // log period
        log_interval_milliseconds = time_diff(gst_end_time,
                        gst_log_interval_time);

        if (log_interval_milliseconds >=
                    log_interval && num_sgemm_ops >= 1 ) {
            // log interval time elapsed => compute the Gflops and log it out
            seconds_elapsed = static_cast<double>
                        (log_interval_milliseconds -
                            total_millis_gen_rand_matrix) / 1000;
            if (seconds_elapsed != 0) {
                gflops_interval = total_gflop / seconds_elapsed;
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

                // reset time & gflops related data
                total_gflop = 0;
                num_sgemm_ops = 0;
                total_millis_gen_rand_matrix = 0;
                gst_log_interval_time = std::chrono::system_clock::now();
            }
        }

        if (!glops_sustained_started)
            num_sgemm_ops_per_matrix_size++;
        else
            num_sgemm_ops_sustained_stage++;

        if (num_sgemm_ops_per_matrix_size ==
                            NUM_MIN_SGEMM_OPS_PER_MATRIX_SIZE) {
            if (!glops_sustained_started) {
                // GPU peformed all the required SGEMM ops
                // compute the Gflops and check if GPU achieved the target
                gst_end_gflops_time = std::chrono::system_clock::now();
                uint64_t millis_sgemm_ops = time_diff(gst_end_gflops_time,
                                                        gst_start_gflops_time);
                if (millis_sgemm_ops != 0) {
                    // compute the GFLOPS
                    seconds_elapsed = static_cast<double>
                                        (millis_sgemm_ops) / 1000;
                    double curr_gflops =
                        static_cast<double>(gpu_blas->gemm_gflop_count() *
                        num_sgemm_ops_per_matrix_size) / seconds_elapsed;

                    if (curr_gflops > max_gflops)
                        max_gflops = curr_gflops;

                    if (curr_gflops >= target_stress) {
                        if (millis_sgemm_ops >
                                NUM_MIN_MS_RAMP_SUSTAIN_TARGET_GFLOPS - 100) {
                            // no need to check if the GPU can sustain the load
                            // for another NUM_MIN_MS_RAMP_SUSTAIN_TARGET_GFLOPS
                            // milliseconds
                            ramp_actual_time = time_diff(gst_end_gflops_time,
                                                         gst_start_time);
                            return true;
                        }

                        // the GPU reached the target stress Gflops but in ran
                        // less than expected.
                        // check if the GPU can sustain this Gflops
                        // for another XYZ millisecond (the idea is to
                        // avoid those cases when the GPU quickly achieves
                        // the target stress - with a given matrix size -
                        // but then it cannot sustain the workload for the
                        // rest of the stress test period
                        glops_sustained_started = true;
                        num_sgemm_ops_sustained_stage = 0;
                        gst_start_gflops_sustained_time =
                                            std::chrono::system_clock::now();
                    } else {
                        num_sgemm_ops_per_matrix_size = 0;
                        n += GST_MATRIX_SIZE_INCREMENT;
                        m += GST_MATRIX_SIZE_INCREMENT;
                        k += GST_MATRIX_SIZE_INCREMENT;
                    }
                }
            } else {
                gst_end_gflops_sustained_time =
                                            std::chrono::system_clock::now();
                uint64_t millis_sustained_sgemm_ops =  time_diff(
                                gst_end_gflops_sustained_time,
                                gst_start_gflops_sustained_time);
                if (millis_sustained_sgemm_ops >
                                    NUM_MIN_MS_RAMP_SUSTAIN_TARGET_GFLOPS) {
                    // sustained period finished => check if thye GPU
                    // was able to sustain the Gflops
                    seconds_elapsed = static_cast<double>
                                        (millis_sustained_sgemm_ops) / 1000;
                    double curr_gflops =
                        static_cast<double>(gpu_blas->gemm_gflop_count() *
                        num_sgemm_ops_sustained_stage) / seconds_elapsed;

                    if (curr_gflops > max_gflops)
                        max_gflops = curr_gflops;
                    if (curr_gflops >= target_stress -
                                                target_stress * tolerance) {
                        ramp_actual_time = time_diff(
                            gst_end_gflops_sustained_time,  gst_start_time);
                        return true;
                    } else {
                        glops_sustained_started = false;
                        num_sgemm_ops_per_matrix_size = 0;
                        n += GST_MATRIX_SIZE_INCREMENT;
                        m += GST_MATRIX_SIZE_INCREMENT;
                        k += GST_MATRIX_SIZE_INCREMENT;
                    }
                }
            }
        }
    }
}

/**
 * @brief performs the rampup stress test (greedy aproach) on the given GPU
 * @param error pointer to a memory location where the error code will be stored
 * @param err_description will store the error description if any 
 * @return true if target_stress is achieved within ramp_interval, false otherwise
 */
bool GSTWorker::do_gst_ramp_greedy(int *error, string *err_description) {
    uint64_t n, m, k;
    uint16_t i;
    std::chrono::time_point<std::chrono::system_clock> gst_start_time,
                        gst_end_time, gst_log_interval_time,
                        gst_start_gflops_time;
    double seconds_elapsed = 0;
    uint16_t num_sgemm_ops = 0, num_sgemm_ops_log_interval = 0;
    string msg;

    *error = 1;
    // determine proper n, m and k for the given <target_stress>
    for (i = 0; i < NUM_MAX_TRY_SGEMM_OPS_DIFF_SIZE; i++) {
        // compute n, m and k for (5...305) SGEMM calls
        // a smaller number of SGEMM calls would lead to smaller
        // kernel launching penalties time
        m = pow(((target_stress / (NUM_INIT_SGEMM_OPS_PER_MATRIX_SIZE + i *
                   NUM_INC_SGEMM_OPS_PER_MATRIX_SIZE)) / 2) * 1e9, 1.0/3.0);

        m += (m * GST_MATRIX_SIZE_PROCENT_DELTA) / 100;

        n = k = m;
        // setup the RVS blas stuff (alllocate memory on host & device,
        // select GPU device etc.)
        gpu_blas = std::unique_ptr<rvs_blas>(
            new rvs_blas(gpu_device_index, m, n, k));
        if (gpu_blas) {
            if (!gpu_blas->error()) {
                *error = 0;
                break;
            }
        }
    }

    if (*error == 1) {
        *err_description = GST_MEM_ALLOC_ERROR;
        return false;
    }

    // generate random matrix & copy it to the GPU
    gpu_blas->generate_random_matrix_data();
    if (!copy_matrix) {
        // copy matrix only once
        if (!gpu_blas->copy_data_to_gpu()) {
            *error = 1;
            *err_description = GST_BLAS_MEMCPY_ERROR;
            return false;
        }
    }

    // record ramp start time and log interval start time
    gst_start_time = std::chrono::system_clock::now();
    gst_log_interval_time = std::chrono::system_clock::now();
    gst_start_gflops_time = std::chrono::system_clock::now();

    for (;;) {
        // useful guard in case gpu_blas->run_blass_gemm() keeps failing
        gst_end_time = std::chrono::system_clock::now();
        if (time_diff(gst_end_time,  gst_start_time) > ramp_interval) {
            // ramp up finished and the GPU did not reach
            // the target_stress FGlops
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
            if (time_diff(gst_end_time,  gst_start_time) > ramp_interval) {
                // ramp up finished and the GPU did not reach
                // the target_stress FGlops
                return false;
            }
        }

        num_sgemm_ops++;
        num_sgemm_ops_log_interval++;

        gst_end_time = std::chrono::system_clock::now();
        uint64_t millis_sgemm_ops =
                        time_diff(gst_end_time, gst_start_gflops_time);

        if (millis_sgemm_ops >= NUM_MS_CHECK_GFLOPS) {
            // compute the GFLOPS
            seconds_elapsed = static_cast<double>
                                (millis_sgemm_ops) / 1000;
            double curr_gflops =
                    static_cast<double>(gpu_blas->gemm_gflop_count() *
                        num_sgemm_ops) / seconds_elapsed;
            if (curr_gflops > max_gflops)
                max_gflops = curr_gflops;
            if (curr_gflops >= target_stress) {
                ramp_actual_time =
                                time_diff(gst_end_time, gst_start_time);
                return true;
            }
            num_sgemm_ops = 0;
            gst_start_gflops_time = std::chrono::system_clock::now();
        }

        millis_sgemm_ops =
                    time_diff(gst_end_time, gst_log_interval_time);

        if (millis_sgemm_ops >= log_interval) {
            // compute the GFLOPS
            seconds_elapsed = static_cast<double>
                                (millis_sgemm_ops) / 1000;
            double curr_gflops =
                    static_cast<double>(gpu_blas->gemm_gflop_count() *
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
                if (gflops_interval <
                        target_stress - target_stress * tolerance) {
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
    bool ramp_up_success;
    if (gflops_greedy_strategy)
        ramp_up_success = do_gst_ramp_greedy(&error, &err_description);
    else
        ramp_up_success = do_gst_ramp(&error, &err_description);

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
