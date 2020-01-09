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
#include "include/gst_worker.h"

#include <unistd.h>
#include <string>
#include <memory>
#include <iostream>
#include <limits>
#include <vector>

#include "include/rvs_blas.h"
#include "include/rvs_module.h"
#include "include/rvsloglp.h"

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

#define PROC_DEC_INC_SGEMM_FREQ_DELAY           10

#define NMAX_MS_GPU_RUN_PEAK_PERFORMANCE        2
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
        new rvs_blas(gpu_device_index, matrix_size, matrix_size, matrix_size));

    if ((!gpu_blas) || (gpu_blas->error()) )  {
        *error = 1;
        *err_description = GST_MEM_ALLOC_ERROR;
        return;
    }

    if(gst_ops_type == "sgemm") {
        // generate random matrix & copy it to the GPU
        gpu_blas->generate_random_matrix_data();

        //Copying host memory to device memory
        if (!gpu_blas->copy_data_to_gpu()) {
           *error = 1;
           *err_description = GST_BLAS_MEMCPY_ERROR;
           return ;
        }
    }

    if(gst_ops_type == "dgemm") {
        // generate random matrix & copy it to the GPU
        gpu_blas->generate_random_dbl_matrix_data();

        //Copying host memory to device memory
        if (!gpu_blas->copy_dbl_data_to_gpu()) {
           *error = 1;
           *err_description = GST_BLAS_MEMCPY_ERROR;
           return ;
        }

    }

    if(gst_ops_type == "hgemm") {
        // generate random matrix & copy it to the GPU
        gpu_blas->generate_random_half_matrix_data();

        //Copying host memory to device memory
        if (!gpu_blas->copy_hlf_data_to_gpu()) {
           *error = 1;
           *err_description = GST_BLAS_MEMCPY_ERROR;
           return ;
        }
    }

}

/**
 * @brief logs the Gflops computed over the last log_interval period 
 * @param gflops_interval the Gflops that the GPU achieved
 */
void GSTWorker::log_interval_gflops(double gflops_interval) {
    string msg;
    msg = "[" + action_name + "] " + MODULE_NAME + " " +
            std::to_string(gpu_id) + " " + GST_LOG_GFLOPS_INTERVAL_KEY + " " +
            std::to_string(gflops_interval);
    rvs::lp::Log(msg, rvs::loginfo);

    log_to_json(GST_LOG_GFLOPS_INTERVAL_KEY, std::to_string(gflops_interval),
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
        rvs::lp::Log(msg, rvs::loginfo);

        log_to_json(GST_STRESS_VIOLATION_MSG, std::to_string(gflops_interval),
                    rvs::loginfo);
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

    double min_seconds = std::numeric_limits<double>::max();
    double max_seconds = std::numeric_limits<double>::min();
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time, end_time;
    std::chrono::duration<double> timetaken;
    double sum_seconds = 0.0;
    double seconds = 0.0;
    std::vector<double> times(target_stress);

    std::cout <<"\n GST Stress Test Started ";

    // make sure that the ramp_interval & duration are not less than
    // NMAX_MS_GPU_RUN_PEAK_PERFORMANCE (e.g.: 1000)
    if (run_duration_ms < NMAX_MS_GPU_RUN_PEAK_PERFORMANCE)
        run_duration_ms += NMAX_MS_GPU_RUN_PEAK_PERFORMANCE;
    if (ramp_level < NMAX_MS_GPU_RUN_PEAK_PERFORMANCE)
        ramp_level = NMAX_MS_GPU_RUN_PEAK_PERFORMANCE;

    // stage 1. setup rvs blas
    setup_blas(error, err_description);

    if (*error)
        return false;

    for (int i =0; i < target_stress; i++) {

        hipDeviceSynchronize();
        start_time = std::chrono::high_resolution_clock::now(); 

        for(unsigned int i = 0; i < ramp_level; i++) {
             if(gst_ops_type == "sgemm") {
                   // run GEMM & wait for completion
                   gpu_blas->run_blass_gemm();
             }

             if(gst_ops_type == "dgemm") {
                  // run GEMM & wait for completion
                  gpu_blas->run_blass_dgemm();
             }

             if(gst_ops_type == "hgemm") {
                  // run GEMM & wait for completion
                  gpu_blas->run_blass_hgemm();
             }

        }

         end_time = std::chrono::high_resolution_clock::now();
        hipDeviceSynchronize();

        timetaken = (end_time - start_time);
        seconds = timetaken.count();

        min_seconds = min_seconds < seconds ? min_seconds : seconds;
        max_seconds = max_seconds > seconds ? max_seconds : seconds;
        sum_seconds = sum_seconds + seconds;

        times[i] = seconds;

    }

    double ave_seconds = sum_seconds / (double) target_stress;
    double ops = ((double) target_stress) * 2.0 * ((double)ramp_level);
    double max_gflops = ops / min_seconds / 1e9;
    double min_gflops = ops / max_seconds / 1e9;
    double ave_gflops = ops / ave_seconds / 1e9;

    //calculate relative standard deviation (rsd). Also called coefficient of variation
    double rsd_seconds = 0.0;

    for(int i = 0; i < target_stress; i++) {
        rsd_seconds += (times[i] - ave_seconds) * (times[i] - ave_seconds) ;
    }
    rsd_seconds = rsd_seconds / (double) target_stress;
    rsd_seconds = sqrt(rsd_seconds) / ave_seconds * 100.0;

    std::cout << "\n Time taken for Gflops in seconds :: Max : "<< max_seconds << " Min :" << min_seconds << " Average : " << ave_seconds << " Standard Deviation : " << rsd_seconds;

    std::cout << "\n Average Gflops " << ave_gflops;
    std::cout << "\n Min Gflops " << min_gflops;
    std::cout << "\n Max Gflops " << max_gflops << "\n";

    return true;
}

/**
 * @brief performs the stress test on the given GPU
 */
void GSTWorker::run() {
    int error = 0;
    string err_description;

    //running gst stress test
    do_gst_stress_test(&error, &err_description);

    // check if stop signal was received
    if (rvs::lp::Stopping())
         return;

     if (error) {
       // GPU didn't complete the test (HIP/rocBlas error(s) occurred)
       std::cout << "\n GST Error while running the stress test " << error;
     }
}

/**
 * @brief logs the GST test result
 * @param gst_test_passed true if test succeeded, false otherwise
 */
void GSTWorker::log_gst_test_result(bool gst_test_passed) {
    string msg;


    double flops_per_op = 2;
    msg = "[" + action_name + "] " + MODULE_NAME + " " +
        std::to_string(gpu_id) + " " + GST_MAX_GFLOPS_OUTPUT_KEY + ": " +
        std::to_string(max_gflops) + " " + GST_FLOPS_PER_OP_OUTPUT_KEY + ": " +
        std::to_string(flops_per_op) + "x1e9" + " " +
        GST_BYTES_COPIED_PER_OP_OUTPUT_KEY + ": " +
        std::to_string(gpu_blas->get_bytes_copied_per_op()) +
        " " + GST_TRY_OPS_PER_SEC_OUTPUT_KEY + ": "+
        std::to_string(target_stress / gpu_blas->gemm_gflop_count()) +
        " "  + GST_PASS_KEY + ": " +
        (gst_test_passed ? GST_RESULT_PASS_MESSAGE : GST_RESULT_FAIL_MESSAGE);
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
        unsigned int sec;
        unsigned int usec;

        rvs::lp::get_ticks(&sec, &usec);
        void *json_node = rvs::lp::LogRecordCreate(MODULE_NAME,
                            action_name.c_str(), log_level, sec, usec);
        if (json_node) {
            rvs::lp::AddString(json_node, GST_JSON_LOG_GPU_ID_KEY,
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
