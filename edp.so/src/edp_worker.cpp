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
#include "include/edp_worker.h"

#include <unistd.h>
#include <string>
#include <memory>
#include <iostream>
#include <atomic>

#include "include/rvs_blas.h"
#include "include/rvs_module.h"
#include "include/rvsloglp.h"
#include "include/rvstimer.h"

extern "C" {
  #include <pci/pci.h>
  #include <linux/pci.h>
}

#define MODULE_NAME                             "edp"

#define EDP_MEM_ALLOC_ERROR                     "memory allocation error!"
#define EDP_BLAS_ERROR                          "memory/blas error!"
#define EDP_BLAS_MEMCPY_ERROR                   "HostToDevice mem copy error!"

#define EDP_MAX_GFLOPS_OUTPUT_KEY               "Gflop"
#define EDP_FLOPS_PER_OP_OUTPUT_KEY             "flops_per_op"
#define EDP_BYTES_COPIED_PER_OP_OUTPUT_KEY      "bytes_copied_per_op"
#define EDP_TRY_OPS_PER_SEC_OUTPUT_KEY          "try_ops_per_sec"

#define EDP_LOG_GFLOPS_INTERVAL_KEY             "Gflops"
#define EDP_JSON_LOG_GPU_ID_KEY                 "gpu_id"

#define PROC_DEC_INC_SGEMM_FREQ_DELAY           10

#define NMAX_MS_GPU_RUN_PEAK_PERFORMANCE        1000
#define NMAX_MS_SGEMM_OPS_RAMP_SUB_INTERVAL     1000
#define USLEEP_MAX_VAL                          (1000000 - 1)

#define EDP_COPY_MATRIX_MSG                     "copy matrix"
#define EDP_START_MSG                           "start"
#define EDP_PASS_KEY                            "pass"
#define EDP_RAMP_EXCEEDED_MSG                   "ramp time exceeded"
#define EDP_TARGET_ACHIEVED_MSG                 "target achieved"
#define EDP_STRESS_VIOLATION_MSG                "stress violation"

using std::string;

bool EDPWorker::bjson = false;
static std::atomic<bool> flag(false);

EDPWorker::EDPWorker() {}
EDPWorker::~EDPWorker() {}

/**
 * @brief performs the rvsBlas setup
 * @param error pointer to a memory location where the error code will be stored
 * @param err_description stores the error description if any
 */
void EDPWorker::setup_blas(int *error, string *err_description) {
    *error = 0;
    // setup rvsBlas
    gpu_blas = std::unique_ptr<rvs_blas>(
        new rvs_blas(gpu_device_index, matrix_size_a, matrix_size_b,
                        matrix_size_c, edp_trans_a, edp_trans_b,
                        edp_alpha_val, edp_beta_val, 
                        edp_lda_offset, edp_ldb_offset, edp_ldc_offset));

    if (!gpu_blas) {
        *error = 1;
        *err_description = EDP_MEM_ALLOC_ERROR;
        return;
    }

    if (gpu_blas->error()) {
        *error = 1;
        *err_description = EDP_MEM_ALLOC_ERROR;
        return;
    }

    // generate random matrix & copy it to the GPU
    gpu_blas->generate_random_matrix_data();
    if (!copy_matrix) {
        // copy matrix only once
        if (!gpu_blas->copy_data_to_gpu(edp_ops_type)) {
            *error = 1;
            *err_description = EDP_BLAS_MEMCPY_ERROR;
        }
    }
}

/**
 * @brief logs the Gflops computed over the last log_interval period 
 * @param gflops_interval the Gflops that the GPU achieved
 */
void EDPWorker::check_target_stress(double gflops_interval) {
    string msg;
    bool result;

    if(gflops_interval >= target_stress){
           result = true;
    }else{
           result = false;
    }

    msg = "[" + action_name + "] " + MODULE_NAME + " " +
              std::to_string(gpu_id) + " " + EDP_LOG_GFLOPS_INTERVAL_KEY + " " + std::to_string(gflops_interval) + " " +
              "Target stress :" + " " + std::to_string(target_stress) + " met :" + (result ? "TRUE" : "FALSE");
    rvs::lp::Log(msg, rvs::logresults);

    log_to_json(EDP_LOG_GFLOPS_INTERVAL_KEY, std::to_string(gflops_interval),
                rvs::loginfo);
}



/**
 * @brief logs the Gflops computed over the last log_interval period 
 * @param gflops_interval the Gflops that the GPU achieved
 */
void EDPWorker::log_interval_gflops(double gflops_interval) {
    string msg;
    msg = "[" + action_name + "] " + MODULE_NAME + " " +
            std::to_string(gpu_id) + " " + EDP_LOG_GFLOPS_INTERVAL_KEY + " " +
            std::to_string(gflops_interval);
    rvs::lp::Log(msg, rvs::loginfo);

    log_to_json(EDP_LOG_GFLOPS_INTERVAL_KEY, std::to_string(gflops_interval),
                rvs::loginfo);
}




/**
 * @brief performs the stress test on the given GPU
 * @param error pointer to a memory location where the error code will be stored
 * @param err_description stores the error description if any
 * @return true if stress violations is less than max_violations, false otherwise
 */
bool EDPWorker::do_edp_stress_test(int *error, std::string *err_description) {
    uint16_t num_sgemm_ops = 0;
    uint16_t num_gflops_violations = 0;
    uint64_t total_milliseconds, log_interval_milliseconds;
    uint64_t start_time, end_time;
    double seconds_elapsed, gflops_interval;
    double timetakenforoneiteration;
    string msg;
    std::chrono::time_point<std::chrono::system_clock> edp_start_time,
                                            edp_end_time, edp_log_interval_time;

    *error = 0;
    max_gflops = 0;
    num_sgemm_ops = 0;
    start_time = 0;
    end_time = 0;

    edp_start_time = std::chrono::system_clock::now();
    edp_log_interval_time = std::chrono::system_clock::now();

    // setup rvs blas
    setup_blas(error, err_description);
    if (*error)
        return false;

    for (;;) {

        //Start the timer
        start_time = gpu_blas->get_time_us();

        // run GEMM & wait for completion
        gpu_blas->run_blass_gemm(edp_ops_type);

        //End the timer
        end_time = gpu_blas->get_time_us();

        //Converting microseconds to seconds
        timetakenforoneiteration = (end_time - start_time)/1e6;

        gflops_interval = gpu_blas->gemm_gflop_count()/timetakenforoneiteration/1e9;

        log_interval_gflops(gflops_interval);

        if(edp_hot_calls == 0) { 
           break;
        }else{
          edp_hot_calls--;
        }

    }

    return true;
}


/**
 * @brief performs the stress test on the given GPU
 */
void EDPWorker::run() {
    //pthread_t thread;
    string    err_description;
    string    msg;
    bool      edp_test_passed;
    int       interval;
    int       error;

    edp_test_passed = true;
    interval        = edp_periodic_wave_timer;
    max_gflops      = 0;
    error           = 0;

    //pthread_create(&thread, NULL, enable_disable_waves, &interval);

    // log EDP stress test - start message
    msg = "[" + action_name + "] " + MODULE_NAME + " " +
            std::to_string(gpu_id) + " " + EDP_START_MSG + " " +
            " Starting the EDP stress test "; 
    rvs::lp::Log(msg, rvs::logtrace);

    log_to_json(EDP_START_MSG, std::to_string(target_stress), rvs::loginfo);
    log_to_json(EDP_COPY_MATRIX_MSG, (copy_matrix ? "true":"false"),
                rvs::loginfo);

    if (run_duration_ms > 0) {
            edp_test_passed = do_edp_stress_test(&error, &err_description);
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
 * @brief logs the EDP test result
 * @param edp_test_passed true if test succeeded, false otherwise
 */
void EDPWorker::log_edp_test_result(bool edp_test_passed) {
    string msg;

    double flops_per_op = (2 * (static_cast<double>(gpu_blas->get_m())/1000) *
                                (static_cast<double>(gpu_blas->get_n())/1000) *
                                (static_cast<double>(gpu_blas->get_k())/1000));
    msg = "[" + action_name + "] " + MODULE_NAME + " " +
        std::to_string(gpu_id) + " " + EDP_MAX_GFLOPS_OUTPUT_KEY + ": " +
        std::to_string(max_gflops) + " " + EDP_FLOPS_PER_OP_OUTPUT_KEY + ": " +
        std::to_string(flops_per_op) + "x1e9" + " " +
        EDP_BYTES_COPIED_PER_OP_OUTPUT_KEY + ": " +
        std::to_string(gpu_blas->get_bytes_copied_per_op()) +
        " " + EDP_TRY_OPS_PER_SEC_OUTPUT_KEY + ": "+
        std::to_string(target_stress / gpu_blas->gemm_gflop_count()) +
        " "  ;
    rvs::lp::Log(msg, rvs::logresults);

    log_to_json(EDP_MAX_GFLOPS_OUTPUT_KEY, std::to_string(max_gflops),
                rvs::loginfo);
    log_to_json(EDP_FLOPS_PER_OP_OUTPUT_KEY, std::to_string(flops_per_op) +
                "x1e9", rvs::loginfo);
    log_to_json(EDP_BYTES_COPIED_PER_OP_OUTPUT_KEY,
                std::to_string(gpu_blas->get_bytes_copied_per_op()),
                rvs::loginfo);
    log_to_json(EDP_TRY_OPS_PER_SEC_OUTPUT_KEY,
                std::to_string(target_stress / gpu_blas->gemm_gflop_count()),
                rvs::loginfo);
    log_to_json(EDP_PASS_KEY, (edp_test_passed ?
            EDP_RESULT_PASS_MESSAGE : EDP_RESULT_FAIL_MESSAGE),
            rvs::logresults);
}

/**
 * @brief computes the difference (in milliseconds) between 2 points in time
 * @param t_end second point in time
 * @param t_start first point in time
 * @return time difference in milliseconds
 */
uint64_t EDPWorker::time_diff(
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
void EDPWorker::log_to_json(const std::string &key, const std::string &value,
                     int log_level) {
    if (EDPWorker::bjson) {
        unsigned int sec;
        unsigned int usec;

        rvs::lp::get_ticks(&sec, &usec);
        void *json_node = rvs::lp::LogRecordCreate(MODULE_NAME,
                            action_name.c_str(), log_level, sec, usec);
        if (json_node) {
            rvs::lp::AddString(json_node, EDP_JSON_LOG_GPU_ID_KEY,
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
void EDPWorker::usleep_ex(uint64_t microseconds) {
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
