/********************************************************************************
 *
 * Copyright (c) 2018-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <unistd.h>
#include <string>
#include <iostream>
#include <chrono>
#include <memory>
#include <exception>

#include "rocm_smi/rocm_smi.h"
#include "include/rvs_module.h"
#include "include/rvsloglp.h"

#include "include/tst_worker.h"

#define MODULE_NAME                             "tst"
#define MAX_MS_TRAIN_GPU                        1000
#define MAX_MS_WAIT_BLAS_THREAD                 10000
#define SGEMM_DELAY_FREQ_DEV                    10

#define TST_RESULT_PASS_MESSAGE                 "TRUE"
#define TST_RESULT_FAIL_MESSAGE                 "FALSE"

#define TST_BLAS_FAILURE                        "BLAS setup failed!"
#define TST_SGEMM_FAILURE                       "GPU failed to run the SGEMMs!"

#define TST_JSON_TARGET_TEMP_KEY                  "target_temp"
#define TST_JSON_THROTTLE_TEMP_KEY                 "throttle_temp"

#define TST_TARGET_MESSAGE                      "target"
#define TST_DTYPE_MESSAGE                       "dtype"
#define TST_PWR_TARGET_ACHIEVED_MSG             "target achieved"
#define TST_PWR_RAMP_EXCEEDED_MSG               "ramp time exceeded"
#define TST_PASS_KEY                            "pass"

#define TST_JSON_LOG_GPU_ID_KEY                 "gpu_id"
#define TST_JSON_LOG_GPU_IDX_KEY                "gpu_index"
#define TST_MEM_ALLOC_ERROR                     1
#define TST_BLAS_ERROR                          2
#define TST_BLAS_MEMCPY_ERROR                   3
#define TST_BLAS_ITERATIONS                     25
#define TST_LOG_GFLOPS_INTERVAL_KEY             "GFLOPS"
#define TST_AVERAGE_EDGE_TEMP_KEY               "average edge temperature"
#define TST_AVERAGE_JUNCTION_TEMP_KEY               "average junction temperature"

using std::string;

bool TSTWorker::bjson = false;

/**
 * @brief computes the difference (in milliseconds) between 2 points in time
 * @param t_end second point in time
 * @param t_start first point in time
 * @return time difference in milliseconds
 */
static uint64_t time_diff(
                std::chrono::time_point<std::chrono::system_clock> t_end,
                std::chrono::time_point<std::chrono::system_clock> t_start) {
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
                            t_end - t_start);
    return milliseconds.count();
}


/**
 * @brief class default constructor
 */
TSTWorker::TSTWorker():endtest(false) {
}

TSTWorker::~TSTWorker() {
}

/**
 * @brief logs the Gflops computed over the last log_interval period
 * @param gflops_interval the Gflops that the GPU achieved
 */
void TSTWorker::log_interval_gflops(double gflops_interval) {
    string msg;
    msg = " GPU flops :" + std::to_string(gflops_interval);
    rvs::lp::Log(msg, rvs::logtrace);

}

/**
 * @brief Thread function to execute blas gemm operations for GPU workload.
 * @param gpuIdx the gpu that will run the GEMM
 * @param matrix_size matrix size
 * @param tst_ops_type blas operation type
 * @param start
 * @param run_duration_ms test run duration
 * @param transa matrix A transpose operation type
 * @param transb matrix B transpose operation type
 * @param alpha scalar for matrix A*B
 * @param beta scalar for matrix C
 * @param tst_lda_offset leading dimension for matrix A
 * @param tst_ldb_offset leading dimension for matrix B
 * @param tst_ldc_offset leading dimension for matrix C
 * @param tst_ldd_offset leading dimension for matrix D
 */
void TSTWorker::blasThread(int gpuIdx, uint64_t matrix_size, std::string tst_ops_type,
    bool start, uint64_t run_duration_ms, int transa, int transb, float alpha, float beta,
    int tst_lda_offset, int tst_ldb_offset, int tst_ldc_offset, int tst_ldd_offset){

    std::chrono::time_point<std::chrono::system_clock> tst_start_time, tst_end_time;
    double timetakenforoneiteration;
    double gflops_interval;
    double duration;
    uint64_t gem_ops;
    std::unique_ptr<rvs_blas> gpu_blas;
    rvs_blas *free_gpublas;
    string msg;
    std::string blas_source = "rocblas";
    std::string compute_type = "fp32_r";

    duration = 0;
    gem_ops = 0;

    // setup rvsBlas
    gpu_blas = std::unique_ptr<rvs_blas>(new rvs_blas(gpuIdx, matrix_size, matrix_size, matrix_size, "default",
          transa, transb, alpha, beta, tst_lda_offset, tst_ldb_offset, tst_ldc_offset, tst_ldd_offset, tst_ops_type,
          "", "", 0, 0, 0, 0, 0, blas_source, compute_type, "", "", ""));

    //Genreate random matrix data
    gpu_blas->generate_random_matrix_data();

    //Copy data to GPU
    gpu_blas->copy_data_to_gpu();

    tst_start_time = std::chrono::system_clock::now();

    //Hit the GPU with load to increase temperature
    while ( (duration < run_duration_ms) && (endtest == false) ){
        //call the gemm blas
        gpu_blas->run_blas_gemm();

        /* Set callback to be called upon completion of blas gemm operations */
        gpu_blas->set_callback(blas_callback, (void *)this);

        std::unique_lock<std::mutex> lk(mutex);
        cv.wait(lk);

        if(!blas_status) {
          msg = "[" + action_name + "] " + MODULE_NAME + " " +
            std::to_string(gpu_id) + " " + " BLAS gemm operations failed !!! ";
          rvs::lp::Log(msg, rvs::logtrace);
        }

        //get the end time
        tst_end_time = std::chrono::system_clock::now();
        //Duration in the call
        duration = time_diff(tst_end_time, tst_start_time);
        gem_ops++;

        //Converting microseconds to seconds
        timetakenforoneiteration = duration/1e6;
        //calculating Gemm count
        gflops_interval = gpu_blas->gemm_gflop_count()/timetakenforoneiteration;
        //Print the gflops interval
        log_interval_gflops(gflops_interval);
        // check end test to avoid unnecessary sleep
        if (endtest)
            break;
        //if gemm ops greater than 10000, lets yield
        //if this is not happening we are ending up in
        //out of memmory state
        if(gem_ops > 10000) {
            sleep(1);
            gem_ops = 0;
        }
    }
}

/**
 * @brief performs the Thermal Stress Test (TST) on 
 * the given GPU (attempts to sustain the target thermal).
 * @return true if test succeeded, false otherwise
 */
bool TSTWorker::do_thermal_stress(void) {

    std::chrono::time_point<std::chrono::system_clock> tst_start_time, 
        end_time,
        sampling_start_time;
    uint64_t  total_time_ms;
    int64_t   temperature = 0;
    float     cur_edge_temperature = 0;
    float     max_edge_temperature = 0;
    float     cur_junction_temperature = 0;
    float     max_junction_temperature = 0;
    string    msg;
    bool      result = true;
    bool      start = true;
    rvs::action_result_t action_result;
    rsmi_status_t rsmi_stat;
    auto desc = action_descriptor{action_name, MODULE_NAME, gpu_id}; 
    // Initiate blas workload thread
    std::thread t(&TSTWorker::blasThread, this, gpu_device_index, matrix_size_a, tst_ops_type, start, run_duration_ms,
            tst_trans_a, tst_trans_b, tst_alpha_val, tst_beta_val, tst_lda_offset, tst_ldb_offset, tst_ldc_offset, tst_ldd_offset);

    // Record ramp-up start time
    tst_start_time = std::chrono::system_clock::now();

    for (;;) {

        // Check if stop signal was received
        if (rvs::lp::Stopping())
            break;

        // Get GPU's current edge temperature
        rsmi_stat = rsmi_dev_temp_metric_get(smi_device_index, RSMI_TEMP_TYPE_EDGE,
                RSMI_TEMP_CURRENT, &temperature);
        if (rsmi_stat == RSMI_STATUS_SUCCESS) {
            cur_edge_temperature = static_cast<float>(temperature)/1e3;
        }
        
        temperature = 0;
   
        // Get GPU's current junction temperature
        rsmi_stat = rsmi_dev_temp_metric_get(smi_device_index, RSMI_TEMP_TYPE_JUNCTION,
                RSMI_TEMP_CURRENT, &temperature);
        if (rsmi_stat == RSMI_STATUS_SUCCESS) {
            cur_junction_temperature = static_cast<float>(temperature)/1e3;
        }

        msg = "[" + action_name + "] " + MODULE_NAME + " " + "GPU " +
            std::to_string(gpu_id) + " " + "Current edge temperature is : " + " " + std::to_string(cur_edge_temperature);
        rvs::lp::Log(msg, rvs::loginfo);

        msg = "[" + action_name + "] " + MODULE_NAME + " " + "GPU " +
            std::to_string(gpu_id) + " " + "Current junction temperature is : " + " " + std::to_string(cur_junction_temperature);
        rvs::lp::Log(msg, rvs::loginfo);

        // Update edge temperature to max if it is valid
        if(cur_edge_temperature > 0) {
            // Max. of edge temperature
            max_edge_temperature = std::max(max_edge_temperature, cur_edge_temperature);
        }

        // Update junction temperature to max if it is valid
        if(cur_junction_temperature > 0) {
            // Max. of junction temperature
            max_junction_temperature = std::max(max_junction_temperature, cur_junction_temperature);
        }

        end_time = std::chrono::system_clock::now();

        total_time_ms = time_diff(end_time, tst_start_time);

        msg = "[" + action_name + "] " + MODULE_NAME + " " +
            std::to_string(gpu_id) + " " + " Total time in ms " + " " + std::to_string(total_time_ms) +
            " Run duration in ms " + " " + std::to_string(run_duration_ms);
        rvs::lp::Log(msg, rvs::logtrace);

        if (total_time_ms > run_duration_ms) {
            break;
        }

        // It doesnt make sense to read temperature continously so slowing down
        sleep(1000);

        // Check if stop signal was received
        if (rvs::lp::Stopping()) {
            result = true;
            goto end;
        }
    }


    msg = "[" + action_name + "] " + MODULE_NAME + " " + "GPU " +
        std::to_string(gpu_id) + " " + "max. edge temperature :" + " " + std::to_string(max_edge_temperature);
    rvs::lp::Log(msg, rvs::loginfo);

    msg = "[" + action_name + "] " + MODULE_NAME + " " + "GPU " +
        std::to_string(gpu_id) + " " + "max. junction temperature :" + " " + std::to_string(max_junction_temperature);
    rvs::lp::Log(msg, rvs::loginfo);


    //check whether we reached the target temperature
    if(max_junction_temperature >= target_temp) {
        msg = "[" + action_name + "] " + MODULE_NAME + " " + "GPU " +
            std::to_string(gpu_id) + " " + " Target temperature met :" + " " + std::to_string(max_junction_temperature);
        rvs::lp::Log(msg, rvs::loginfo);
        result = true;
    }
    else {
        msg = "[" + action_name + "] " + MODULE_NAME + " " + "GPU " +
            std::to_string(gpu_id) + " " + " Target temperature could not be met :" + " " + std::to_string(max_junction_temperature);
        rvs::lp::Log(msg, rvs::loginfo);
        result = false;
    }
    //check whether we reached the trottle temperature
    if(max_junction_temperature >= throttle_temp) {
        msg = "[" + action_name + "] " + MODULE_NAME + " " + "GPU " +
            std::to_string(gpu_id) + " " + " Thermal throttling condition met :" + " " + std::to_string(max_junction_temperature);
        rvs::lp::Log(msg, rvs::loginfo);
    }
    else {
        msg = "[" + action_name + "] " + MODULE_NAME + " " + "GPU " +
            std::to_string(gpu_id) + " " + " Thermal throttling condition could not be met :" + " " + std::to_string(max_junction_temperature);
        rvs::lp::Log(msg, rvs::loginfo);
    }
    if (bjson)
        log_to_json(desc, rvs::logresults,
           TST_JSON_THROTTLE_TEMP_KEY, std::to_string(throttle_temp), 
           TST_JSON_TARGET_TEMP_KEY, std::to_string(target_temp),
           TST_DTYPE_MESSAGE, tst_ops_type,
           TST_AVERAGE_EDGE_TEMP_KEY, std::to_string(max_edge_temperature),
           TST_AVERAGE_JUNCTION_TEMP_KEY, std::to_string(max_junction_temperature),
           "pass", result ? "true" : "false");
    action_result.state = rvs::actionstate::ACTION_RUNNING;
    action_result.status = (true == result) ? rvs::actionstatus::ACTION_SUCCESS : rvs::actionstatus::ACTION_FAILED;
    action_result.output = msg.c_str();
    action.action_callback(&action_result);

    msg = "[" + action_name + "] " + MODULE_NAME + " " +
        std::to_string(gpu_id) + " " + " End of worker thread " ;
    rvs::lp::Log(msg, rvs::loginfo);

end:

    endtest = true;

    if (true == t.joinable()) {

        try {
            t.join();
        }
        catch (std::exception& e) {
            std::cout << "Standard exception: " << e.what() << std::endl;
        }
    }
    return result;
}

/**
 * @brief performs the TST test on the given GPU
 */
void TSTWorker::run() {
    string msg, err_description;

    msg = "[" + action_name + "] " + MODULE_NAME + " " +
            std::to_string(gpu_id) + " start " + std::to_string(throttle_temp);

    rvs::lp::Log(msg, rvs::loginfo);

    if (run_duration_ms < MAX_MS_TRAIN_GPU)
        run_duration_ms += MAX_MS_TRAIN_GPU;

    bool pass = do_thermal_stress();

    // check if stop signal was received
    if (rvs::lp::Stopping())
         return;

    msg = "[" + action_name + "] "  +
               "[GPU:: " + std::to_string(gpu_id) + "] " + TST_PASS_KEY + ": " +
               (pass ? TST_RESULT_PASS_MESSAGE : TST_RESULT_FAIL_MESSAGE);
    rvs::lp::Log(msg, rvs::logresults);

    sleep(5);
}

/**
 * @brief blas callback function upon gemm operation completion
 * @param status gemm operation status
 * @param user_data user data set
 */
void TSTWorker::blas_callback (bool status, void *user_data) {

  if(!user_data) {
    return;
  }
  TSTWorker* worker = (TSTWorker*)user_data;

  /* Notify gst worker thread gemm operation completion */
  std::lock_guard<std::mutex> lk(worker->mutex);
  worker->blas_status = status;
  worker->cv.notify_one();
}

