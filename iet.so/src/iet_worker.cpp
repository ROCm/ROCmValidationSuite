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

#include <unistd.h>
#include <string>
#include <iostream>
#include <chrono>
#include <memory>
#include <mutex>

#include "rocm_smi/rocm_smi.h"
#include "include/rvs_module.h"
#include "include/rvsloglp.h"

#include "include/iet_worker.h"

#define MODULE_NAME                             "iet"
#define POWER_PROCESS_DELAY                     5
#define MAX_MS_TRAIN_GPU                        1000
#define MAX_MS_WAIT_BLAS_THREAD                 10000
#define SGEMM_DELAY_FREQ_DEV                    10

#define IET_RESULT_PASS_MESSAGE                 "TRUE"
#define IET_RESULT_FAIL_MESSAGE                 "FALSE"

#define IET_BLAS_FAILURE                        "BLAS setup failed!"
#define IET_POWER_PROC_ERROR                    "could not get/process the GPU"\
                                                " power!"
#define IET_SGEMM_FAILURE                       "GPU failed to run the SGEMMs!"

#define IET_PWR_VIOLATION_MSG                   "power violation"
#define IET_PWR_TARGET_ACHIEVED_MSG             "target achieved"
#define IET_PWR_RAMP_EXCEEDED_MSG               "ramp time exceeded"
#define IET_PASS_KEY                            "pass"

#define IET_JSON_LOG_GPU_ID_KEY                 "gpu_id"
#define IET_MEM_ALLOC_ERROR                     1
#define IET_BLAS_ERROR                          2
#define IET_BLAS_MEMCPY_ERROR                   3
#define IET_BLAS_ITERATIONS                     25

using std::string;

bool IETWorker::bjson = false;


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
IETWorker::IETWorker() {
}

IETWorker::~IETWorker() {
}


/**
 * @brief logs a message to JSON
 * @param key info type
 * @param value message to log
 * @param log_level the level of log (e.g.: info, results, error)
 */
void IETWorker::log_to_json(const std::string &key, const std::string &value,
                     int log_level) {
    if (IETWorker::bjson) {
        unsigned int sec;
        unsigned int usec;

        rvs::lp::get_ticks(&sec, &usec);
        void *json_node = rvs::lp::LogRecordCreate(MODULE_NAME,
                            action_name.c_str(), log_level, sec, usec);
        if (json_node) {
            rvs::lp::AddString(json_node, IET_JSON_LOG_GPU_ID_KEY,
                            std::to_string(gpu_id));
            rvs::lp::AddString(json_node, key, value);
            rvs::lp::LogRecordFlush(json_node);
        }
    }
}


void blasThread(int gpuIdx,  uint64_t matrix_size, std::string  iet_ops_type, 
    bool start, uint64_t run_duration_ms, int transa, int transb, float alpha, float beta,
    int iet_lda_offset, int iet_ldb_offset, int iet_ldc_offset)
{
    std::chrono::time_point<std::chrono::system_clock> iet_start_time, end_time;
    std::unique_ptr<rvs_blas> gpu_blas;
    rvs_blas  *free_gpublas;
    uint64_t  duration;

    duration = 0;
   // setup rvsBlas
    gpu_blas = std::unique_ptr<rvs_blas>(new rvs_blas(gpuIdx,  matrix_size,  matrix_size,  matrix_size, transa, transb, alpha, beta, 
          iet_lda_offset, iet_ldb_offset, iet_ldc_offset));

    iet_start_time = std::chrono::system_clock::now();
    //Hit the GPU with load to increase temperature
    while(duration < run_duration_ms){
         gpu_blas->run_blass_gemm(iet_ops_type);
         end_time = std::chrono::system_clock::now();
         duration = time_diff(end_time, iet_start_time);
    }

    free_gpublas = gpu_blas.release();
    delete free_gpublas;
}


/**
 * @brief performs the EDPp stress test on the given GPU (attempts to sustain
 * the target power)
 * @return true if EDPp test succeeded, false otherwise
 */
bool IETWorker::do_iet_power_stress(void) {
    std::chrono::time_point<std::chrono::system_clock> iet_start_time, end_time,
                                                        sampling_start_time;
    uint64_t  power_sampling_iters = 0;
    uint64_t  total_time_ms;
    uint64_t  last_avg_power;
    string    msg;
    float     cur_power_value;
    float     totalpower;
    float     max_power;
    bool      result;
    bool      start;
   
    max_power = 0;
    totalpower = 0;
    result = true;
    start = true;

    std::thread t(blasThread, gpu_device_index, matrix_size_a, iet_ops_type, start, run_duration_ms, 
		    iet_trans_a, iet_trans_b, iet_alpha_val, iet_beta_val, iet_lda_offset, iet_ldb_offset, iet_ldc_offset);
    t.detach();
 
    // record EDPp ramp-up start time
    iet_start_time = std::chrono::system_clock::now();

    for (;;) {
        // check if stop signal was received
        if (rvs::lp::Stopping())
            break;

       // get GPU's current average power
       rsmi_status_t rmsi_stat = rsmi_dev_power_ave_get(gpu_device_index, 0,
                                    &last_avg_power);

       if (rmsi_stat == RSMI_STATUS_SUCCESS) {
            cur_power_value = static_cast<float>(last_avg_power)/1e6;
       }

        msg = "[" + action_name + "] " + MODULE_NAME + " " +
                     std::to_string(gpu_id) + " " + " Target power is : " + " " + std::to_string(target_power);
        rvs::lp::Log(msg, rvs::logtrace);

        //check whether we reached the target power
        if(cur_power_value > target_power){
            max_power = cur_power_value;
        }

        end_time = std::chrono::system_clock::now();

        total_time_ms = time_diff(end_time, iet_start_time);

        msg = "[" + action_name + "] " + MODULE_NAME + " " +
                     std::to_string(gpu_id) + " " + " Average power" + " " + std::to_string(cur_power_value);
        rvs::lp::Log(msg, rvs::loginfo);

        msg = "[" + action_name + "] " + MODULE_NAME + " " +
                     std::to_string(gpu_id) + " " + " Total time in ms " + " " + std::to_string(total_time_ms) +
                     " Run duration in ms " + " " + std::to_string(run_duration_ms);
        rvs::lp::Log(msg, rvs::logtrace);

        if (total_time_ms > run_duration_ms) {
            break;
	}

       sleep(1000);

       // check if stop signal was received
       if (rvs::lp::Stopping())
         return true;
       }

       if(max_power >= target_power) {
             msg = "[" + action_name + "] " + MODULE_NAME + " " +
                     std::to_string(gpu_id) + " " + " Average power met the target power :" + " " + std::to_string(max_power);
            rvs::lp::Log(msg, rvs::loginfo);
            result = true;
       }else{
            msg = "[" + action_name + "] " + MODULE_NAME + " " +
                     std::to_string(gpu_id) + " " + " Average power couldnt meet the target power  \
                     in the given interval, increase the duration and try again, \
                     Average power is :" + " " + std::to_string(cur_power_value);
            rvs::lp::Log(msg, rvs::loginfo);
            result = false;
       }

       msg = "[" + action_name + "] " + MODULE_NAME + " " +
                   std::to_string(gpu_id) + " " + " End of worker thread " ;
       rvs::lp::Log(msg, rvs::loginfo);

       return result;
}


/**
 * @brief performs the Input EDPp test on the given GPU
 */
void IETWorker::run() {
    string msg, err_description;
    int error;

    msg = "[" + action_name + "] " + MODULE_NAME + " " +
            std::to_string(gpu_id) + " start " + std::to_string(target_power);

    rvs::lp::Log(msg, rvs::loginfo);
    log_to_json("start", std::to_string(target_power), rvs::loginfo);

    if (run_duration_ms < MAX_MS_TRAIN_GPU)
        run_duration_ms += MAX_MS_TRAIN_GPU;

    bool pass = do_iet_power_stress();

    // check if stop signal was received
    if (rvs::lp::Stopping())
         return;

    msg = "[" + action_name + "] " + MODULE_NAME + " " +
               std::to_string(gpu_id) + " " + IET_PASS_KEY + ": " +
               (pass ? IET_RESULT_PASS_MESSAGE : IET_RESULT_FAIL_MESSAGE);
    rvs::lp::Log(msg, rvs::logresults);
}
