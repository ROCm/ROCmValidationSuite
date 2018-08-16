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
#include "iet_worker.h"

#include <unistd.h>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>

#include "hwmon_util.h"
#include "blas_worker.h"
#include "rvs_module.h"
#include "rvsloglp.h"

#define MODULE_NAME                             "iet"
#define POWER_PROCESS_DELAY                     1
#define MAX_MS_TRAIN_GPU                        1000
#define MAX_MS_WAIT_BLAS_THREAD                 (1000 * 100)

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

IETWorker::IETWorker() {}
IETWorker::~IETWorker() {}

/**
 * @brief performs the EDPp rampup on the given GPU (attempts to reach the given target power)
 * @param error pointer to a memory location where the error code will be stored
 * @param err_description stores the error description if any
 * @return true if gpu training succeeded, false otherwise
 */
bool IETWorker::do_gpu_init_training(int *error, string *err_description) {
    std::chrono::time_point<std::chrono::system_clock>  start_time, end_time;

    // init with no error
    *error = 0;
    *err_description = "";

    // let the GPU run SGEMMs for MAX_MS_TRAIN_GPU ms (e.g.: 1000) and:
    // 1. get the number of SGEMMs the GPU managed to run (needed in order
    // to detect/change the SGEMMs frequency)
    // 2. get the max power

    blas_worker gpu_worker(gpu_device_index, matrix_size);
    *error = gpu_worker.get_blas_error();
    if (*error) {
        // TODO(Tudor) - log the error
        return false;
    }
    gpu_worker.set_sgemm_delay(0);
    gpu_worker.set_bcount_sgemm(true);

    // start the SGEMM workload
    gpu_worker.start();

    // wait for the BLAS setup to complete
    while (!gpu_worker.is_setup_complete()) {}

    // record inital time
    start_time = std::chrono::system_clock::now();

    for (;;) {
        end_time = std::chrono::system_clock::now();
        if (time_diff(end_time, start_time) >= MAX_MS_TRAIN_GPU) {
            // stop the blas worker thread;
            gpu_worker.stop();
            break;
        }
    }

    // join gpu_worker's thread
    gpu_worker.join();

    // wait some more ms to allow the thread to stop
    usleep(MAX_MS_WAIT_BLAS_THREAD);

    // gather the GPUS stats
    std::cout << "blas worker thread has STOPPED" << std::endl;
    std::cout << "num sgemm:" << gpu_worker.get_num_sgemm_ops() << std::endl;
    return true;
}

/**
 * @brief performs the EDPp rampup on the given GPU (attempts to reach the given target power)
 * @param error pointer to a memory location where the error code will be stored
 * @param err_description stores the error description if any
 * @return true if target power is achieved within the ramp_interval, false otherwise
 */
bool IETWorker::do_iet_ramp(int *error, string *err_description) {
    std::chrono::time_point<std::chrono::system_clock>  iet_start_time,
                                                        end_time,
                                                        sampling_start_time;
    float cur_power_value, avg_power = 0;
    uint64_t power_sampling_iters = 0, cur_milis_sampling;
    string msg;

    *error = 0;
    *err_description = "";

    do_gpu_init_training(error, err_description);

    // record EDPp ramp-up start time
    iet_start_time = std::chrono::system_clock::now();
    sampling_start_time = std::chrono::system_clock::now();

    for (;;) {
        // get GPU's current avverage power
        cur_power_value = get_power_data(gpu_hwmon_entry);
        if (cur_power_value != 0) {
            avg_power += cur_power_value;
            power_sampling_iters++;
        }
        usleep(POWER_PROCESS_DELAY);

        end_time = std::chrono::system_clock::now();
        cur_milis_sampling = time_diff(end_time, sampling_start_time);
        if (cur_milis_sampling >= sample_interval) {
            // it's sampling time => check the power value against target_power
            // testing purpose ...
            if (power_sampling_iters != 0) {
                avg_power /= power_sampling_iters;
                if (avg_power >= target_power - tolerance * target_power &&
                    avg_power <= target_power + tolerance * target_power) {
                        std::cout << "reached the desired power_target: " <<
                        avg_power << std::endl;
                        return true;
                }
                msg = "power = " + std::to_string(avg_power);
                log(msg.c_str(), rvs::loginfo);
                avg_power = 0;
                power_sampling_iters = 0;
                sampling_start_time = std::chrono::system_clock::now();
            }
        }
        cur_milis_sampling = time_diff(end_time, iet_start_time);
        if (cur_milis_sampling > ramp_interval)
            return false;
    }

    return false;
}

/**
 * @brief performs the Input EDPp test on the given GPU
 */
void IETWorker::run() {
    string msg, err_description;
    int error;

    // log GST stress test - start
    msg = " IETWorker [" + std::to_string(gpu_id) + "] with power data [" +
    gpu_hwmon_entry + "] is running ... ";
    log(msg.c_str(), rvs::loginfo);

    do_iet_ramp(&error, &err_description);
}
