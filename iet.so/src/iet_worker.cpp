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
#include <chrono>
#include <memory>

#include "hwmon_util.h"
#include "blas_worker.h"
#include "rvs_module.h"
#include "rvsloglp.h"

#define MODULE_NAME                             "iet"
#define POWER_PROCESS_DELAY                     5
#define MAX_MS_TRAIN_GPU                        1000
#define MAX_MS_WAIT_BLAS_THREAD                 (1000 * 100)
#define SGEMM_DELAY_FREQ_DEV                    10

#define IET_RESULT_PASS_MESSAGE                 "TRUE"
#define IET_RESULT_FAIL_MESSAGE                 "FALSE"

#define IET_BLAS_FAILURE                        "BLAS setup failed!"
#define IET_MEM_ALLOC_ERROR                     "memory allocation error!"
#define IET_POWER_PROC_ERROR                    "could not get/process the GPU"\
                                                " power!"
#define IET_SGEMM_FAILURE                       "GPU failed to run the SGEMMs!"

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
 * @brief performs the EDPp rampup on the given GPU (attempts to reach the given
 * target power)
 * @param err_description stores the error description if any
 * @return true if gpu training succeeded, false otherwise
 */
bool IETWorker::do_gpu_init_training(string *err_description) {
    std::chrono::time_point<std::chrono::system_clock>  start_time, end_time;
    float cur_power_value;
    uint64_t power_sampling_iters = 0;

    // init with no error
    *err_description = "";

    // let the GPU run SGEMMs for MAX_MS_TRAIN_GPU ms (e.g.: 1000) and:
    // 1. get the number of SGEMMs the GPU managed to run (needed in order
    // to detect/change the SGEMMs frequency)
    // 2. get the max power
    num_sgemms_training = 0;
    avg_power_training = 0;

     gpu_worker = std::unique_ptr<blas_worker>(
        new blas_worker(gpu_device_index, matrix_size));
    if (gpu_worker == nullptr) {
        *err_description = IET_MEM_ALLOC_ERROR;
        return false;
    }
    if (gpu_worker->get_blas_error()) {
        *err_description = IET_BLAS_FAILURE;
        return false;
    }
    gpu_worker->set_sgemm_delay(0);
    gpu_worker->set_bcount_sgemm(true);

    // start the SGEMM workload
    gpu_worker->start();

    // wait for the BLAS setup to complete
    while (!gpu_worker->is_setup_complete()) {}

    // record inital time
    start_time = std::chrono::system_clock::now();

    for (;;) {
        // get power data
        cur_power_value = get_power_data(gpu_hwmon_entry);
        if (cur_power_value != 0) {
            avg_power_training += cur_power_value;
            power_sampling_iters++;
        }
        usleep(POWER_PROCESS_DELAY);

        end_time = std::chrono::system_clock::now();
        uint64_t diff_ms = time_diff(end_time, start_time);
        if (diff_ms >= MAX_MS_TRAIN_GPU) {
            // wait for the last sgemm to finish
            while (!gpu_worker->is_sgemm_complete()) {}
            // record the actual training time
            end_time = std::chrono::system_clock::now();
            diff_ms = time_diff(end_time, start_time);
            training_time_ms = diff_ms;
            // stop the training
            break;
        }
    }

    // gather the GPUS stats
    num_sgemms_training = gpu_worker->get_num_sgemm_ops();
    if (num_sgemms_training  == 0) {
        *err_description = IET_SGEMM_FAILURE;
        return false;
    }

    if (power_sampling_iters != 0) {
        avg_power_training /= power_sampling_iters;
        if (avg_power_training > 0)
            return true;

        *err_description = IET_POWER_PROC_ERROR;
        return false;
    }

    *err_description = IET_POWER_PROC_ERROR;
    return false;
}

/**
 * @brief computes SGEMMs and power related statistics after the training stage
 */
void IETWorker::compute_gpu_stats(void) {
    float ms_per_sgemm, sgemm_target_power;
    float sgemm_target_power_si, total_ms_sgemm_si;

    // compute SGEMM time (ms)
    ms_per_sgemm = static_cast<float>(training_time_ms) / num_sgemms_training;
    // compute required number of SGEMM for the given target_power
    sgemm_target_power =
                    (target_power * num_sgemms_training) / avg_power_training;
    sgemm_target_power_si =
                    (sample_interval * sgemm_target_power) / training_time_ms;
    // compute the actual SGEMM frequency for the given target_power
    total_ms_sgemm_si = sgemm_target_power_si * ms_per_sgemm;
    sgemm_si_delay = sample_interval - total_ms_sgemm_si;
    if (sgemm_si_delay < 0) {
        sgemm_si_delay = 0;
    } else {
        if (sgemm_target_power_si > 0)
            sgemm_si_delay /= sgemm_target_power_si;
        else
            sgemm_target_power_si = sample_interval;

        sgemm_si_delay = sgemm_si_delay + sgemm_si_delay / SGEMM_DELAY_FREQ_DEV;
    }
}

/**
 * @brief computes the new SGEMM frequency so that the GPU will achieve the
 * given target_power
 * @param avg_power the last GPU average power over the last sample_interval
 */
void IETWorker::compute_new_sgemm_freq(float avg_power) {
    // compute the difference between the actual power data and the target_power
    float diff_power = avg_power - target_power;
    // gradually & dynamically increase/decrease the SGEMM frequency
    float sgemm_delay_dev = (abs(diff_power) * sgemm_si_delay) / target_power;
    if (diff_power < 0) {
        if (sgemm_si_delay - sgemm_delay_dev < 0)
            sgemm_si_delay = 1;
        else
            sgemm_si_delay -= sgemm_delay_dev;
    } else {
        sgemm_si_delay += sgemm_delay_dev;
    }
}

/**
 * @brief performs the EDPp rampup on the given GPU (attempts to reach the given
 * target power)
 * @param error pointer to a memory location where the error code will be stored
 * @param err_description stores the error description if any
 * @return true if target power is achieved within the ramp_interval, 
 * false otherwise
 */
bool IETWorker::do_iet_ramp(int *error, string *err_description) {
    std::chrono::time_point<std::chrono::system_clock> iet_start_time, end_time,
                                                        sampling_start_time;
    float cur_power_value, avg_power = 0;
    uint64_t power_sampling_iters = 0, cur_milis_sampling;
    string msg;

    *error = 0;
    *err_description = "";

    if (!do_gpu_init_training(err_description)) {
        *error = 1;
        return false;
    }

    compute_gpu_stats();

    gpu_worker->pause();
    // let the BLAS worker complete the last SGEMM
    usleep(MAX_MS_WAIT_BLAS_THREAD);
    gpu_worker->set_sgemm_delay(sgemm_si_delay * 1000);

    // record EDPp ramp-up start time
    iet_start_time = std::chrono::system_clock::now();
    sampling_start_time = std::chrono::system_clock::now();

    // restart the worker
    gpu_worker->resume();

    for (;;) {
        // get GPU's current average power
        cur_power_value = get_power_data(gpu_hwmon_entry);
        if (cur_power_value != 0) {
            avg_power += cur_power_value;
            power_sampling_iters++;
        }

        end_time = std::chrono::system_clock::now();
        cur_milis_sampling = time_diff(end_time, sampling_start_time);
        if (cur_milis_sampling >= sample_interval &&
                                    gpu_worker->is_sgemm_complete()) {
            gpu_worker->pause();
            // it's sampling time => check the power value against target_power
            if (power_sampling_iters != 0) {
                avg_power /= power_sampling_iters;
                if (!(avg_power >= target_power - tolerance * target_power &&
                    avg_power <= target_power + tolerance * target_power)) {
                    // compute the new SGEMMs frequency
                    compute_new_sgemm_freq(avg_power);
                    // set the new SGEMM frequency
                    gpu_worker->set_sgemm_delay(sgemm_si_delay * 1000);

                } else {
                    ramp_actual_time = training_time_ms +
                                        time_diff(end_time, iet_start_time);
                    return true;
                }
            }

            avg_power = 0;
            power_sampling_iters = 0;
            sampling_start_time = std::chrono::system_clock::now();
            gpu_worker->resume();
        }

        cur_milis_sampling = time_diff(end_time, iet_start_time);
        if (cur_milis_sampling > ramp_interval - training_time_ms)
            return false;

        usleep(POWER_PROCESS_DELAY);
    }
}


/**
 * @brief performs the EDPp stress test on the given GPU (attempts to sustain
 * the target power)
 * @return true if EDPp test succeeded, false otherwise
 */
bool IETWorker::do_iet_power_stress(void) {
    std::chrono::time_point<std::chrono::system_clock> iet_start_time, end_time,
                                                        sampling_start_time;
    float cur_power_value, avg_power = 0;
    uint64_t power_sampling_iters = 0, cur_milis_sampling, total_time_ms;
    uint16_t num_power_violations = 0;
    string msg;

    // record EDPp ramp-up start time
    iet_start_time = std::chrono::system_clock::now();
    sampling_start_time = std::chrono::system_clock::now();

    // restart the worker
    gpu_worker->resume();

    for (;;) {
        // get GPU's current average power
        cur_power_value = get_power_data(gpu_hwmon_entry);
        if (cur_power_value != 0) {
            avg_power += cur_power_value;
            power_sampling_iters++;
        }

        end_time = std::chrono::system_clock::now();
        cur_milis_sampling = time_diff(end_time, sampling_start_time);
        if (cur_milis_sampling >= sample_interval &&
                                    gpu_worker->is_sgemm_complete()) {
            gpu_worker->pause();
            // it's sampling time => check the power value against target_power
            if (power_sampling_iters != 0) {
                avg_power /= power_sampling_iters;
                if (!(avg_power >= target_power - tolerance * target_power &&
                    avg_power <= target_power + tolerance * target_power)) {
                    // detected a target_power violation
                    num_power_violations++;
                    msg = action_name + " " + MODULE_NAME + " " +
                            std::to_string(gpu_id) + " power violation " +
                            std::to_string(avg_power);
                    log(msg.c_str(), rvs::loginfo);
                }
            }

            avg_power = 0;
            power_sampling_iters = 0;
            sampling_start_time = std::chrono::system_clock::now();
            gpu_worker->resume();
        }

        total_time_ms = time_diff(end_time, iet_start_time);
        if (total_time_ms > run_duration_ms - ramp_actual_time)
            break;

        usleep(POWER_PROCESS_DELAY);
    }

    gpu_worker->stop();
    usleep(MAX_MS_WAIT_BLAS_THREAD);
    // terminate the blas worker thread
    gpu_worker->join();

    if (num_power_violations > max_violations)
        return false;

    return true;
}

/**
 * @brief performs the Input EDPp test on the given GPU
 */
void IETWorker::run() {
    string msg, err_description;
    int error;

    msg = action_name + " " + MODULE_NAME + " " + std::to_string(gpu_id) +
            " start " + std::to_string(target_power);
    log(msg.c_str(), rvs::loginfo);

    if (ramp_interval < MAX_MS_TRAIN_GPU)
        ramp_interval += MAX_MS_TRAIN_GPU;
    if (run_duration_ms < MAX_MS_TRAIN_GPU)
        run_duration_ms += MAX_MS_TRAIN_GPU;

    if (!do_iet_ramp(&error, &err_description)) {
        // terminate the blas worker thread
        gpu_worker->stop();
        usleep(MAX_MS_WAIT_BLAS_THREAD);
        gpu_worker->join();

        if (error) {
            msg = action_name + " " + MODULE_NAME + " "
                    + std::to_string(gpu_id) + " " + err_description;
        } else  {
            msg = action_name + " " + MODULE_NAME + " " +
            std::to_string(gpu_id) + " ramp time exceeded " +
            std::to_string(ramp_interval);
        }

        log(msg.c_str(), rvs::logerror);

        msg = action_name + " " + MODULE_NAME + " " + std::to_string(gpu_id) +
                " pass: " + IET_RESULT_PASS_MESSAGE;
        log(msg.c_str(), rvs::logresults);

    } else {
        // the GPU succeeded in achieving the given target_power
        // => log a message and start the sustained stress test
        msg = action_name + " " + MODULE_NAME + " " + std::to_string(gpu_id) +
                " target achieved " + std::to_string(target_power);
        log(msg.c_str(), rvs::loginfo);

        bool pass = do_iet_power_stress();
        msg = action_name + " " + MODULE_NAME + " " + std::to_string(gpu_id) +
        " pass: " + (pass ? IET_RESULT_PASS_MESSAGE : IET_RESULT_FAIL_MESSAGE);
        log(msg.c_str(), rvs::logresults);
    }
}
