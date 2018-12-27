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
#include "include/log_worker.h"

#include <unistd.h>
#include <string>
#include <iostream>
#include <mutex>
#include <chrono>

#include "rocm_smi/rocm_smi.h"

#include "include/rvs_module.h"
#include "include/rvsloglp.h"

#define MODULE_NAME                             "iet"
#define POWER_PROCESS_DELAY                     5

#define IET_LOGGER_JSON_LOG_GPU_ID_KEY          "gpu_id"
#define IET_LOGGER_CURRENT_POWER_MSG            "current power"

using std::string;

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
 * @brief default class constructor
 * @param _bjson true if JSON logging is needed, false otherwise
 */
log_worker::log_worker(bool _bjson):
                        bjson(_bjson) {
    bpaused = false;
}

log_worker::~log_worker() {}

/**
 * @brief stop the thread
 */
void log_worker::stop(void) {
    {
        std::lock_guard<std::mutex> lck(mtx_brun);
        brun = false;
    }

    // wait a bit to make sure thread has exited
    std::this_thread::yield();

    try {
      if (t.joinable())
        t.join();
    } catch(...) {
    }
}

/**
 * @brief pauses the worker
 */
void log_worker::pause(void) {
    std::lock_guard<std::mutex> lck(mtx_bpaused);
    bpaused = true;
}

/**
 * @brief resumes the worker
 */
void log_worker::resume(void) {
    std::lock_guard<std::mutex> lck(mtx_bpaused);
    bpaused = false;
}

/**
 * @brief logs a message to JSON
 * @param key info type
 * @param value message to log
 * @param log_level the level of log (e.g.: info, results, error)
 */
void log_worker::log_to_json(const std::string &key, const std::string &value,
                     int log_level) {
    if (bjson) {
        unsigned int sec;
        unsigned int usec;

        rvs::lp::get_ticks(&sec, &usec);
        void *json_node = rvs::lp::LogRecordCreate(MODULE_NAME,
                            action_name.c_str(), log_level, sec, usec);
        if (json_node) {
            rvs::lp::AddString(json_node, IET_LOGGER_JSON_LOG_GPU_ID_KEY,
                            std::to_string(gpu_id));
            rvs::lp::AddString(json_node, key, value);
            rvs::lp::LogRecordFlush(json_node);
        }
    }
}

/**
 * @brief computes the GPU power for each log_interval and logs the data
 */
void log_worker::run() {
    std::chrono::time_point<std::chrono::system_clock> start_time, end_time;
    float cur_power_value, avg_power = 0;
    uint64_t power_sampling_iters = 0, cur_milis, last_avg_power;
    string msg;

    {
        std::lock_guard<std::mutex> lck(mtx_brun);
        brun = true;
    }

    start_time = std::chrono::system_clock::now();
    for (;;) {
        // check if stop signal was received
        if (rvs::lp::Stopping())
            break;

        {
            std::lock_guard<std::mutex> lck(mtx_brun);
            if (!brun)
                break;
        }

        {
            std::lock_guard<std::mutex> lck(mtx_bpaused);
            if (bpaused)
                continue;
        }

        // get GPU's current average power

        rsmi_status_t rmsi_stat = rsmi_dev_power_ave_get(pwr_device_id, 0,
                                    &last_avg_power);
        if (rmsi_stat == RSMI_STATUS_SUCCESS) {
            cur_power_value = static_cast<float>(last_avg_power)/1e6;
            avg_power += cur_power_value;
            power_sampling_iters++;
        }

        end_time = std::chrono::system_clock::now();
        cur_milis = time_diff(end_time, start_time);
        if (cur_milis >= log_interval) {
            if (power_sampling_iters != 0) {
                avg_power /= power_sampling_iters;
                msg = "[" + action_name + "] " + MODULE_NAME + " " +
                        std::to_string(gpu_id) + " " +
                        IET_LOGGER_CURRENT_POWER_MSG + " " +
                        std::to_string(avg_power);
                rvs::lp::Log(msg, rvs::loginfo);
                log_to_json(IET_LOGGER_CURRENT_POWER_MSG,
                                std::to_string(avg_power), rvs::loginfo);
            }

            avg_power = 0;
            power_sampling_iters = 0;
            start_time = std::chrono::system_clock::now();
        } else {
            usleep(POWER_PROCESS_DELAY);
        }
    }
}
