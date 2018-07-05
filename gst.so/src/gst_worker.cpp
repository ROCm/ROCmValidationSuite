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

using std::string;

GSTWorker::GSTWorker() {}
GSTWorker::~GSTWorker() {}

void GSTWorker::run() {
    bool brun = true;
    string msg;
    uint64_t total_run_duration;
    uint64_t num_sgemm = 0;
    uint64_t total_milliseconds = 0;

    // worker thread has started
    msg = "[GST] worker thread for gpu_id: " + std::to_string(gpu_id) +
                " is running...";
    log(msg.c_str(), rvs::logdebug);

    if (run_duration_ms != 0) {
        total_run_duration = run_duration_ms;
    } else {  // no duration specified
        total_run_duration = ramp_interval;
    }

    std::chrono::time_point<std::chrono::system_clock> gst_session_start_time;

    // rvs_blas ...  testing
    rvs_blas gpu_blas(gpu_device_index, 6000, 6000, 6000);
    if (!gpu_blas.error()) {
        msg = "[GST] Generating matrix for [: " + std::to_string(gpu_id) +"]";
        log(msg.c_str(), rvs::logdebug);
        gpu_blas.generate_random_matrix_data();
        msg = "[GST] Matrix ready for [: " + std::to_string(gpu_id) +"]";
        log(msg.c_str(), rvs::logdebug);
        gpu_blas.copy_data_to_gpu();
        gst_session_start_time = std::chrono::system_clock::now();
        while (brun) {
            // run GEMM & wait for completion
            gpu_blas.run_blass_gemm();
            while (!gpu_blas.is_gemm_op_complete()) {}
            if (!gpu_blas.error())
                num_sgemm++;
            std::chrono::time_point<std::chrono::system_clock>
                    gst_session_end_time = std::chrono::system_clock::now();
            auto milliseconds =
                    std::chrono::duration_cast<std::chrono::milliseconds>
                        (gst_session_end_time - gst_session_start_time);
            if (milliseconds.count() >= run_duration_ms) {
                total_milliseconds = milliseconds.count();
                brun = false;
            }
        }

        double sec = static_cast<double>(total_milliseconds)/1000;
        double gflops = static_cast<double>(gpu_blas.gemm_gflop_count()
                            * num_sgemm)/sec;

        msg = "[GST] GFLOPS[" + std::to_string(gpu_id) +
            "] = " + std::to_string(gflops);
        log(msg.c_str(), rvs::logdebug);
    }

    msg = "[GST] worker thread for gpu_id: " +
            std::to_string(gpu_id) +" has finished...";
    log(msg.c_str(), rvs::logdebug);
}
