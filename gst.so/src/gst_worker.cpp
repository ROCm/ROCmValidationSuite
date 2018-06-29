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

#include "rvsliblogger.h"
#include "rvs_module.h"
#include "rvsloglp.h"

using namespace std;

GSTWorker::GSTWorker() {}
GSTWorker::~GSTWorker() {}

void GSTWorker::run() {
    bool brun = true;
    string msg;
    unsigned long total_run_duration;
    
    // worker thread has started    
    msg = "[GST] worker thread for gpu_id: " + std::to_string(gpu_id) +" is running...";
    log(msg.c_str(), rvs::logdebug);
    
    if (run_duration_ms != 0) {
        total_run_duration = run_duration_ms;
    } else {  // no duration specified
        total_run_duration = ramp_interval;
    }
            
    std::chrono::time_point<std::chrono::system_clock> gst_session_start_time = std::chrono::system_clock::now();
    
    while (brun) {
        // TODO(Tudor) add the actual rocBlas SGEMM logic
        std::chrono::time_point<std::chrono::system_clock> gst_session_curr_time = std::chrono::system_clock::now();
	auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(gst_session_curr_time - gst_session_start_time);
	if (milliseconds.count() >= run_duration_ms) {
            brun = false;
        }        
    }
    
    msg = "[GST] worker thread for gpu_id: " + std::to_string(gpu_id) +" has finished...";
    log(msg.c_str(), rvs::logdebug);
}
