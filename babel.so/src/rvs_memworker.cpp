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
#include <memory>
#include <iostream>
#include <sys/time.h>
#include <mutex>

#include "hip/hip_runtime.h"
#include "include/rvs_memworker.h"
#include "include/rvsloglp.h"

#include "include/Stream.h"

using std::string;
bool MemWorker::bjson = false;

extern void run_babel(std::pair<int, uint16_t> device, int num_times, int array_size, bool output_csv, bool mibibytes,
    int test_type, int subtest, uint16_t dwords_per_lane, uint16_t chunks_per_block, uint16_t tb_size, bool json, std::string action);

#define FLOAT_TEST     1 
#define DOUBLE_TEST    2 
#define TRIAD_FLOAT    3 
#define TRIAD_DOUBLE   4 


MemWorker::MemWorker() {}
MemWorker::~MemWorker() {}

/**
 * @brief performs the stress test on the given GPU
 */
void MemWorker::run() {
    hipDeviceProp_t props;
    char*           ptr = NULL;
    string          err_description;
    string          msg;
    int             deviceId;
    uint16_t        gpuId;
    std::pair<int, uint16_t> device;

    // log MEM stress test - start message
    msg = "[" + action_name + "] " + "[GPU:: " +
            std::to_string(gpu_id) + "] " + "Starting the Babel memory stress test";
    rvs::lp::Log(msg, rvs::logresults);

    /* Device Index */
    deviceId  = get_gpu_device_index();
    device.first = deviceId;

    /* GPU ID */
    gpuId = get_gpu_id();
    device.second = gpuId;

    HIP_CHECK(hipGetDeviceProperties(&props, deviceId));

    HIP_CHECK(hipSetDevice(deviceId));

    run_babel(device, num_iterations, array_size, output_csv, mibibytes,
        test_type, subtest, dwords_per_lane, chunks_per_block, tb_size,
        MemWorker::bjson, action_name);
}

