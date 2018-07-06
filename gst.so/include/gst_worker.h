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
#ifndef GST_SO_INCLUDE_GST_WORKER_H_
#define GST_SO_INCLUDE_GST_WORKER_H_

#include <string>
#include "rvsthreadbase.h"

class GSTWorker : public rvs::ThreadBase {
 public:
    GSTWorker();
    ~GSTWorker();

    void set_name(const std::string& name) { action_name = name; }
    const std::string& get_name(void) { return action_name; }

    void set_gpu_id(uint16_t _gpu_id) { gpu_id = _gpu_id; }
    uint16_t get_gpu_id(void) { return gpu_id; }

    void set_gpu_device_index(int _gpu_device_index) {
        gpu_device_index = _gpu_device_index;
    }

    int get_gpu_device_index(void) { return gpu_device_index; }

    void set_run_wait_ms(uint64_t _run_wait_ms) {
        run_wait_ms = _run_wait_ms;
    }

    uint64_t get_run_wait_ms(void) { return run_wait_ms; }

    void set_run_duration_ms(uint64_t _run_duration_ms) {
        run_duration_ms = _run_duration_ms;
    }

    uint64_t get_run_duration_ms(void) { return run_duration_ms; }

    void set_ramp_interval(uint64_t _ramp_interval) {
        ramp_interval = _ramp_interval;
    }

    uint64_t get_ramp_interval(void) { return ramp_interval; }

    void set_log_interval(uint64_t _log_interval) {
        log_interval = _log_interval;
    }

    uint64_t get_log_interval(void) { return log_interval; }

    void set_max_violations(int _max_violations) {
        max_violations = _max_violations;
    }

    int get_max_violations(void) { return max_violations; }

    void set_copy_matrix(bool _copy_matrix) { copy_matrix = _copy_matrix; }
    bool get_copy_matrix(void) { return copy_matrix; }

    void set_target_stress(float _target_stress) {
        target_stress = _target_stress;
    }

    float get_target_stress(void) { return target_stress; }

    void set_tolerance(float _tolerance) { tolerance = _tolerance; }
    float get_tolerance(void) { return tolerance; }

 protected:
    virtual void run(void);

 protected:
    std::string action_name;
    int gpu_device_index;
    uint16_t gpu_id;
    uint64_t run_wait_ms;
    uint64_t run_duration_ms;
    uint64_t ramp_interval;
    uint64_t log_interval;
    int max_violations;
    bool copy_matrix;
    float target_stress;
    float tolerance;
};

#endif  // GST_SO_INCLUDE_GST_WORKER_H_
