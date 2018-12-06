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
#ifndef IET_SO_INCLUDE_IET_WORKER_H_
#define IET_SO_INCLUDE_IET_WORKER_H_

#include <string>
#include <memory>
#include "include/rvsthreadbase.h"
#include "include/blas_worker.h"
#include "include/log_worker.h"

/**
 * @class IETWorker
 * @ingroup IET
 *
 * @brief IETWorker action implementation class
 *
 * Derives from rvs::ThreadBase and implements actual action functionality
 * in its run() method.
 *
 */
class IETWorker : public rvs::ThreadBase {
 public:
    IETWorker();
    virtual ~IETWorker();

    //! sets action name
    void set_name(const std::string& name) { action_name = name; }
    //! returns action name
    const std::string& get_name(void) { return action_name; }

    //! sets GPU ID
    void set_gpu_id(uint16_t _gpu_id) { gpu_id = _gpu_id; }
    //! returns GPU ID
    uint16_t get_gpu_id(void) { return gpu_id; }

    //! sets the GPU index
    void set_gpu_device_index(int _gpu_device_index) {
        gpu_device_index = _gpu_device_index;
    }
    //! returns the GPU index
    int get_gpu_device_index(void) { return gpu_device_index; }

    //! sets the GPU power-index
    void set_pwr_device_id(int _pwr_device_id) {
        pwr_device_id = _pwr_device_id;
    }
    //! returns the GPU power-index
    int get_pwr_device_id(void) { return pwr_device_id; }

    //! sets the run delay
    void set_run_wait_ms(uint64_t _run_wait_ms) {
        run_wait_ms = _run_wait_ms;
    }
    //! returns the run delay
    uint64_t get_run_wait_ms(void) { return run_wait_ms; }

    //! sets the total EDPp test run duration
    void set_run_duration_ms(uint64_t _run_duration_ms) {
        run_duration_ms = _run_duration_ms;
    }
    //! returns the total EDPp test run duration
    uint64_t get_run_duration_ms(void) { return run_duration_ms; }

    //! sets the EDPp test ramp duration
    void set_ramp_interval(uint64_t _ramp_interval) {
        ramp_interval = _ramp_interval;
    }
    //! returns the EDPp test ramp duration
    uint64_t get_ramp_interval(void) { return ramp_interval; }

    //! sets the time interval at which the module reports the GPU's power
    void set_log_interval(uint64_t _log_interval) {
        log_interval = _log_interval;
    }
    //! returns the time interval at which the module reports the GPU's power
    uint64_t get_log_interval(void) { return log_interval; }

    //! sets the sampling rate for the target_power
    void set_sample_interval(uint64_t _sample_interval) {
        sample_interval = _sample_interval;
    }
    //! returns the sampling rate for the target_power
    uint64_t get_sample_interval(void) { return sample_interval; }

    //! sets the maximum allowed number of target_power violations
    void set_max_violations(uint64_t _max_violations) {
        max_violations = _max_violations;
    }
    //! returns the maximum allowed number of target_power violations
    uint64_t get_max_violations(void) { return max_violations; }

    //! sets the target power level for the EDPp test
    void set_target_power(float _target_power) {
        target_power = _target_power;
    }
    //! returns the target power level for the test
    float get_target_power(void) { return target_power; }

    //! sets the SGEMM matrix size
    void set_matrix_size(uint64_t _matrix_size) {
        matrix_size = _matrix_size;
    }
    //! returns the SGEMM matrix size
    uint64_t get_matrix_size(void) { return matrix_size; }

    //! sets the EDPp power tolerance
    void set_tolerance(float _tolerance) { tolerance = _tolerance; }
    //! returns the EDPp power tolerance
    float get_tolerance(void) { return tolerance; }

    //! sets the JSON flag
    static void set_use_json(bool _bjson) { bjson = _bjson; }
    //! returns the JSON flag
    static bool get_use_json(void) { return bjson; }

 protected:
    virtual void run(void);
    bool do_gpu_init_training(std::string *err_description);
    void compute_gpu_stats(void);
    void compute_new_sgemm_freq(float avg_power);
    bool do_iet_ramp(int *error, std::string *err_description);
    bool do_iet_power_stress(void);
    void log_to_json(const std::string &key, const std::string &value,
                        int log_level);


 protected:
    //! name of the action
    std::string action_name;
    //! index of the GPU (as reported by HIP API) that will run the EDPp test
    int gpu_device_index;
    //! ID of the GPU that will run the EDPp test
    uint16_t gpu_id;
    //! index of the GPU device as requested by rocm_smi
    uint32_t pwr_device_id;
    //! EDPp test run delay
    uint64_t run_wait_ms;
    //! EDPp test run duration
    uint64_t run_duration_ms;
    //! stress test ramp duration
    uint64_t ramp_interval;
    //! time interval at which the GPU's power is logged out
    uint64_t log_interval;
    //! sampling rate for the target_power
    uint64_t sample_interval;
    //! maximum allowed number of target_power violations
    uint64_t max_violations;
    //! target power level for the test
    float target_power;
    //! power tolerance (how much the target_power can fluctuare after
    //! the ramp period for the test to succeed)
    float tolerance;
    //! SGEMM matrix size
    uint64_t matrix_size;
    //! TRUE if JSON output is required
    static bool bjson;
    //! blas_worker pointer
    std::unique_ptr<blas_worker> gpu_worker;
    //! log_worker pointer
    std::unique_ptr<log_worker> pwr_log_worker;

    //! actual training time
    uint64_t training_time_ms;
    //! actual ramp time
    uint64_t ramp_actual_time;
    //! number of SGEMMs that the GPU achieved during the training
    uint64_t num_sgemms_training;
    //! average GPU power during training
    float avg_power_training;
    //! the SGEMM delay which gives the actual GPU SGEMM frequency
    float sgemm_si_delay;
};
#endif  // IET_SO_INCLUDE_IET_WORKER_H_
