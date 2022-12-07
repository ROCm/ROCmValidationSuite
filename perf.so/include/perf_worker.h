/********************************************************************************
 *
 * Copyright (c) 2018-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef PERF_SO_INCLUDE_PERF_WORKER_H_
#define PERF_SO_INCLUDE_PERF_WORKER_H_

#include <string>
#include <memory>
#include "include/rvsthreadbase.h"
#include "include/rvs_blas.h"
#include "include/rvsactionbase.h"

#define PERF_RESULT_PASS_MESSAGE         "true"
#define PERF_RESULT_FAIL_MESSAGE         "false"

/**
 * @class PERFWorker
 * @ingroup PERF
 *
 * @brief PERFWorker action implementation class
 *
 * Derives from rvs::ThreadBase and implements actual action functionality
 * in its run() method.
 *
 */
class PERFWorker : public rvs::ThreadBase {
 public:
    PERFWorker();
    virtual ~PERFWorker();

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

    //! sets the run delay
    void set_run_wait_ms(uint64_t _run_wait_ms) { run_wait_ms = _run_wait_ms; }
    //! returns the run delay
    uint64_t get_run_wait_ms(void) { return run_wait_ms; }

    //! sets the total stress test run duration
    void set_run_duration_ms(uint64_t _run_duration_ms) {
        run_duration_ms = _run_duration_ms;
    }
    //! returns the total stress test run duration
    uint64_t get_run_duration_ms(void) { return run_duration_ms; }

    //! sets the stress test ramp duration
    void set_ramp_interval(uint64_t _ramp_interval) {
        ramp_interval = _ramp_interval;
    }
    //! returns the stress test ramp duration
    uint64_t get_ramp_interval(void) { return ramp_interval; }

    //! sets the time interval at which the module reports the average GFlops
    void set_log_interval(uint64_t _log_interval) {
        log_interval = _log_interval;
    }
    //! returns the time interval at which the module reports the average GFlops
    uint64_t get_log_interval(void) { return log_interval; }

    //! sets the maximum allowed number of target_stress violations
    void set_max_violations(uint64_t _max_violations) {
        max_violations = _max_violations;
    }
    //! returns the maximum allowed number of target_stress violations
    uint64_t get_max_violations(void) { return max_violations; }

    //! sets the copy_matrix (true = the matrix will be copied to GPU each
    //! time a new SGEMM will run, false = the matrix will be copied only once)
    void set_copy_matrix(bool _copy_matrix) { copy_matrix = _copy_matrix; }
    //! returns the copy_matrix value
    bool get_copy_matrix(void) { return copy_matrix; }

    //! sets the target stress (in GFlops) that the GPU will try to achieve
    void set_target_stress(float _target_stress) {
        target_stress = _target_stress;
    }
    //! returns the target stress (in GFlops) that the GPU will try to achieve
    float get_target_stress(void) { return target_stress; }

    //! sets hot calls
    void set_perf_hot_calls(uint64_t _hot_calls) {
        perf_hot_calls = _hot_calls;
    }
 
    //! sets hot calls
    uint64_t get_perf_hot_calls(void) {
        return perf_hot_calls;
    }

    //! sets the SGEMM matrix size
    void set_matrix_size_a(uint64_t _matrix_size_a) {
        matrix_size_a = _matrix_size_a;
    }
   //! sets the SGEMM matrix size
    void set_matrix_size_b(uint64_t _matrix_size_b) {
        matrix_size_b = _matrix_size_b;
    }
   //! sets the SGEMM matrix size
    void set_matrix_size_c(uint64_t _matrix_size_c) {
        matrix_size_c = _matrix_size_c;
    }
    //! sets the transpose matrix a
    void set_matrix_transpose_a(int transa) {
        perf_trans_a = transa;
    }
    //! sets the transpose matrix b
    void set_matrix_transpose_b(int transb) {
        perf_trans_b = transb;
    }
    //! sets alpha val
    void set_alpha_val(float alpha_val) {
        perf_alpha_val = alpha_val;
    }
    //! sets beta val
    void set_beta_val(float beta_val) {
        perf_beta_val = beta_val;
    }

    //! sets offsets
    void set_lda_offset(int lda) {
        perf_lda_offset = lda;
    }
    //! sets offsets
    void set_ldb_offset(int ldb) {
        perf_ldb_offset = ldb;
    }
    //! sets offsets
    void set_ldc_offset(int ldc) {
        perf_ldc_offset = ldc;
    }

    //! returns the SGEMM matrix size
    uint64_t get_matrix_size_a(void) { return matrix_size_a; }

    //! returns the SGEMM matrix size
    uint64_t get_matrix_size_b(void) { return matrix_size_b; }

    //! returns the SGEMM matrix size
    uint64_t get_matrix_size_c(void) { return matrix_size_b; }

    //! sets the GFlops tolerance
    void set_tolerance(float _tolerance) { tolerance = _tolerance; }
    //! returns the GFlops tolerance
    float get_tolerance(void) { return tolerance; }


    //! returns the difference (in milliseconds) between 2 points in time
    uint64_t time_diff(
                std::chrono::time_point<std::chrono::system_clock> t_end,
                    std::chrono::time_point<std::chrono::system_clock> t_start);

    //! sets the JSON flag
    static void set_use_json(bool _bjson) { bjson = _bjson; }
    //! returns the JSON flag
    static bool get_use_json(void) { return bjson; }

    void set_perf_ops_type(std::string _ops_type) { perf_ops_type = _ops_type; }

    //! Set action callback 
    void set_callback(void (*_callback)(const rvs::action_result_t * result, void * user_param), void * _user_param) {
      callback = _callback;
      user_param = _user_param;
    }

 protected:
    void setup_blas(int *error, std::string *err_description);
    void hit_max_gflops(int *error, std::string *err_description);
    bool do_perf_ramp(int *error, std::string *err_description);
    bool do_perf_stress_test(int *error, std::string *err_description);
    void log_perf_test_result(bool perf_test_passed);
    virtual void run(void);
    void log_to_json(const std::string &key, const std::string &value,
                     int log_level);
    void log_interval_gflops(double gflops_interval);
    bool check_gflops_violation(double gflops_interval);
    void check_target_stress(double gflops_interval);
    void usleep_ex(uint64_t microseconds);

 protected:
    //! name of the action
    std::string action_name;
    //! index of the GPU that will run the stress test
    int gpu_device_index;
    //Matrix transpose A
    int perf_trans_a;
    //Matrix transpose B
    int perf_trans_b;
    //! ID of the GPU that will run the stress test
    uint16_t gpu_id;
    //PERF aplha value 
    float perf_alpha_val;
    //PERF beta value
    float perf_beta_val;
    //leading offsets
    int perf_lda_offset;
    int perf_ldb_offset;
    int perf_ldc_offset;
    //! stress test run delay
    uint64_t run_wait_ms;
    //! stress test run duration
    uint64_t run_duration_ms;
    //! stress test ramp duration
    uint64_t ramp_interval;
    //! time interval at which the module reports the average GFlops
    uint64_t log_interval;
    //! maximum allowed number of target_stress violations
    uint64_t max_violations;
    //! specifies whether to copy the matrix to the GPU for each SGEMM operation
    bool copy_matrix;
    //! target stress (in GFlops) that the GPU will try to achieve
    float target_stress;
    //! GFlops tolerance (how much the GFlops can fluctuare after
    //! the ramp period for the test to succeed)
    float tolerance;
    //! SGEMM matrix size
    uint64_t matrix_size_a;
    uint64_t matrix_size_b;
    uint64_t matrix_size_c;
    //num of hot calls
    uint64_t perf_hot_calls;
    //! actual ramp time in case the GPU achieves the given target_stress Gflops
    uint64_t ramp_actual_time;
    //! rvs_blas pointer
    std::unique_ptr<rvs_blas> gpu_blas;
    //! max gflops achieved during the stress test
    double max_gflops;
    //! delay used to reduce SGEMM frequency
    double delay_target_stress;
    //! TRUE if JSON output is required
    static bool bjson;
    //Type of operation
    std::string perf_ops_type;
    // callback
    void (*callback)(const rvs::action_result_t * result, void * user_param);
    // User parameter
    void * user_param;
};

#endif  // PERF_SO_INCLUDE_PERF_WORKER_H_
