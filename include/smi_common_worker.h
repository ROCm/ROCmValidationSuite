/********************************************************************************
 *
 * Copyright (c) 2018-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef TST_SO_INCLUDE_TST_WORKER_H_
#define TST_SO_INCLUDE_TST_WORKER_H_

#include <string>
#include <memory>
#include <mutex>
#include <condition_variable>
#include "include/rvsthreadbase.h"
#include "include/rvs_blas.h"
#include "include/rvs_util.h"
#include "include/rvsactionbase.h"
#include "include/action.h"

/**
 * @class smi_worker
 * @ingroup smi
 *
 * @brief smi_worker action implementation class
 *
 * Derives from rvs::ThreadBase and implements actual action functionality
 * in its run() method.
 *
 */
class smi_worker : public rvs::ThreadBase {
 public:
    smi_worker();
    virtual ~smi_worker();

    //! sets action name
    void set_name(const std::string& name) { action_name = name; }
    //! sets action
    void set_action(const action& _action) { action = _action; }
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
    //! sets the GPU smi index
    void set_smi_device_index(int _smi_device_index) {
        smi_device_index = _smi_device_index;
    }
    //! returns the GPU smi index
    int get_smi_device_index(void) { return smi_device_index; }

    //! sets the run delay
    void set_run_wait_ms(uint64_t _run_wait_ms) {
        run_wait_ms = _run_wait_ms;
    }
    //! returns the run delay
    uint64_t get_run_wait_ms(void) { return run_wait_ms; }

    //! sets the total TST test run duration
    void set_run_duration_ms(uint64_t _run_duration_ms) {
        run_duration_ms = _run_duration_ms;
    }
    //! returns the total TST test run duration
    uint64_t get_run_duration_ms(void) { return run_duration_ms; }

    //! sets the TST test ramp duration
    void set_ramp_interval(uint64_t _ramp_interval) {
        ramp_interval = _ramp_interval;
    }
    //! returns the TST test ramp duration
    uint64_t get_ramp_interval(void) { return ramp_interval; }

    //! sets the time interval at which the module reports the GPU's temperature 
    void set_log_interval(uint64_t _log_interval) {
        log_interval = _log_interval;
    }
    //! returns the time interval at which the module reports the GPU's temperature
    uint64_t get_log_interval(void) { return log_interval; }

    //! sets the sampling rate for the target_temperature
    void set_sample_interval(uint64_t _sample_interval) {
        sample_interval = _sample_interval;
    }
    //! returns the sampling rate for the target_temperature
    uint64_t get_sample_interval(void) { return sample_interval; }

    //! sets the maximum allowed number of target_temperature violations
    void set_max_violations(uint64_t _max_violations) {
        max_violations = _max_violations;
    }
    //! returns the maximum allowed number of target_temperature violations
    uint64_t get_max_violations(void) { return max_violations; }

    //! sets the target temperature level for the TST test
    void set_target_temp(float _temp) {
        target_temp = _temp;
    }

    //! sets the target trottle temperature level for the TST test
    void set_throttle_temp(float _throttle_temp) {
        throttle_temp = _throttle_temp;
    }

    //! returns the target temperature level for the test
    float get_target_temp(void) { return target_temp; }

    //! returns the target trottle temperature level for the test
    float get_throttle_temp(void) { return throttle_temp; }

    //! sets the SGEMM matrix size
    void set_matrix_size(uint64_t _matrix_size) {
        matrix_size = _matrix_size;
    }
    //! returns the SGEMM matrix size
    uint64_t get_matrix_size(void) { return matrix_size; }

    //! sets the GEMM operation type
    void set_tst_ops_type(std::string ops_type) { tst_ops_type = ops_type; }
    //! returns GEMM operation type
    std::string get_ops_type(void) { return tst_ops_type; }

    //! sets the target temperature flag
    void set_tt_flag(bool _tt_flag) { tst_tt_flag = _tt_flag; }
    //! returns the target temperature flag
    bool get_tt_flag(void) { return tst_tt_flag; }

    //! sets the TST temperature tolerance
    void set_tolerance(float _tolerance) { tolerance = _tolerance; }
    //! returns the TST temperature tolerance
    float get_tolerance(void) { return tolerance; }

    //! sets the JSON flag
    static void set_use_json(bool _bjson) { bjson = _bjson; }

    //! returns the JSON flag
    static bool get_use_json(void) { return bjson; }
    //! returns the SGEMM matrix size
    uint64_t get_matrix_size_a(void) { return matrix_size_a; }

    //! returns the SGEMM matrix size
    uint64_t get_matrix_size_b(void) { return matrix_size_b; }

    //! returns the SGEMM matrix size
    uint64_t get_matrix_size_c(void) { return matrix_size_b; }

    //! sets the transpose matrix a
    void set_matrix_transpose_a(int transa) {
        tst_trans_a = transa;
    }
    //! sets the transpose matrix b
    void set_matrix_transpose_b(int transb) {
        tst_trans_b = transb;
    }
    //! sets alpha val
    void set_alpha_val(float alpha_val) {
        tst_alpha_val = alpha_val;
    }
    //! sets beta val
    void set_beta_val(float beta_val) {
        tst_beta_val = beta_val;
    }

    //! sets offsets
    void set_lda_offset(int lda) {
        tst_lda_offset = lda;
    }
    //! sets offsets
    void set_ldb_offset(int ldb) {
        tst_ldb_offset = ldb;
    }
    //! sets offsets
    void set_ldc_offset(int ldc) {
        tst_ldc_offset = ldc;
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

    //! BLAS callback
    static void blas_callback (bool status, void *user_data);

 protected:
    virtual void run(void);
    bool do_gpu_init_training(int gpuIdx,  uint64_t matrix_size, std::string  tst_ops_type);
    void compute_gpu_stats(void);
    bool do_thermal_stress(void);
    void log_interval_gflops(double gflops_interval);
    void log_to_json(const std::string &key, const std::string &value,
        int log_level);
    void blasThread(int gpuIdx,  uint64_t matrix_size, std::string  tst_ops_type,
        bool start, uint64_t run_duration_ms, int transa, int transb, float alpha, float beta,
        int tst_lda_offset, int tst_ldb_offset, int tst_ldc_offset);
 protected:
    std::unique_ptr<rvs_blas> gpu_blas;

    //! name of the action
    std::string action_name;
    //! action instance
    tst_action action;
    //! index of the GPU (as reported by HIP API) that will run the TST test
    int gpu_device_index;
    //! index of GPU (in view of smi lib) which is sometimes different to above index
    int smi_device_index;
    //! ID of the GPU that will run the TST test
    uint16_t gpu_id;

    int blas_error;

    //! TST test run delay
    uint64_t run_wait_ms;
    //! TST test run duration
    uint64_t run_duration_ms;
      //! stress test ramp duration
    uint64_t ramp_interval;
    //! time interval at which the GPU's temperature is logged out
    uint64_t log_interval;
    //! sampling rate for the target_target
    uint64_t sample_interval;
    //! maximum allowed number of target_temperature violations
    uint64_t max_violations;
    //! target temperature level for the test
    float target_temp;
    //! target trottle temperature level for the test
    float throttle_temp;
    //! temperature tolerance (how much the target_temperature can fluctuare after
    //! the ramp period for the test to succeed)
    float tolerance;
    //! SGEMM matrix size
    uint64_t matrix_size;
    //! TRUE if JSON output is required
    static bool bjson;
    bool sgemm_success;
    //! GEMM operation type
    std::string tst_ops_type;

    //! actual training time
    uint64_t training_time_ms;
    //! actual ramp time
    uint64_t ramp_actual_time;
    //! number of SGEMMs that the GPU achieved during the training
    uint64_t num_sgemms_training;
    //! the SGEMM delay which gives the actual GPU SGEMM frequency
    float sgemm_si_delay;
    //! SGEMM matrix size
    uint64_t matrix_size_a;
    uint64_t matrix_size_b;
    uint64_t matrix_size_c;
    //! leading offsets
    int tst_lda_offset;
    int tst_ldb_offset;
    int tst_ldc_offset;
    //! Matrix transpose A
    int tst_trans_a;
    //! Matrix transpose B
    int tst_trans_b;
    //! TST aplha value
    float tst_alpha_val;
    //! TST beta value
    float tst_beta_val;
    //! TST target temperature flag
    bool tst_tt_flag;
    bool endtest = false;
    //! GEMM operations synchronization mutex
    std::mutex mutex;
    //! GEMM operations synchronization condition variable
    std::condition_variable cv;
    //! blas gemm operations status
    bool blas_status;
};

#endif  // TST_SO_INCLUDE_TST_WORKER_H_
