/********************************************************************************
 *
 * Copyright (c) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <mutex>
#include <condition_variable>
#include "include/rvsthreadbase.h"
#include "include/rvs_blas.h"
#include "include/rvs_util.h"
#include "include/rvsactionbase.h"
#include "include/action.h"

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
    //! sets action
    void set_action(const iet_action& _action) { action = _action; }
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

    //! sets the matrix size
    void set_matrix_size(uint64_t _matrix_size) {
        matrix_size = _matrix_size;
    }
    //! returns the matrix size
    uint64_t get_matrix_size(void) { return matrix_size; }

    //! sets gemm operation type
    void set_iet_ops_type(std::string ops_type) { iet_ops_type = ops_type; }
    //! get gemm operation type
    std::string get_ops_type(void) { return iet_ops_type; }

    //! sets gemm data type
    void set_iet_data_type(std::string data_type) { iet_data_type = data_type; }
    //! get gemm data type
    std::string get_data_type(void) { return iet_data_type; }

    //! sets the EDPp power tolerance
    void set_tp_flag(bool _tp_flag) { iet_tp_flag = _tp_flag; }
    //! returns the EDPp power tolerance
    bool get_tp_flag(void) { return iet_tp_flag; }

    //! sets the EDPp power tolerance
    void set_tolerance(float _tolerance) { tolerance = _tolerance; }
    //! returns the EDPp power tolerance
    float get_tolerance(void) { return tolerance; }

    //! sets the JSON flag
    static void set_use_json(bool _bjson) { bjson = _bjson; }

    //! returns the JSON flag
    static bool get_use_json(void) { return bjson; }

    //! returns the matrix size a
    uint64_t get_matrix_size_a(void) { return matrix_size_a; }

    //! returns the matrix size b
    uint64_t get_matrix_size_b(void) { return matrix_size_b; }

    //! returns the matrix size c
    uint64_t get_matrix_size_c(void) { return matrix_size_c; }

    //! sets the transpose matrix a
    void set_matrix_transpose_a(int transa) {
        iet_trans_a = transa;
    }
    //! sets the transpose matrix b
    void set_matrix_transpose_b(int transb) {
        iet_trans_b = transb;
    }
    //! sets alpha val
    void set_alpha_val(float alpha_val) {
        iet_alpha_val = alpha_val;
    }
    //! sets beta val
    void set_beta_val(float beta_val) {
        iet_beta_val = beta_val;
    }

    //! sets offsets
    void set_lda_offset(int lda) {
        iet_lda_offset = lda;
    }
    //! sets offsets
    void set_ldb_offset(int ldb) {
        iet_ldb_offset = ldb;
    }
    //! sets offsets
    void set_ldc_offset(int ldc) {
        iet_ldc_offset = ldc;
    }
    //! sets offsets
    void set_ldd_offset(int ldd) {
        iet_ldd_offset = ldd;
    }
   //! sets the matrix size a
    void set_matrix_size_a(uint64_t _matrix_size_a) {
        matrix_size_a = _matrix_size_a;
    }
   //! sets the matrix size b
    void set_matrix_size_b(uint64_t _matrix_size_b) {
        matrix_size_b = _matrix_size_b;
    }
   //! sets the matrix size c
    void set_matrix_size_c(uint64_t _matrix_size_c) {
        matrix_size_c = _matrix_size_c;
    }

    //! sets bandwidth workload enable/disable
    void set_bw_workload(bool _bw_workload) { iet_bw_workload = _bw_workload; }

    //! returns bandwidth workload status
    bool get_bw_workload(void) { return iet_bw_workload; }

    //! sets compute workload enable/disable
    void set_cp_workload(bool _cp_workload) { iet_cp_workload = _cp_workload; }

    //! returns compute workload status
    bool get_cp_workload(void) { return iet_cp_workload; }

   //! sets hot calls
    void set_hot_calls(uint64_t _hot_calls) { iet_hot_calls = _hot_calls; }

    //! returns hot calls
    uint64_t get_iet_hot_calls(void) { return iet_hot_calls; }

    //! sets the matrix init
    void set_matrix_init(std::string _matrix_init) { matrix_init = _matrix_init; }

    //! returns matrix init
    std::string get_matrix_init(void) { return matrix_init; }

    //! sets the gemm mode
    void set_gemm_mode(std::string _gemm_mode) { gemm_mode = _gemm_mode; }

    //! returns gemm mode
    std::string get_gemm_mode(void) { return gemm_mode; }

    //! sets the batch size
    void set_batch_size(int _batch_size) { batch_size = _batch_size; }

    //! returns the batch size
    int get_batch_size(void) { return batch_size; }

    //! sets the matrix a stride
    void set_stride_a(uint64_t _stride_a) { stride_a = _stride_a; }

    //! returns the matrix a stride
    uint64_t get_stride_a(void) { return stride_a; }

    //! sets the matrix b stride
    void set_stride_b(uint64_t _stride_b) { stride_b = _stride_b; }

    //! returns the matrix b stride
    uint64_t get_stride_b(void) { return stride_b; }

    //! sets the matrix c stride
    void set_stride_c(uint64_t _stride_c) { stride_c = _stride_c; }

    //! returns the matrix c stride
    uint64_t get_stride_c(void) { return stride_c; }

    //! sets the matrix d stride
    void set_stride_d(uint64_t _stride_d) { stride_d = _stride_d; }

    //! returns the matrix d stride
    uint64_t get_stride_d(void) { return stride_d; }

    //! BLAS callback
    static void blas_callback (bool status, void *user_data);

 protected:
    virtual void run(void);
    bool do_gpu_init_training(int gpuIdx,  uint64_t matrix_size, std::string  iet_ops_type);
    void compute_gpu_stats(void);
    void compute_new_sgemm_freq(float avg_power);
    bool do_iet_power_stress(void);
    void log_interval_gflops(double gflops_interval);
    void log_to_json(const std::string &key, const std::string &value,
        int log_level);

    void computeThread(void);
    void bandwidthThread(void);
 protected:
    std::unique_ptr<rvs_blas> gpu_blas;

    //! name of the action
    std::string action_name;
    //! action instance
    iet_action action;
    //! index of the GPU (as reported by HIP API) that will run the EDPp test
    int gpu_device_index;
    //! index of GPU (in view of smi lib) which is sometimes different to above index
    int smi_device_index;
    //! ID of the GPU that will run the EDPp test
    uint16_t gpu_id;

    int blas_error;

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
    //! matrix size
    uint64_t matrix_size;
    //! TRUE if JSON output is required
    static bool bjson;
    bool sgemm_success;
    //! gemm operation type
    std::string iet_ops_type;
    //! gemm data type
    std::string iet_data_type;

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
   //! matrix sizes
    uint64_t matrix_size_a;
    uint64_t matrix_size_b;
    uint64_t matrix_size_c;
    //! leading offsets
    int iet_lda_offset;
    int iet_ldb_offset;
    int iet_ldc_offset;
    int iet_ldd_offset;
    //! Matrix transpose A
    int iet_trans_a;
    //! Matrix transpose B
    int iet_trans_b;
    //! IET aplha value
    float iet_alpha_val;
    //! IET beta value
    float iet_beta_val;
    //! IET TP flag
    bool iet_tp_flag;
    //! Bandwidth workload enable/disable
    bool iet_bw_workload;
    //! Bandwidth compute enable/disable
    bool iet_cp_workload;
    //! hot calls
    uint64_t iet_hot_calls;
    //! matrix init
    std::string matrix_init;
    //! gemm mode : basic (single), batched or strided batched
    std::string gemm_mode;
    //! Matrix batch count
    int batch_size;
    //! Stride from the start of matrix a(i)
    //! to next matrix a(i+1) in the strided batch
    uint64_t stride_a;
    //! Stride from the start of matrix b(i)
    //! to next matrix b(i+1) in the strided batch
    uint64_t stride_b;
    //! Stride from the start of matrix c(i)
    //! to next matrix c(i+1) in the strided batch
    uint64_t stride_c;
    //! Stride from the start of matrix d(i)
    //! to next matrix d(i+1) in the strided batch
    uint64_t stride_d;

    bool endtest = false;
    //! GEMM operations synchronization mutex
    std::mutex mutex;
    //! GEMM operations synchronization condition variable
    std::condition_variable cv;
    //! blas gemm operations status
    bool blas_status;
};

#endif  // IET_SO_INCLUDE_IET_WORKER_H_
