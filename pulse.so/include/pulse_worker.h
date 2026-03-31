/********************************************************************************
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef PULSE_SO_INCLUDE_PULSE_WORKER_H_
#define PULSE_SO_INCLUDE_PULSE_WORKER_H_

#include <string>
#include <memory>
#include <mutex>
#include <barrier>
#include <vector>
#include <atomic>
#include <cstdint>

#include "include/rvsthreadbase.h"
#include "include/rvs_blas.h"
#include "include/rvs_util.h"
#include "include/rvsactionbase.h"
#include "include/action.h"

/**
 * @class PulseWorker
 * @ingroup PULSE
 *
 * @brief PulseWorker action implementation class
 *
 * Derives from rvs::ThreadBase and implements the per-GPU pulse
 * stress workload including cross-GPU synchronized avalanche release.
 */
class PulseWorker : public rvs::ThreadBase {
 public:
    PulseWorker();
    virtual ~PulseWorker();

    void set_name(const std::string& name) { action_name = name; }
    void set_action(const pulse_action& _action) { action = _action; }
    const std::string& get_name(void) { return action_name; }

    void set_gpu_id(uint16_t _gpu_id) { gpu_id = _gpu_id; }
    uint16_t get_gpu_id(void) { return gpu_id; }

    void set_gpu_device_index(int _gpu_device_index) {
        gpu_device_index = _gpu_device_index;
    }
    int get_gpu_device_index(void) { return gpu_device_index; }

    void set_smi_device_handle(amdsmi_processor_handle _handle) {
        smi_device_handle = _handle;
    }
    amdsmi_processor_handle get_smi_device_handle(void) {
        return smi_device_handle;
    }

    void set_run_duration_ms(uint64_t _run_duration_ms) {
        run_duration_ms = _run_duration_ms;
    }

    void set_sample_interval(uint64_t _sample_interval) {
        sample_interval = _sample_interval;
    }

    void set_log_interval(uint64_t _log_interval) {
        log_interval = _log_interval;
    }

    void set_pulse_rate(int _pulse_rate) {
        pulse_rate = _pulse_rate;
    }

    void set_high_phase_ratio(float _ratio) {
        high_phase_ratio = _ratio;
    }

    void set_tolerance(float _tolerance) {
        tolerance = _tolerance;
    }

    void set_matrix_size(uint64_t _matrix_size) {
        matrix_size = _matrix_size;
    }

    void set_ops_type(std::string _ops_type) {
        pulse_ops_type = _ops_type;
    }

    void set_data_type(std::string _data_type) {
        pulse_data_type = _data_type;
    }

    void set_out_data_type(std::string _out_data_type) {
        pulse_out_data_type = _out_data_type;
    }

    void set_matrix_transpose_a(int transa) { pulse_trans_a = transa; }
    void set_matrix_transpose_b(int transb) { pulse_trans_b = transb; }
    void set_alpha_val(float alpha) { pulse_alpha_val = alpha; }
    void set_beta_val(float beta) { pulse_beta_val = beta; }
    void set_lda_offset(int lda) { pulse_lda_offset = lda; }
    void set_ldb_offset(int ldb) { pulse_ldb_offset = ldb; }
    void set_ldc_offset(int ldc) { pulse_ldc_offset = ldc; }
    void set_ldd_offset(int ldd) { pulse_ldd_offset = ldd; }

    void set_workload_iterations(int _iters) {
        workload_iterations = _iters;
    }

    void set_halt_on_error(bool _halt) {
        halt_on_error = _halt;
    }

    void set_verify_mode(std::string _mode) {
        verify_mode = _mode;
    }

    void set_hot_calls(uint64_t _hot_calls) {
        pulse_hot_calls = _hot_calls;
    }

    void set_matrix_init(std::string _init) {
        matrix_init = _init;
    }

    void set_blas_source(std::string _source) {
        blas_source = _source;
    }

    void set_compute_type(std::string _type) {
        compute_type = _type;
    }

    void set_mcm_type(mcm_type_t _mcm_type) {
        mcm_type = _mcm_type;
    }

    void set_num_gpus(int _num_gpus) {
        num_gpus = _num_gpus;
    }

    void set_worker_index(int _idx) {
        worker_index = _idx;
    }

    void set_sync_resources(std::barrier<>* _cpu_barrier,
                            int32_t* _gpu_arrival_count,
                            int32_t* _gpu_release_flag,
                            std::atomic<bool>* _done_flag) {
        cpu_barrier = _cpu_barrier;
        gpu_arrival_count = _gpu_arrival_count;
        gpu_release_flag = _gpu_release_flag;
        done_flag = _done_flag;
    }

    static void set_use_json(bool _bjson) { bjson = _bjson; }
    static bool get_use_json(void) { return bjson; }
    bool get_result(void) { return result; }

 protected:
    virtual void run(void);
    bool do_pulse_stress(void);
    bool setup_blas(void);
    float read_power(void);
    float read_temperature(void);

    bool discover_valid_clock_levels(void);
    bool set_highest_clocks(void);
    bool set_lowest_clocks(void);
    bool restore_clocks(void);

    bool gpu_barrier_sync(bool time_up);

 protected:
    std::unique_ptr<rvs_blas> gpu_blas;

    std::string action_name;
    pulse_action action;
    int gpu_device_index;
    amdsmi_processor_handle smi_device_handle;
    uint16_t gpu_id;

    uint64_t run_duration_ms;
    uint64_t sample_interval;
    uint64_t log_interval;
    int pulse_rate;
    float high_phase_ratio;
    float tolerance;
    uint64_t matrix_size;

    std::string pulse_ops_type;
    std::string pulse_data_type;
    std::string pulse_out_data_type;
    int pulse_trans_a;
    int pulse_trans_b;
    float pulse_alpha_val;
    float pulse_beta_val;
    int pulse_lda_offset;
    int pulse_ldb_offset;
    int pulse_ldc_offset;
    int pulse_ldd_offset;

    int workload_iterations;
    bool halt_on_error;
    std::string verify_mode;
    uint64_t pulse_hot_calls;
    std::string matrix_init;
    std::string blas_source;
    std::string compute_type;
    mcm_type_t mcm_type;
    int num_gpus;
    int worker_index;

    //! Cross-GPU synchronization (set by action when parallel + multi-GPU)
    std::barrier<>* cpu_barrier;
    int32_t* gpu_arrival_count;
    int32_t* gpu_release_flag;
    std::atomic<bool>* done_flag;

    //! Valid clock levels discovered via AMDSMI
    std::vector<uint32_t> valid_gfx_levels;
    std::vector<uint32_t> valid_mem_levels;

    static bool bjson;
    bool result;
};

#endif  // PULSE_SO_INCLUDE_PULSE_WORKER_H_
