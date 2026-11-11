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
#ifndef PULSE_SO_INCLUDE_ACTION_H_
#define PULSE_SO_INCLUDE_ACTION_H_


#include <vector>
#include <string>
#include <map>
#include <atomic>
#include <cstdint>

#include "include/rvsactionbase.h"
#include "amd_smi/amdsmi.h"

// Shared by the GPU workers of one action. The host rendezvous can time out,
// so one worker that never arrives cannot leave the others blocked.
struct PulseSync {
  std::atomic<int> arrival{0};
  std::atomic<int> epoch{0};
  std::atomic<bool> abort_flag{false};
  int num_gpus{1};
  int wait_ms{10000};
  int32_t* gpu_arrival_count{nullptr};
  int32_t* gpu_release_flag{nullptr};
  int32_t* gpu_timeout_flag{nullptr};
};

using std::vector;
using std::string;

class PulseWorker;

/**
 * @class pulse_action
 * @ingroup PULSE
 *
 * @brief Pulse test action implementation class
 *
 * Derives from rvs::actionbase and implements the GPU power pulse
 * stress test in its run() method.
 */
class pulse_action: public rvs::actionbase {
 public:
    pulse_action();
    virtual ~pulse_action();
    virtual int run(void);

 protected:
    //! pulse cycle rate in Hz
    int pulse_rate;
    //! fraction of each cycle spent in high-power phase (0.0-1.0)
    float high_phase_ratio;
    //! GEMM operation type (sgemm, dgemm, hgemm, etc.)
    std::string pulse_ops_type;
    //! GEMM data type
    std::string pulse_data_type;
    //! GEMM output data type
    std::string pulse_out_data_type;
    //! square GEMM dimension, used when M/N/K are omitted
    uint64_t pulse_matrix_size;
    //! GEMM M, N, K. Each defaults to pulse_matrix_size.
    uint64_t pulse_matrix_m;
    uint64_t pulse_matrix_n;
    uint64_t pulse_matrix_k;
    //! GEMM alpha scalar
    float pulse_alpha_val;
    //! GEMM beta scalar
    float pulse_beta_val;
    //! transpose A setting
    int pulse_trans_a;
    //! transpose B setting
    int pulse_trans_b;
    //! leading dimension offsets
    int pulse_lda_offset;
    int pulse_ldb_offset;
    int pulse_ldc_offset;
    int pulse_ldd_offset;
    //! matrix initialization method
    std::string pulse_matrix_init;
    //! GEMM verify error limit (self-check or accuracy)
    float pulse_tolerance;
    //! minimum average high-minus-low power, watts. 0 disables the check
    float pulse_min_power_delta_w;
    //! sampling rate for power readings (ms)
    uint64_t pulse_sample_interval;
    //! kernel calls between health checks
    int pulse_workload_iterations;
    //! stop immediately on first error
    bool pulse_halt_on_error;
    //! bound on a host rendezvous, a device barrier, and one GEMM wait (ms)
    int pulse_gpu_sync_wait;
    //! how long each selected pattern runs. 0 means split duration across patterns
    uint64_t pulse_pattern_duration;
    //! true when the conf set pattern_duration explicitly
    bool pulse_pattern_duration_set;
    //! selected built-in pattern indices. Empty when pulse_custom is set.
    std::vector<int> pulse_patterns;
    //! run the custom waveform from pulse_rate and high_phase_ratio
    bool pulse_custom;
    //! fail the run if the uncorrectable ECC count rises
    bool pulse_ecc_check;
    //! compute verification mode: "crc", "diff", "both", or "none"
    std::string pulse_verify_mode;
    //! hot calls for BLAS warmup
    uint64_t pulse_hot_calls;
    //! blas backend source library
    std::string pulse_blas_source;
    //! gemm compute type
    std::string pulse_compute_type;
    //! fail high phase if junction/edge temp (C) exceeds this; 0 disables check
    float pulse_max_temp_c;

    friend class PulseWorker;

    std::map<int, amdsmi_processor_handle> hip_to_smi_idxs;
    void hip_to_smi_indices();
    bool get_all_pulse_config_keys(void);
    bool parse_patterns(const std::string& spec);
    int get_num_amd_gpu_devices(void);
    int get_all_selected_gpus(void);
    bool do_pulse_test(std::map<int, uint16_t> pulse_gpus_device_index,
        std::vector<mcm_type_t>& mcm_type);
};

#endif  // PULSE_SO_INCLUDE_ACTION_H_
