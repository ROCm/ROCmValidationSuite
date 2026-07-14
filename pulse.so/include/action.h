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

#ifdef __cplusplus
extern "C" {
#endif
#include <pci/pci.h>
#ifdef __cplusplus
}
#endif

#include <vector>
#include <string>
#include <map>

#include "include/rvsactionbase.h"
#include "amd_smi/amdsmi.h"

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
    //! matrix size for GEMM K dimension
    uint64_t pulse_matrix_size;
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
    //! power tolerance percentage
    float pulse_tolerance;
    //! sampling rate for power readings (ms)
    uint64_t pulse_sample_interval;
    //! kernel calls between health checks
    int pulse_workload_iterations;
    //! stop immediately on first error
    bool pulse_halt_on_error;
    //! cross-GPU sync timeout (ms)
    int pulse_gpu_sync_wait;
    //! compute verification mode: "crc" or "diff"
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
    int get_num_amd_gpu_devices(void);
    int get_all_selected_gpus(void);
    bool do_pulse_test(std::map<int, uint16_t> pulse_gpus_device_index,
        std::vector<mcm_type_t>& mcm_type);
};

#endif  // PULSE_SO_INCLUDE_ACTION_H_
