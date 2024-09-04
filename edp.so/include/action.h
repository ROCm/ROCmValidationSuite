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
#ifndef EDP_SO_INCLUDE_ACTION_H_
#define EDP_SO_INCLUDE_ACTION_H_

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

using std::vector;
using std::string;
using std::map;

/**
 * @class edp_action
 * @ingroup EDP
 *
 * @brief EDP action implementation class
 *
 * Derives from rvs::actionbase and implements actual action functionality
 * in its run() method.
 *
 */
class edp_action: public rvs::actionbase {
 public:
    edp_action();
    virtual ~edp_action();

    virtual int run(void);

    std::string edp_ops_type;

 protected:

    //! stress test ramp duration
    uint64_t edp_ramp_interval;
    //! maximum allowed number of target_stress violations
    int edp_max_violations;
    //! specifies whether to copy the matrices to the GPU before each
    //! SGEMM operation
    bool edp_copy_matrix;
    //! target stress (in GFlops) that the GPU will try to achieve
    float edp_target_stress;
    //! GFlops tolerance (how much the GFlops can fluctuare after
    //! the ramp period for the test to succeed)
    float edp_tolerance;
    
    //Alpha and beta value
    float      edp_alpha_val;
    float      edp_beta_val;
    
    //! matrix size for SGEMM
    uint64_t edp_matrix_size_a;
    uint64_t edp_matrix_size_b;
    uint64_t edp_matrix_size_c;

    //Parameter to heat up
    uint64_t edp_hot_calls;

    //Tranpose set to none or enabled
    int      edp_trans_a;
    int      edp_trans_b;

    //Leading offset values
    int      edp_lda_offset;
    int      edp_ldb_offset;
    int      edp_ldc_offset;

    uint64_t edp_wave_iterations;
    uint64_t edp_halt_timer;
    uint64_t edp_restart_wave_timer;
    bool edp_broadast_wave;

    // EDP specific config keys
//     void property_get_edp_target_stress(int *error);
//     void property_get_edp_tolerance(int *error);

    bool get_all_edp_config_keys(void);
  /**
  * @brief reads all common configuration keys from
  * the module's properties collection
  * @return true if no fatal error occured, false otherwise
  */
    bool get_all_common_config_keys(void);

  /**
  * @brief gets the number of ROCm compatible AMD GPUs
  * @return run number of GPUs
  */
    int get_num_amd_gpu_devices(void);
    int get_all_selected_gpus(void);
    bool do_gpu_stress_test(map<int, uint16_t> edp_gpus_device_index);
    void StartPeakPowerThread(unsigned int);
};

#endif  // EDP_SO_INCLUDE_ACTION_H_
