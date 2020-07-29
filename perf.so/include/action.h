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
#ifndef PERF_SO_INCLUDE_ACTION_H_
#define PERF_SO_INCLUDE_ACTION_H_

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
 * @class perf_action
 * @ingroup PERF
 *
 * @brief PERF action implementation class
 *
 * Derives from rvs::actionbase and implements actual action functionality
 * in its run() method.
 *
 */
class perf_action: public rvs::actionbase {
 public:
    perf_action();
    virtual ~perf_action();

    virtual int run(void);

    std::string perf_ops_type;

 protected:
    //! TRUE if JSON output is required
    bool bjson;

    //! stress test ramp duration
    uint64_t perf_ramp_interval;
    //! maximum allowed number of target_stress violations
    int perf_max_violations;
    //! specifies whether to copy the matrices to the GPU before each
    //! SGEMM operation
    bool perf_copy_matrix;
    //! target stress (in GFlops) that the GPU will try to achieve
    float perf_target_stress;
    //! GFlops tolerance (how much the GFlops can fluctuare after
    //! the ramp period for the test to succeed)
    float perf_tolerance;
    
    //Alpha and beta value
    float      perf_alpha_val;
    float      perf_beta_val;
    
    //! matrix size for SGEMM
    uint64_t perf_matrix_size_a;
    uint64_t perf_matrix_size_b;
    uint64_t perf_matrix_size_c;

    //Parameter to heat up
    uint64_t perf_hot_calls;

    //Tranpose set to none or enabled
    int      perf_trans_a;
    int      perf_trans_b;

    //Leading offset values
    int      perf_lda_offset;
    int      perf_ldb_offset;
    int      perf_ldc_offset;

    // PERF specific config keys
//     void property_get_perf_target_stress(int *error);
//     void property_get_perf_tolerance(int *error);

    bool get_all_perf_config_keys(void);
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
    bool do_gpu_stress_test(map<int, uint16_t> perf_gpus_device_index);
};

#endif  // PERF_SO_INCLUDE_ACTION_H_
