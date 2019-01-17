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
#ifndef IET_SO_INCLUDE_ACTION_H_
#define IET_SO_INCLUDE_ACTION_H_

#ifdef __cplusplus
extern "C" {
#endif
#include <pci/pci.h>
#ifdef __cplusplus
}
#endif

#include <vector>
#include <string>
#include <utility>
#include <memory>


#include "include/rvsactionbase.h"
#include "rocm_smi/rocm_smi.h"

using std::vector;
using std::string;

//! structure containing GPU identification related data
struct gpu_hwmon_info {
    //! GPU device index (0..n) as reported by HIP API
    int hip_gpu_deviceid;
    //! real GPU ID (e.g.: 53645) as exported by kfd
    uint16_t gpu_id;
    //! BDF id
    uint32_t bdf_id;
};

/**
 * @class iet_action
 * @ingroup IET
 *
 * @brief IET action implementation class
 *
 * Derives from rvs::actionbase and implements actual action functionality
 * in its run() method.
 *
 */
class iet_action: public rvs::actionbase {
 public:
    iet_action();
    virtual ~iet_action();

    virtual int run(void);

 protected:
    //! TRUE if JSON output is required
    bool bjson;

    //! target power level for the test
    float iet_target_power;
    //! IET test ramp duration
    uint64_t iet_ramp_interval;
    //! power tolerance (how much the target_power can fluctuare after
    //! the ramp period for the test to succeed)
    float iet_tolerance;
    //! maximum allowed number of target_power violations
    int iet_max_violations;
    //! sampling rate for the target_power
    uint64_t iet_sample_interval;
    //! matrix size for SGEMM
    uint64_t iet_matrix_size;

    //! list of GPUs (along with some identification data) which are
    //! selected for EDPp test
    std::vector<gpu_hwmon_info> edpp_gpus;


    bool get_all_iet_config_keys(void);
    /**
    * @brief reads all common configuration keys from
    * the module's properties collection
    * @return true if no fatal error occured, false otherwise
    */
    bool get_all_common_config_keys(void);
    bool add_gpu_to_edpp_list(uint16_t dev_location_id, int32_t gpu_id,
                              int hip_num_gpu_devices);

/**
 * @brief gets the number of ROCm compatible AMD GPUs
 * @return run number of GPUs
 */
    int get_num_amd_gpu_devices(void);
/**
 * @brief gets all selected GPUs and starts the worker threads
 * @return run result
 */    
    int get_all_selected_gpus(void);

    bool do_edp_test(void);
};

#endif  // IET_SO_INCLUDE_ACTION_H_
