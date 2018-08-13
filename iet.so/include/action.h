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
#include <map>
#include <utility>

#include "rvsactionbase.h"

using std::vector;
using std::string;
using std::map;

/**
 * @class action
 * @ingroup IET
 *
 * @brief IET action implementation class
 *
 * Derives from rvs::actionbase and implements actual action functionality
 * in its run() method.
 *
 */
class action: public rvs::actionbase {
 public:
    action();
    virtual ~action();

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
    //! time interval at which the module reports the GPU's power
    uint64_t iet_log_interval;
    //! matrix size for SGEMM
    uint64_t iet_matrix_size;

    //! TRUE if device config key is "all
    bool device_all_selected;
    //! TRUE if deviceid filtering was enabled
    bool device_id_filtering;
    //! GPU device type config key value
    uint16_t deviceid;

    // configuration properties getters

    // IET specific config keys
    void property_get_iet_target_power(int *error);
    void property_get_iet_ramp_interval(int *error);
    void property_get_iet_tolerance(int *error);
    void property_get_iet_max_violations(int *error);
    void property_get_iet_sample_interval(int *error);
    void property_get_iet_log_interval(int *error);
    void property_get_iet_matrix_size(int *error);

    bool get_all_iet_config_keys(void);
    bool get_all_common_config_keys(void);
    const std::string get_irq(const std::string dev_path);
    int get_num_amd_gpu_devices(void);
    int get_all_selected_gpus(void);

    bool do_edp_test(std::map<int,
                     std::pair<uint16_t, string>> iet_gpus_device_index);
};

#endif  // IET_SO_INCLUDE_ACTION_H_
