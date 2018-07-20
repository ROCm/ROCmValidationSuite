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
#ifndef GST_SO_INCLUDE_ACTION_H_
#define GST_SO_INCLUDE_ACTION_H_

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

#include "rvsactionbase.h"

using std::vector;
using std::string;
using std::map;

/**
 * @class action
 * @ingroup GST
 *
 * @brief GST action implementation class
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

    //! stress test ramp duration
    uint64_t gst_ramp_interval;
    //! time interval at which the module reports the average GFlops
    uint64_t gst_log_interval;
    //! maximum allowed number of target_stress violations
    int gst_max_violations;
    //! specifies whether to copy the matrix to the GPU for each SGEMM operation
    bool gst_copy_matrix;
    //! target stress (in GFlops) that the GPU will try to achieve
    float gst_target_stress;
    //! GFlops tolerance (how much the GFlops can fluctuare after
    //! the ramp period for the test to succeed)
    float gst_tolerance;
    //! matrix size for SGEMM
    uint64_t gst_matrix_size;

    //! TRUE if device config key is "all
    bool device_all_selected;
    //! TRUE if deviceid filtering was enabled
    bool device_id_filtering;
    //! GPU device type config key value
    uint16_t deviceid;

    // configuration properties getters


    // GST specific config keys
    void property_get_gst_ramp_interval(int *error);
    void property_get_gst_log_interval(int *error);
    void property_get_gst_max_violations(int *error);
    void property_get_gst_copy_matrix(int *error);

    void property_get_gst_target_stress(int *error);
    void property_get_gst_tolerance(int *error);
    void property_get_gst_matrix_size(int *error);

    bool get_all_gst_config_keys(void);
    bool get_all_common_config_keys(void);

    bool do_gpu_stress_test(map<int, uint16_t> gst_gpus_device_index);
};

#endif  // GST_SO_INCLUDE_ACTION_H_
