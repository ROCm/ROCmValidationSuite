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

class action: public rvs::actionbase {
 public:
    action();
    virtual ~action();

    virtual int run(void);

 private:
    vector<string> device_prop_gpu_id_list;  // the list of all gpu_id
                                             // in the <device> property
    string action_name;
    bool gst_runs_parallel;
    unsigned int gst_run_count;
    unsigned long gst_run_wait_ms;
    unsigned long gst_run_duration_ms;
    unsigned long gst_ramp_interval;
    unsigned long gst_log_interval;
    int gst_max_violations;
    bool gst_copy_matrix;
    float gst_target_stress;
    float gst_tolerance;
    

    // configuration properties getters
    // gets the device property value (list of gpu_id)
    // from the module's properties collection
    bool property_get_device(int *error);
    void property_get_action_name(void);  
    int property_get_deviceid(int *error); 
    void property_get_run_parallel(void);
    void property_get_run_count(void);
    void property_get_run_wait(void);
    void property_get_run_duration(void);
    void property_get_gst_ramp_interval(void);
    void property_get_gst_log_interval(void);
    void property_get_gst_max_violations(void);
    void property_get_gst_copy_matrix(void);
    void property_get_gst_target_stress(int *error);
    void property_get_gst_tolerance(void);
    
    void log_module_error(const string &error);
    void do_gpu_stress_test(map<int, uint16_t> gst_gpus_device_index);
 protected:
};

#endif  // GST_SO_INCLUDE_ACTION_H_
