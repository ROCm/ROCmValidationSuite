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

#include "include/rvsactionbase.h"

using std::vector;
using std::string;
using std::map;

/**
 * @class gst_action
 * @ingroup GST
 *
 * @brief GST action implementation class
 *
 * Derives from rvs::actionbase and implements actual action functionality
 * in its run() method.
 *
 */
class gst_action: public rvs::actionbase {
  public:
    gst_action();
    virtual ~gst_action();

    virtual int run(void);
    static void cleanup_logs();

  protected:
    //! TRUE if JSON output is required
    bool bjson;

    //! stress test ramp duration
    uint64_t gst_ramp_interval;
    //! maximum allowed number of target_stress violations
    int gst_max_violations;
    //! specifies whether to copy the matrices
    //! to the GPU before each GEMM operation
    bool gst_copy_matrix;
    //! target stress (in GFlops) that the GPU will try to achieve
    float gst_target_stress;
    //! GFlops tolerance (how much the GFlops can fluctuare after
    //! the ramp period for the test to succeed)
    float gst_tolerance;

    //! Alpha and beta value
    float gst_alpha_val;
    float gst_beta_val;

    //! matrix sizes for GEMM operation
    uint64_t gst_matrix_size_a;
    uint64_t gst_matrix_size_b;
    uint64_t gst_matrix_size_c;

    //! matrix initialization method :
    //! default, random integer or trignometric float
    std::string gst_matrix_init;

    //Parameter to heat up
    uint64_t gst_hot_calls;

    //Tranpose set to none or enabled
    int      gst_trans_a;
    int      gst_trans_b;

    //Leading offset values
    int      gst_lda_offset;
    int      gst_ldb_offset;
    int      gst_ldc_offset;
    int      gst_ldd_offset;

    // type of gemm operation
    std::string gst_ops_type;

    // gemm data type
    std::string gst_data_type;

    // gemm output self-check
    bool gst_self_check;

    // gemm output accuracy-check
    bool gst_accu_check;

    // Inject error in gemm output
    // Note : This is just for testing purpose. Shouldn't be enabled otherwise.
    bool     gst_error_inject;
    // error injection frequency (number of gemm calls per error injection)
    uint64_t gst_error_freq;
    // number of errors injected in gemm output
    uint64_t gst_error_count;

    // gemm mode : basic (single), batched or strided batched
    std::string gst_gemm_mode;

    // Matrix batch count
    int gst_batch_size;

    // Stride from the start of matrix a(i)
    // to next matrix a(i+1) in the strided batch
    uint64_t gst_stride_a;

    // Stride from the start of matrix b(i)
    // to next matrix b(i+1) in the strided batch
    uint64_t gst_stride_b;

    // Stride from the start of matrix c(i)
    // to next matrix c(i+1) in the strided batch
    uint64_t gst_stride_c;

    // Stride from the start of matrix d(i)
    // to next matrix d(i+1) in the strided batch
    uint64_t gst_stride_d;

    friend class GSTWorker;

    bool get_all_gst_config_keys(void);
    void json_add_primary_fields();

    /**
     * @brief gets the number of ROCm compatible AMD GPUs
     * @return run number of GPUs
     */
    int get_num_amd_gpu_devices(void);
    int get_all_selected_gpus(void);
    bool do_gpu_stress_test(map<int, uint16_t> gst_gpus_device_index);
};

#endif  // GST_SO_INCLUDE_ACTION_H_
