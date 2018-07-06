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
#ifndef GST_SO_INCLUDE_RVS_BLAS_H_
#define GST_SO_INCLUDE_RVS_BLAS_H_

#define __HIP_PLATFORM_HCC__

#include "rocblas.h"
#include "hip_runtime.h"
#include "hip_runtime_api.h"

class rvs_blas {
 public:
    rvs_blas(int , rocblas_int, rocblas_int, rocblas_int);
    ~rvs_blas();

    // gpu_device_index, m, n and k getters
    int get_gpu_device_index(void) { return gpu_device_index; }
    rocblas_int get_m(void) { return m; }
    rocblas_int get_n(void) { return n; }
    rocblas_int get_k(void) { return k; }

    double gemm_gflop_count(void) {
        return static_cast<double>(2.0 * m * n * k) / 1e9;
    }

    bool error(void) { return is_error; }
    void generate_random_matrix_data(void);
    bool copy_data_to_gpu(void);
    bool run_blass_gemm(void);
    bool is_gemm_op_complete(void);

 private:
    // data members
    int gpu_device_index;  // the GPU device that will run the S/D GEMM
    rocblas_int m, n, k;   // data matrixes size
    rocblas_int size_a, size_b, size_c;   // data matrixes total mem size
    float *da, *db, *dc;  // pointers to device (GPU) memory
    float *ha, *hb, *hc;  // pointers to host memory

    // HIP API stream - used to query for GEMM completion
    // needed because rocBlass GEMM operates async
    hipStream_t hip_stream;
    rocblas_handle blas_handle;

    bool is_handle_init;

    // rocBlas guard (prevents executing blass_gemm when there are mem errors)
    bool is_error;

    bool init_gpu_device(void);
    bool allocate_gpu_matrix_mem(void);  // allocate memory on the GPU device
    void release_gpu_matrix_mem(void);  // release GPU mem

    bool alocate_host_matrix_mem(void);  // alocate host memory for matrixes
    void release_host_matrix_mem(void);  // generates random matrix data
};

#endif  // GST_SO_INCLUDE_RVS_BLAS_H_
