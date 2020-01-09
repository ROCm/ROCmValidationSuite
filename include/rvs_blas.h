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
#ifndef INCLUDE_RVS_BLAS_H_
#define INCLUDE_RVS_BLAS_H_

#define __HIP_PLATFORM_HCC__

#include "rocblas.h"
#include "include/hip/hip_runtime.h"
#include "include/hip/hip_runtime_api.h"

/**
 * @class rvs_blas
 * @ingroup GST
 *
 * @brief implements the SGEMM logic
 *
 */
class rvs_blas {
 public:
    rvs_blas(int _gpu_device_index, int _m, int _n, int _k);
    ~rvs_blas();

    //! returns the GPU index
    int get_gpu_device_index(void) { return gpu_device_index; }
    //! returns m (matrix size)
    rocblas_int get_m(void) { return m; }
    //! returns n (matrix size)
    rocblas_int get_n(void) { return n; }
    //! returns k (matrix size)
    rocblas_int get_k(void) { return k; }

    //! computes the number of bytes which are copied to
    //! the GPU for one SGEMM operation
    uint64_t get_bytes_copied_per_op(void) {
        return sizeof(float) * (size_a + size_b + size_c);
    }
    //! computes the gflop for a SGEMM operation
    double gemm_gflop_count(void) {
        return static_cast<double>(2.0 * m * n * k) / 1e9;
    }

    //! returns TRUE if an error occured
    bool error(void) { return is_error; }
    void generate_random_matrix_data(void);
    void generate_random_dbl_matrix_data(void);
    void generate_random_half_matrix_data(void);
    

    bool copy_data_to_gpu(void);
    bool copy_dbl_data_to_gpu(void);
    bool copy_hlf_data_to_gpu(void);
    bool run_blass_gemm(void);
    bool run_blass_dgemm(void);
    bool run_blass_hgemm(void);
    bool is_gemm_op_complete(void);

 protected:
    //! GPU device index
    int gpu_device_index;
    //! matrix size m
    rocblas_int m;
    //! matrix size n
    rocblas_int n;
    //! matrix size k
    rocblas_int k;

    //! amount of memory to allocate for the matrix
    rocblas_int size_a;
    //! amount of memory to allocate for the matrix
    rocblas_int size_b;
    //! amount of memory to allocate for the matrix
    rocblas_int size_c;

    //DGEMM
    //Device Memory
    //! pointer to device (GPU) memory
    double *ddbla;
    //! pointer to device (GPU) memory
    double *ddblb;
    //! pointer to device (GPU) memory
    double *ddblc;
    //Host Memory
    //! pointer to host memory
    double *hdbla;
    //! pointer to host memory
    double *hdblb;
    //! pointer to host memory
    double *hdblc;

    //HGEMM
    //! pointer to host memory
    rocblas_half *hlfa;
    //! pointer to host memory
    rocblas_half *hlfb;
    //! pointer to host memory
    rocblas_half *hlfc;
    //! pointer to device (GPU) memory
    rocblas_half *dhlfa;
    //! pointer to device (GPU) memory
    rocblas_half *dhlfb;
    //! pointer to device (GPU) memory
    rocblas_half *dhlfc;
 
    //SGEMM 
    //! pointer to device (GPU) memory
    float *da;
    //! pointer to device (GPU) memory
    float *db;
    //! pointer to device (GPU) memory
    float *dc;
    //! pointer to host memory
    float *ha;
    //! pointer to host memory
    float *hb;
    //! pointer to host memory
    float *hc;

    //! HIP API stream - used to query for GEMM completion
    hipStream_t hip_stream;
    //! rocBlas related handle
    rocblas_handle blas_handle;
    //! TRUE is rocBlas handle was successfully initialized
    bool is_handle_init;
    //! rocBlas guard (prevents executing blass_gemm when there are mem errors)
    bool is_error;

    bool init_gpu_device(void);
    bool allocate_gpu_matrix_mem(void);
    void release_gpu_matrix_mem(void);

    bool alocate_host_matrix_mem(void);
    void release_host_matrix_mem(void);

    double genDoubleRand();
    float genFloatRand();
#if __clang__ && (__STDC_VERSION__ >= 201112L || __cplusplus >= 201103L)
    rocblas_half genHalfRand();
#else
    uint16_t genHalfRand();
#endif
};

#endif  // INCLUDE_RVS_BLAS_H_
