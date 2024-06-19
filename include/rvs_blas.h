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
#ifndef INCLUDE_RVS_BLAS_H_
#define INCLUDE_RVS_BLAS_H_

#define __HIP_PLATFORM_HCC__

/* To enable rocblas beta functions in rocblas.h */
#define ROCBLAS_BETA_FEATURES_API 1

/*Based on Version of ROCBLAS use correct include header*/
#if(defined(RVS_ROCBLAS_VERSION_FLAT) && ((RVS_ROCBLAS_VERSION_FLAT) >= 2044000))
  #include <rocblas/rocblas.h>
#else
  #include <rocblas.h>
#endif

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include <sys/time.h>

typedef void (*rvsBlasCallback_t) (bool status, void *userData);

/**
 * @class rvs_blas
 * @ingroup GST
 *
 * @brief RVS blas implementation for gemm operations
 *
 */
class rvs_blas {
 public:
   rvs_blas(int _gpu_device_index, int _m, int _n, int _k, std::string _matrix_init,
       int transa, int transb, float aplha, float beta,
       rocblas_int lda, rocblas_int ldb, rocblas_int ldc, rocblas_int ldd,
       std::string _ops_type, std::string _data_type);
    rvs_blas() = delete;
    rvs_blas(const rvs_blas&) = delete;
    rvs_blas& operator=(const rvs_blas&) = delete;
    rvs_blas(rvs_blas&&) = delete;
    rvs_blas& operator=(rvs_blas&&) = delete;

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

    //! returns theoretical GFLOPs for gemm  
    double gemm_gflop_count(void) {
        return (2.0 * m * n * k) / 1e9;
    }

    double get_time_us(void);
    //! returns TRUE if an error occured
    bool error(void) { return is_error; }
    void generate_random_matrix_data(void);
    bool copy_data_to_gpu(std::string);
    bool run_blass_gemm(std::string);
    bool is_gemm_op_complete(void);
    bool validate_gemm(void);

    template <typename T>
      bool check_result_consistency(void * dout, uint64_t size);

    template <typename T>
      bool check_result_accuracy(void * dout, uint64_t size);

    bool set_callback(rvsBlasCallback_t callback, void *user_data);

    static void hip_stream_callback (hipStream_t stream, hipError_t status, void *user_data);

    rvsBlasCallback_t callback;
    void * user_data;

 protected:
    //! GPU device index
    int gpu_device_index;
    //! Type of operation
    std::string ops_type;
    //! Type of data
    std::string data_type;
    //! matrix size m
    rocblas_int m;
    //! matrix size n
    rocblas_int n;
    //! matrix size k
    rocblas_int k;
    //! amount of memory to allocate for the matrix
    size_t size_a;
    //! amount of memory to allocate for the matrix
    size_t size_b;
    //! amount of memory to allocate for the matrix
    size_t size_c;
    //! amount of memory to allocate for the matrix
    size_t size_d;
    //! matrix initialization
    std::string matrix_init;
    //! Transpose matrix A
    rocblas_operation transa;
    //! Transpose matrix B
    rocblas_operation transb;

    //SGEMM DECLARATION
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

    //DGEMM Declaration
    //! pointer to device (GPU) memory
    double *ddbla;
    //! pointer to device (GPU) memory
    double *ddblb;
    //! pointer to device (GPU) memory
    double *ddblc;
    //! pointer to host memory
    double *hdbla;
    //! pointer to host memory
    double *hdblb;
    //! pointer to host memory
    double *hdblc;

    //Data type Declaration
    //! pointer to device (GPU) memory
    void *dda;
    //! pointer to device (GPU) memory
    void *ddb;
    //! pointer to device (GPU) memory
    void *ddc;
    //! pointer to device (GPU) memory
    void *ddd;
    //! pointer to host memory
    void *hda;
    //! pointer to host memory
    void *hdb;
    //! pointer to host memory
    void *hdc;

    //! pointer to current gemm output (host memory)
    void *hco;
    //! pointer to previous gemm output (host memory)
    void *hpo;

    //!GST Aplha Val 
    float blas_alpha_val;
    //! GST Beta Val
    float blas_beta_val;

    //!Blas offsets
    rocblas_int blas_lda_offset;
    //!Blas offsets
    rocblas_int blas_ldb_offset;
    //!Blas offsets
    rocblas_int blas_ldc_offset;
    //!Blas offsets
    rocblas_int blas_ldd_offset;

    //HGEMM Declaration
    //! pointer to device (GPU) memory
    rocblas_half *dhlfa;
    //! pointer to device (GPU) memory
    rocblas_half *dhlfb;
    //! pointer to device (GPU) memory
    rocblas_half *dhlfc;
    //! pointer to device (GPU) memory
    rocblas_half *dhlfd;

    //! pointer to host memory
    rocblas_half *hhlfa;
    //! pointer to host memory
    rocblas_half *hhlfb;
    //! pointer to host memory
    rocblas_half *hhlfc;

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
    float fast_pseudo_rand(uint64_t *nextr, size_t i);

};

#endif  // INCLUDE_RVS_BLAS_H_
