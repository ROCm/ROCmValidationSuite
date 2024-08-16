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
#include <hiprand/hiprand.h>

#include <hipblaslt/hipblaslt.h>

#define RVS_BLAS_HIP_DATATYPE_INVALID static_cast<hipDataType>(0XFFFF)
#define RVS_BLAS_HIPBLAS_COMPUTETYPE_INVALID static_cast<hipblasComputeType_t>(0XFFFF)

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
       int transa, int transb, float alpha, float beta,
       int lda, int ldb, int ldc, int ldd,
       std::string _ops_type, std::string _data_type, std::string _gemm_mode,
       int _batch_count, uint64_t stride_a, uint64_t stride_b, uint64_t stride_c, uint64_t stride_d,
       std::string _blas_source, std::string _compute_type);
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

      if(gemm_mode == "strided_batched") {
        return (2.0 * m * n * k * batch_size) / 1e9;
      }
      else {
        return (2.0 * m * n * k) / 1e9;
      }
    }

    double get_time_us(void);
    //! returns TRUE if an error occured
    bool error(void) { return is_error; }
    void generate_random_matrix_data(void);
    bool copy_data_to_gpu(void);
    bool run_blas_gemm(void);
    bool is_gemm_op_complete(void);
    bool validate_gemm(bool self_check, bool accu_check, double &self_error, double &accu_error);
    void set_gemm_error(uint64_t _error_freq, uint64_t _error_count);

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
    //! amount of memory to allocate for the matrix a
    size_t size_a;
    //! amount of memory to allocate for the matrix b
    size_t size_b;
    //! amount of memory to allocate for the matrix c
    size_t size_c;
    //! amount of memory to allocate for the matrix d
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
    //! pointer to host (CPU) gemm output (host memory)
    void* hout;
    //! pointer to device (GPU) gemm output (host memory)
    void* hdout;

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
    //! random number generator
    hiprandGenerator_t hiprand_generator;

    //! rocBlas related handle
    rocblas_handle blas_handle;
    //! TRUE is rocBlas handle was successfully initialized
    bool is_handle_init;
    //! rocBlas guard (prevents executing blass_gemm when there are mem errors)
    bool is_error;

    // error injection frequency (number of gemm calls per error injection)
    uint64_t error_freq;
    // number of errors injected in gemm output
    uint64_t error_count;
    // gemm check counter
    uint64_t check_count;

    //! gemm mode : basic (single), batched or strided batched
    std::string gemm_mode;

    //! Matrix batch count
    int batch_size;

    //! Stride from the start of matrix A(i)
    //! to next matrix A(i+1) in the strided batch
    uint64_t stride_a;
    //! Stride from the start of matrix B(i)
    //! to next matrix B(i+1) in the strided batch
    uint64_t stride_b;
    //! Stride from the start of matrix C(i)
    //! to next matrix C(i+1) in the strided batch
    uint64_t stride_c;
    //! Stride from the start of matrix D(i)
    //! to next matrix D(i+1) in the strided batch
    uint64_t stride_d;

    //! blas backend source library - rocblas,hipblaslt
    std::string blas_source;

    //! gemm compute type
    std::string compute_type;

    //! hipblaslt related handle
    hipblasLtHandle_t hbl_handle;

    //! Matrix Layouts for matrix A
    hipblasLtMatrixLayout_t hbl_layout_a;
    //! Matrix Layouts for matrix B
    hipblasLtMatrixLayout_t hbl_layout_b;
    //! Matrix Layouts for matrix C
    hipblasLtMatrixLayout_t hbl_layout_c;
    //! Matrix Layouts for matrix D
    hipblasLtMatrixLayout_t hbl_layout_d;

    //! hipblaslt matrix data-type
    hipDataType hbl_datatype;

    //! hipblaslt compute-type
    hipblasComputeType_t hbl_computetype;

    //! Create hipblaslt matrix multiply descriptor
    hipblasLtMatmulDesc_t hbl_matmul;

    //! Transpose matrix A
    hipblasOperation_t hbl_trans_a;
    //! Transpose matrix B
    hipblasOperation_t hbl_trans_b;

    //! Workspace buffer for matrix multiplication
    void* hbl_workspace;

    //! hipblaslt matrix a leading dimension
    int64_t hbl_lda_offset;
    //! hipblaslt matrix b leading dimension
    int64_t hbl_ldb_offset;
    //! hipblaslt matrix c leading dimension
    int64_t hbl_ldc_offset;
    //! hipblaslt matrix d leading dimension
    int64_t hbl_ldd_offset;

    bool init_gpu_device(void);
    bool allocate_gpu_matrix_mem(void);
    void release_gpu_matrix_mem(void);

    bool allocate_host_matrix_mem(void);
    void release_host_matrix_mem(void);
    float fast_pseudo_rand(uint64_t *nextr, size_t i);

    template <typename T>
      bool check_result_consistency(void * dout, size_t size, double &error);

    template <typename T>
      bool check_result_accuracy(void * dout, size_t size, double &error);

    hipDataType datatype_to_hip_datatype(const std::string& datatype)
    {
      return
        (datatype == "i8_r")   ? HIP_R_8I  :
        (datatype == "fp8_r")  ? HIP_R_8F_E4M3_FNUZ  :
        (datatype == "bf16_r") ? HIP_R_16BF :
        (datatype == "fp16_r") ? HIP_R_16F  :
        (datatype == "fp32_r") ? HIP_R_32F  :
        (datatype == "fp64_r") ? HIP_R_64F  :
        RVS_BLAS_HIP_DATATYPE_INVALID;
    }


    hipblasComputeType_t computetype_to_hipblas_computetype(const std::string& computetype)
    {
      return
        computetype == "fp32_r" ? HIPBLAS_COMPUTE_32F  :
        computetype == "xf32_r" ? HIPBLAS_COMPUTE_32F_FAST_TF32 :
        computetype == "fp64_r" ? HIPBLAS_COMPUTE_64F :
        computetype == "i32_r"  ? HIPBLAS_COMPUTE_32I :
        RVS_BLAS_HIPBLAS_COMPUTETYPE_INVALID;
    }

};

#endif  // INCLUDE_RVS_BLAS_H_
