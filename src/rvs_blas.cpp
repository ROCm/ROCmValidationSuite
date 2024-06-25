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
#include "include/rvs_blas.h"

#include <time.h>
#include <iostream>
#include <cmath>
#include <random>
#include <thread>
#include <iomanip>

/* ============================================================================================ */
// Random number generator
using rvsblas_rng_t = std::mt19937;

// Random number generator
rvsblas_rng_t rvsblas_seed(69069); // A fixed seed to start at

// This records the main thread ID at startup
std::thread::id rvsblas_main_thread_id = std::this_thread::get_id();

// For the main thread, we use g_rocblas_seed; for other threads, we start with a different seed but
// deterministically based on the thread id's hash function.
inline rvsblas_rng_t get_seed()
{
  auto tid = std::this_thread::get_id();
  return tid == rvsblas_main_thread_id ? rvsblas_seed
    : rvsblas_rng_t(std::hash<std::thread::id>{}(tid));
}

// For the main thread, we use g_rvsblas_seed; for other threads, we start with a different seed but
// deterministically based on the thread id's hash function.
thread_local rvsblas_rng_t rvsblas_t_rng = get_seed();

thread_local int rvsblas_t_rand_idx;

// length to allow use as bitmask to wraparound
#define RANDLEN 1024
#define RANDWIN 256
#define RANDBUF RANDLEN + RANDWIN
static thread_local int    rvsblas_t_rand_init = 0;
static thread_local float  rvsblas_t_rand_f_array[RANDBUF];
static thread_local double rvsblas_t_rand_d_array[RANDBUF];

/* ============================================================================================ */

#define RANDOM_CT               320000
#define RANDOM_DIV_CT           0.1234

/**
 * @brief class constructor
 * @param _gpu_device_index the gpu that will run the GEMM
 * @param _m matrix A rows
 * @param _n matrix B cols
 * @param _k matrix A/B cols/rows respectively
 * @param transA matrix A transpose operation type
 * @param transB matrix B transpose operation type
 * @param alpha scalar for matrix A*B
 * @param beta scalar for matrix C
 * @param lda leading dimension for matrix A
 * @param ldb leading dimension for matrix B
 * @param ldc leading dimension for matrix C
 * @param _ops_type type of BLAS operation to test with
 */
rvs_blas::rvs_blas(int _gpu_device_index, int _m, int _n, int _k, std::string _matrix_init, int transA, int transB,
    float alpha , float beta, rocblas_int lda, rocblas_int ldb, rocblas_int ldc, rocblas_int ldd,
    std::string _ops_type, std::string _data_type)
  : gpu_device_index(_gpu_device_index)
  , ops_type(_ops_type)
  , data_type(_data_type)
  , m(_m), n(_n), k(_k)
  , matrix_init (_matrix_init)
  , size_a(0), size_b(0), size_c(0), size_d(0)
  , da(nullptr), db(nullptr), dc(nullptr)
  , ha(nullptr), hb(nullptr), hc(nullptr)
  , ddbla(nullptr), ddblb(nullptr), ddblc(nullptr)
  , hdbla(nullptr), hdblb(nullptr), hdblc(nullptr)
  , dhlfa(nullptr), dhlfb(nullptr), dhlfc(nullptr), dhlfd(nullptr)
  , hhlfa(nullptr), hhlfb(nullptr), hhlfc(nullptr)
  , dda(nullptr), ddb(nullptr), ddc(nullptr), ddd(nullptr)
  , hda(nullptr), hdb(nullptr), hdc(nullptr)
  , hpo(nullptr), hco(nullptr)
  , hip_stream(nullptr)
  , blas_handle(nullptr)
  , is_handle_init(false)
  , is_error(false)
{

  if(transA == 0) {
    transa = rocblas_operation_none;
  }else{
    transa = rocblas_operation_transpose;
  }

  if(transB == 0) {
    transb = rocblas_operation_none;
  }else{
    transb = rocblas_operation_transpose;
  }

  // minimum leading dimensions
  rocblas_int min_lda = transA == rocblas_operation_none ? m : k;
  rocblas_int min_ldb = transB == rocblas_operation_none ? k : n;
  rocblas_int min_ldc = m;
  rocblas_int min_ldd = m;

  // setting actual leading dimensions
  blas_lda_offset = (lda < min_lda) ? min_lda : lda;
  blas_ldb_offset = (ldb < min_ldb) ? min_ldb : ldb;
  blas_ldc_offset = (ldc < min_ldc) ? min_ldc : ldc;
  blas_ldd_offset = (ldd < min_ldd) ? min_ldd : ldd;

  if(ops_type == "hgemm") {
    auto A_col = transA == rocblas_operation_none ? k : m;
    auto B_col = transB == rocblas_operation_none ? n : k;

    size_a = size_t(lda) * A_col;
    size_b = size_t(ldb) * B_col;
    size_c = size_t(ldc) * n;
    size_d = size_t(ldc) * n;
  } else {

    size_a = transA == rocblas_operation_none ? size_t(k) * blas_lda_offset : size_t(m) * blas_lda_offset;
    size_b = transB == rocblas_operation_none ? size_t(n) * blas_lda_offset : size_t(k) * blas_ldb_offset;

    size_c = size_t(n) * blas_ldc_offset;

    // gemm based on data type, size of output matrix d.
    if (!data_type.empty()) {
      size_d = size_t(n) * blas_ldd_offset;
    }
  }

  //setting alpha and beta val
  blas_alpha_val = alpha;
  blas_beta_val = beta;

  if (alocate_host_matrix_mem()) {
    if (!init_gpu_device())
      is_error = true;
  } else {
    is_error = true;
  }
}

/**
 * @brief class destructor
 */
rvs_blas::~rvs_blas() {
    release_host_matrix_mem();
    release_gpu_matrix_mem();
}

/**
 * @brief selects GPU device, allocates GPU memory, creates a rocBlas
 * handle and get a reference to the rocBlas's stream
 * @return true if everything went fine, otherwise false
 */
bool rvs_blas::init_gpu_device(void) {

  // select GPU device & allocate memory
  if (hipSetDevice(gpu_device_index) != hipSuccess) {
    // cannot select the given GPU device
    return false;
  }

  // rocblas initialize
  rocblas_initialize();

  if (!allocate_gpu_matrix_mem()) {
    std::cout << "\n allocate_gpu_matrix_mem() failed !!!" << "\n";
    return false;
  }

  if (hipStreamCreate(&hip_stream) != hipSuccess) {
    std::cout << "\n hipStreamCreate() failed !!!" << "\n";
    return false;
  }

  if (rocblas_create_handle(&blas_handle) != rocblas_status_success) {
    std::cout << "\n rocblas_create_handle() failed !!!" << "\n";
    return false;
  }

  if (rocblas_set_stream(blas_handle, hip_stream) != rocblas_status_success) {
    std::cout << "\n rocblas_set_stream() failed !!!" << "\n";
    return false;
  }

  is_handle_init = true;
  return true;
}

/**
 * @brief copy data matrix from host to gpu
 * @return true if everything went fine, otherwise false
 */
bool rvs_blas::copy_data_to_gpu(std::string ops_type) {

  if(ops_type == "sgemm") {

    if (da) {
      if (hipMemcpy(da, ha, sizeof(float) * size_a, hipMemcpyHostToDevice)
          != hipSuccess) {
        is_error = true;
        return false;
      }
    }

    if (db) {
      if (hipMemcpy(db, hb, sizeof(float) * size_b, hipMemcpyHostToDevice)
          != hipSuccess) {
        is_error = true;
        return false;
      }
    }

    if (dc) {
      if (hipMemcpy(dc, hc, sizeof(float) * size_c, hipMemcpyHostToDevice)
          != hipSuccess) {
        is_error = true;
        return false;
      }
    }
  }

  if(ops_type == "dgemm") {

    if (ddbla) {
      if (hipMemcpy(ddbla, hdbla, sizeof(double) * size_a, hipMemcpyHostToDevice)
          != hipSuccess) {
        is_error = true;
        return false;
      }
    }

    if (ddblb) {
      if (hipMemcpy(ddblb, hdblb, sizeof(double) * size_b, hipMemcpyHostToDevice)
          != hipSuccess) {
        is_error = true;
        return false;
      }
    }

    if (ddblc) {
      if (hipMemcpy(ddblc, hdblc, sizeof(double) * size_c, hipMemcpyHostToDevice)
          != hipSuccess) {
        is_error = true;
        return false;
      }
    }

  }

  if(ops_type == "hgemm") {

    if (dhlfa) {
      if (hipMemcpy(dhlfa, hhlfa, sizeof(rocblas_half) * size_a, hipMemcpyHostToDevice)
          != hipSuccess) {
        is_error = true;
        return false;
      }
    }

    if (dhlfb) {
      if (hipMemcpy(dhlfb, hhlfb, sizeof(rocblas_half) * size_b, hipMemcpyHostToDevice)
          != hipSuccess) {
        is_error = true;
        return false;
      }
    }

    if (dhlfc) {
      if (hipMemcpy(dhlfc, hhlfc, sizeof(rocblas_half) * size_c, hipMemcpyHostToDevice)
          != hipSuccess) {
        is_error = true;
        return false;
      }
    }
  }

  if(data_type == "fp8_r") {

    if (dda) {

      printf("sizeof(struct rocblas_f8) -> %d\n", sizeof(struct rocblas_f8));
      printf("sizeof(rocblas_f8) -> %d\n", sizeof(rocblas_f8));

      if (hipMemcpy(dda, hda, sizeof(struct rocblas_f8) * size_a, hipMemcpyHostToDevice)
          != hipSuccess) {
        is_error = true;
        return false;
      }
    }

    if (ddb) {
      if (hipMemcpy(ddb, hdb, sizeof(struct rocblas_f8) * size_b, hipMemcpyHostToDevice)
          != hipSuccess) {
        is_error = true;
        return false;
      }
    }

    if (ddc) {
      if (hipMemcpy(ddc, hdc, sizeof(struct rocblas_f8) * size_c, hipMemcpyHostToDevice)
          != hipSuccess) {
        is_error = true;
        return false;
      }
    }
  }

  if(data_type == "fp16_r") {

    if (dda) {
      if (hipMemcpy(dda, hda, sizeof(rocblas_half) * size_a, hipMemcpyHostToDevice)
          != hipSuccess) {
        is_error = true;
        return false;
      }
    }

    if (ddb) {
      if (hipMemcpy(ddb, hdb, sizeof(rocblas_half) * size_b, hipMemcpyHostToDevice)
          != hipSuccess) {
        is_error = true;
        return false;
      }
    }

    if (ddc) {
      if (hipMemcpy(ddc, hdc, sizeof(rocblas_half) * size_c, hipMemcpyHostToDevice)
          != hipSuccess) {
        is_error = true;
        return false;
      }
    }
  }

  if(data_type == "bf16_r") {

    if (dda) {

      printf ("sizeof(struct rocblas_bfloat16) -> %d",sizeof(struct rocblas_bfloat16));
      printf ("sizeof(rocblas_bfloat16) -> %d",sizeof(rocblas_bfloat16));

      if (hipMemcpy(dda, hda, sizeof(struct rocblas_bfloat16) * size_a, hipMemcpyHostToDevice)
          != hipSuccess) {
        is_error = true;
        return false;
      }
    }

    if (ddb) {
      if (hipMemcpy(ddb, hdb, sizeof(struct rocblas_bfloat16) * size_b, hipMemcpyHostToDevice)
          != hipSuccess) {
        is_error = true;
        return false;
      }
    }

    if (ddc) {
      if (hipMemcpy(ddc, hdc, sizeof(struct rocblas_bfloat16) * size_c, hipMemcpyHostToDevice)
          != hipSuccess) {
        is_error = true;
        return false;
      }
    }
  }

  is_error = false;
  return true;
}

/**
 * @brief allocates memory (for matrix multiplication) on the selected GPU
 * @return true if everything went fine, otherwise false
 */
bool rvs_blas::allocate_gpu_matrix_mem(void) {

  if(ops_type == "sgemm") {
    if (hipMalloc(&da, size_a * sizeof(float)) != hipSuccess)
      return false;
    if (hipMalloc(&db, size_b * sizeof(float)) != hipSuccess)
      return false;
    if (hipMalloc(&dc, size_c * sizeof(float)) != hipSuccess)
      return false;
  }

  if(ops_type == "dgemm") {
    if (hipMalloc(&ddbla, size_a * sizeof(double)) != hipSuccess)
      return false;
    if (hipMalloc(&ddblb, size_b * sizeof(double)) != hipSuccess)
      return false;
    if (hipMalloc(&ddblc, size_c * sizeof(double)) != hipSuccess)
      return false;
  }

  if(ops_type == "hgemm") {
    if (hipMalloc(&dhlfa, size_a * sizeof(rocblas_half)) != hipSuccess)
      return false;
    if (hipMalloc(&dhlfb, size_b * sizeof(rocblas_half)) != hipSuccess)
      return false;
    if (hipMalloc(&dhlfc, size_c * sizeof(rocblas_half)) != hipSuccess)
      return false;
    if (hipMalloc(&dhlfd, size_d * sizeof(rocblas_half)) != hipSuccess)
      return false;
  }

  if(data_type == "fp8_r") {
    if (hipMalloc(&dda, size_a * sizeof(struct rocblas_f8)) != hipSuccess)
      return false;
    if (hipMalloc(&ddb, size_b * sizeof(struct rocblas_f8)) != hipSuccess)
      return false;
    if (hipMalloc(&ddc, size_c * sizeof(struct rocblas_f8)) != hipSuccess)
      return false;
    if (hipMalloc(&ddd, size_d * sizeof(struct rocblas_f8)) != hipSuccess)
      return false;
  }

  if(data_type == "fp16_r") {
    if (hipMalloc(&dda, size_a * sizeof(rocblas_half)) != hipSuccess)
      return false;
    if (hipMalloc(&ddb, size_b * sizeof(rocblas_half)) != hipSuccess)
      return false;
    if (hipMalloc(&ddc, size_c * sizeof(rocblas_half)) != hipSuccess)
      return false;
    if (hipMalloc(&ddd, size_d * sizeof(rocblas_half)) != hipSuccess)
      return false;
  }

  if(data_type == "bf16_r") {
    if (hipMalloc(&dda, size_a * sizeof(struct rocblas_bfloat16)) != hipSuccess)
      return false;
    if (hipMalloc(&ddb, size_b * sizeof(struct rocblas_bfloat16)) != hipSuccess)
      return false;
    if (hipMalloc(&ddc, size_c * sizeof(struct rocblas_bfloat16)) != hipSuccess)
      return false;
    if (hipMalloc(&ddd, size_d * sizeof(struct rocblas_bfloat16)) != hipSuccess)
      return false;
  }

  return true;
}

/**
 * @brief gets steady clock time since epoch in microseconds
 */
double rvs_blas::get_time_us(void) {

  // Get steady clock now
  auto now = std::chrono::steady_clock::now();

  // Get duration since epoch in microseconds
  auto duration
    = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();

  return (static_cast<double>(duration));
}

/**
 * @brief releases GPU mem & destroys the rocBlas handle
 */
void rvs_blas::release_gpu_matrix_mem(void) {

  if (da)
    hipFree(da);
  if (db)
    hipFree(db);
  if (dc)
    hipFree(dc);

  if (ddbla)
    hipFree(ddbla);
  if (ddblb)
    hipFree(ddblb);
  if (ddblc)
    hipFree(ddblc);

  if (dhlfa)
    hipFree(dhlfa);
  if (dhlfb)
    hipFree(dhlfb);
  if (dhlfc)
    hipFree(dhlfc);
  if (dhlfd)
    hipFree(dhlfd);

  if (dda)
    hipFree(dda);
  if (ddb)
    hipFree(ddb);
  if (ddc)
    hipFree(ddc);
  if (ddd)
    hipFree(ddd);

  if (is_handle_init) {
    rocblas_destroy_handle(blas_handle);
    hipStreamDestroy(hip_stream);
  }
}

/**
 * @brief allocate host matrix memory
 * @return true if everything went fine, otherwise false
 */
bool rvs_blas::alocate_host_matrix_mem(void) {

  try {

    if(ops_type == "sgemm") {

      ha = new float[size_a];
      hb = new float[size_b];
      hc = new float[size_c];
    }

    if(ops_type == "dgemm") {

      hdbla = new double[size_a];
      hdblb = new double[size_b];
      hdblc = new double[size_c];
    }

    if(ops_type == "hgemm") {

      hhlfa = new rocblas_half[size_a];
      hhlfb = new rocblas_half[size_b];
      hhlfc = new rocblas_half[size_c];
    }

    if(data_type == "fp8_r") {

      hda = new struct rocblas_f8[size_a];
      hdb = new struct rocblas_f8[size_b];
      hdc = new struct rocblas_f8[size_c];
    }

    if(data_type == "fp16_r") {

      hda = new rocblas_half[size_a];
      hdb = new rocblas_half[size_b];
      hdc = new rocblas_half[size_c];
    }

    if(data_type == "bf16_r") {

      hda = new struct rocblas_bfloat16[size_a];
      hdb = new struct rocblas_bfloat16[size_b];
      hdc = new struct rocblas_bfloat16[size_c];
    }

    return true;
  } catch (std::bad_alloc&) {
    return false;
  }
}

/**
 * @brief releases the host matrix memory
 */
void rvs_blas::release_host_matrix_mem(void) {

  if (ha)
    delete []ha;
  if (hb)
    delete []hb;
  if (hc)
    delete []hc;

  if (hdbla)
    delete []hdbla;
  if (hdblb)
    delete []hdblb;
  if (hdblc)
    delete []hdblc;

  if (hhlfa)
    delete []hhlfa;
  if (hhlfb)
    delete []hhlfb;
  if (hhlfc)
    delete []hhlfc;

  if (hda)
    delete []hda;
  if (hdb)
    delete []hdb;
  if (hdc)
    delete []hdc;

}

/**
 * @brief checks whether the matrix multiplication completed
 * @return true if GPU finished with matrix multiplication, otherwise false
 */
bool rvs_blas::is_gemm_op_complete(void) {

  if (is_error)
    return true;  // avoid blocking the calling thread

  if(hipStreamSynchronize(hip_stream) != hipSuccess) {
    std::cout << "hipStreamSynchronize() failed !!! for stream " << hip_stream << std::endl;
    return false;
  }

  return true;
}

/**
 * @brief performs the GEMM matrix multiplication operations
 * @return true if GPU was able to enqueue the GEMM operation, otherwise false
 */
bool rvs_blas::run_blass_gemm(std::string ops_type) {

  if (!is_error) {

    if(ops_type == "sgemm") {

      float alpha = blas_alpha_val, beta = blas_beta_val;

      if (rocblas_sgemm(blas_handle, transa, transb,
            rvs_blas::m, rvs_blas::n, rvs_blas::k,
            &alpha, da, blas_lda_offset,
            db, blas_ldb_offset, &beta,
            dc, blas_ldc_offset) != rocblas_status_success) {
        is_error = true;  // GPU cannot enqueue the gemm
        std::cout << "\nError in rocblas_sgemm() !!!" << "\n";
        return false;
      } else {
        return true;
      }
    }

    if(ops_type == "dgemm") {

      double alpha = blas_alpha_val, beta = blas_beta_val;

      if (rocblas_dgemm(blas_handle, transa, transb,
            rvs_blas::m, rvs_blas::n, rvs_blas::k,
            &alpha, ddbla, blas_lda_offset,
            ddblb, blas_ldb_offset, &beta,
            ddblc, blas_ldc_offset) != rocblas_status_success) {
        is_error = true;  // GPU cannot enqueue the gemm
        std::cout << "\nError in rocblas_dgemm() !!!" << "\n";
        return false;
      } else {
        return true;
      }
    }

    if(ops_type == "hgemm") {

      _Float16 alpha = (float)blas_alpha_val;
      _Float16 beta = (float)blas_beta_val;

      if (rocblas_hgemm(blas_handle, transa, transb,
            rvs_blas::m, rvs_blas::n, rvs_blas::k,
            &alpha, dhlfa , blas_lda_offset,
            dhlfb, blas_ldb_offset, &beta,
            dhlfc, blas_ldc_offset) != rocblas_status_success) {
        is_error = true;  // GPU cannot enqueue the gemm
        std::cout << "\nError in rocblas_hgemm() !!!" << "\n";
        return false;
      } else {
        return true;
      }
    }

    if(data_type == "fp8_r") {

      rocblas_datatype a_type = rocblas_datatype_f8_r;
      rocblas_datatype b_type = rocblas_datatype_f8_r;
      rocblas_datatype c_type = rocblas_datatype_f8_r;
      rocblas_datatype d_type = rocblas_datatype_f8_r;

      rocblas_computetype compute_type = rocblas_compute_type_f32;
      rocblas_gemm_algo algo = rocblas_gemm_algo_standard;
      int32_t sol_index = 0;
      uint32_t flags = 0;

      rocblas_float alpha = (rocblas_float) blas_alpha_val;
      rocblas_float beta = (rocblas_float) blas_beta_val;

      if (rocblas_gemm_ex3(blas_handle, transa, transb,
            rvs_blas::m, rvs_blas::n, rvs_blas::k, &alpha,
            dda, a_type, blas_lda_offset,
            ddb, b_type, blas_ldb_offset, &beta,
            ddc, c_type, blas_ldc_offset,
            ddd, d_type, blas_ldd_offset,
            compute_type, algo, sol_index, flags) != rocblas_status_success) {

        is_error = true;  // GPU cannot enqueue the gemm
        std::cout << "\nError in rocblas_gemm_ex3() !!! " << "\n";
        return false;
      } else {
        return true;
      }
    }

    if(data_type == "fp16_r") {

      rocblas_datatype a_type = rocblas_datatype_f16_r;
      rocblas_datatype b_type = rocblas_datatype_f16_r;
      rocblas_datatype c_type = rocblas_datatype_f16_r;
      rocblas_datatype d_type = rocblas_datatype_f16_r;

      rocblas_datatype compute_type = rocblas_datatype_f32_r;
      rocblas_gemm_algo algo = rocblas_gemm_algo_standard;
      int32_t sol_index = 0;
      uint32_t flags = 0;

      rocblas_float alpha = (rocblas_float) blas_alpha_val;
      rocblas_float beta = (rocblas_float) blas_beta_val;

      if (rocblas_gemm_ex(blas_handle, transa, transb,
            rvs_blas::m, rvs_blas::n, rvs_blas::k, &alpha,
            dda, a_type, blas_lda_offset,
            ddb, b_type, blas_ldb_offset, &beta,
            ddc, c_type, blas_ldc_offset,
            ddd, d_type, blas_ldd_offset,
            compute_type, algo, sol_index, flags) != rocblas_status_success) {

        is_error = true;  // GPU cannot enqueue the gemm
        std::cout << "\nError in rocblas_gemm_ex() !!!" << "\n";
        return false;
      } else {
        return true;
      }
    }

    if(data_type == "bf16_r") {

      rocblas_datatype a_type = rocblas_datatype_bf16_r;
      rocblas_datatype b_type = rocblas_datatype_bf16_r;
      rocblas_datatype c_type = rocblas_datatype_bf16_r;
      rocblas_datatype d_type = rocblas_datatype_bf16_r;

      rocblas_datatype compute_type = rocblas_datatype_f32_r;
      rocblas_gemm_algo algo = rocblas_gemm_algo_standard;
      int32_t sol_index = 0;
      uint32_t flags = 0;

      rocblas_float alpha = (rocblas_float) blas_alpha_val;
      rocblas_float beta = (rocblas_float) blas_beta_val;

      if (rocblas_gemm_ex(blas_handle, transa, transb,
            rvs_blas::m, rvs_blas::n, rvs_blas::k, &alpha,
            dda, a_type, blas_lda_offset,
            ddb, b_type, blas_ldb_offset, &beta,
            ddc, c_type, blas_ldc_offset,
            ddd, d_type, blas_ldd_offset,
            compute_type, algo, sol_index, flags) != rocblas_status_success) {

        is_error = true;  // GPU cannot enqueue the gemm
        std::cout << "\nError in rocblas_gemm_ex() !!!" << "\n";
        return false;
      } else {
        return true;
      }
    }

  } else {
    return false;
  }

  return true;
}

/**
 * @brief generate matrix random data
 * it should be called before rocBlas GEMM
 */
void rvs_blas::generate_random_matrix_data(void) {

  size_t i;
  if (!is_error) {
    uint64_t nextr = (uint64_t) time(NULL);

    if(ops_type == "sgemm") {

      //SGEMM stuff
      for (i = 0; i < size_a; ++i)
        ha[i] = fast_pseudo_rand(&nextr, i);

      for (i = 0; i < size_b; ++i)
        hb[i] = fast_pseudo_rand(&nextr, i);

      for (int i = 0; i < size_c; ++i)
        hc[i] = fast_pseudo_rand(&nextr, i);
    }

    if(ops_type == "dgemm") {

      //DGEMM stuff
      for (i = 0; i < size_a; ++i)
        hdbla[i] = (double)fast_pseudo_rand(&nextr, i);

      for (i = 0; i < size_b; ++i)
        hdblb[i] = (double)fast_pseudo_rand(&nextr, i);

      for (int i = 0; i < size_c; ++i)
        hdblc[i] = (double)fast_pseudo_rand(&nextr, i);
    }

    if(ops_type == "hgemm") {

      //HGEMM stuff
      for (i = 0; i < size_a; ++i)
        hhlfa[i] = fast_pseudo_rand(&nextr, i);

      for (i = 0; i < size_b; ++i)
        hhlfb[i] = fast_pseudo_rand(&nextr, i);

      for (int i = 0; i < size_c; ++i)
        hhlfc[i] = fast_pseudo_rand(&nextr, i);
    }

    // 8-bit floating point real (fp8_r) format
    if(data_type == "fp8_r") {

      for (i = 0; i < size_a; ++i)
        ((struct rocblas_f8* )hda)[i] = rocblas_f8(fast_pseudo_rand(&nextr, i));

      for (i = 0; i < size_b; ++i)
        ((struct rocblas_f8* )hdb)[i] = rocblas_f8(fast_pseudo_rand(&nextr, i));

      for (i = 0; i < size_c; ++i)
        ((struct rocblas_f8* )hdc)[i] = rocblas_f8(fast_pseudo_rand(&nextr, i));
    }

    // 16-bit floating point real (fp16_r) format
    if(data_type == "fp16_r") {

      for (i = 0; i < size_a; ++i)
        ((rocblas_half* )hda)[i] = rocblas_half(fast_pseudo_rand(&nextr, i));

      for (i = 0; i < size_b; ++i)
        ((rocblas_half* )hdb)[i] = rocblas_half(fast_pseudo_rand(&nextr, i));

      for (i = 0; i < size_c; ++i)
        ((rocblas_half* )hdc)[i] = rocblas_half(fast_pseudo_rand(&nextr, i));
    }

    // 16-bit brain floating point real (bf16_r) format
    if(data_type == "bf16_r") {

      for (i = 0; i < size_a; ++i)
        ((struct rocblas_bfloat16* )hda)[i] = rocblas_bfloat16(fast_pseudo_rand(&nextr, i));

      for (i = 0; i < size_b; ++i)
        ((struct rocblas_bfloat16* )hdb)[i] = rocblas_bfloat16(fast_pseudo_rand(&nextr, i));

      for (i = 0; i < size_c; ++i)
        ((struct rocblas_bfloat16* )hdc)[i] = rocblas_bfloat16(fast_pseudo_rand(&nextr, i));
    }
  }
}

float rvsblas_uniform_int_1_10()
{
  if(!rvsblas_t_rand_init)
  {
    for(int i = 0; i < RANDBUF; i++)
    {
      rvsblas_t_rand_f_array[i]
        = (float)std::uniform_int_distribution<unsigned>(1, 10)(rvsblas_t_rng);
      rvsblas_t_rand_d_array[i] = (double)rvsblas_t_rand_f_array[i];
    }
    rvsblas_t_rand_init = 1;
  }
  rvsblas_t_rand_idx = (rvsblas_t_rand_idx + 1) & (RANDLEN - 1);
  return rvsblas_t_rand_f_array[rvsblas_t_rand_idx];
}

/**
 * @brief fast pseudo random generator 
 * @return floating point random number
 */
float rvs_blas::fast_pseudo_rand(uint64_t *nextr, size_t i) {

  if ("rand" == matrix_init) {

    if("fp8_r" == data_type) {
      return (float)std::uniform_int_distribution<int>(1, 2)(rvsblas_t_rng);
    }
    else if (("fp16_r" == data_type) || ("hgemm" == ops_type))
    {
      return (float)std::uniform_int_distribution<int>(-2, 2)(rvsblas_t_rng);
    }
    else if("bf16_r" == data_type)
    {
      return (float)std::uniform_int_distribution<int>(-2, 2)(rvsblas_t_rng);
    }
    else { /* sgemm, dgemm */
      return rvsblas_uniform_int_1_10();
    }
  }
  else if ("trig" == matrix_init) {
    return sin(static_cast<float>(i));
  }
  else {
    *nextr = *nextr * 1103515245 + 12345;
    return static_cast<float>(static_cast<uint32_t>
        ((*nextr / 65536) % RANDOM_CT)) / RANDOM_DIV_CT;
  }
}

/**
 * @brief HIP callback function
 * @param stream stream identifier
 * @param status status of stream operations
 * @param user_data user specified data
 * @return true if everything went fine, otherwise false
 */
void rvs_blas::hip_stream_callback (hipStream_t stream, hipError_t status, void *user_data) {

  bool error = false;

  if(nullptr == user_data)
  {
    return;
  }

  /* Call the registered callback function */
  rvs_blas *rvsblas = (rvs_blas *)user_data;

  if (hipSuccess == status) {
    error = true;
  }
  rvsblas->callback(error, rvsblas->user_data);
}

/**
 * @brief Set rvs blas callback
 * @param callback registered callback function
 * @param user_data user data
 * @return true if everything went fine, otherwise false
 */
bool rvs_blas::set_callback(rvsBlasCallback_t callback, void *user_data) {

  if(nullptr == callback) {
    return false;
  }

  this->callback = callback;
  this->user_data = user_data;

  /* Add callback to be called items in stream is completed */
  if(hipSuccess != hipStreamAddCallback (hip_stream, this->hip_stream_callback , (void *)this, 0)) {
    return false;
  }

  return true;
}

template <typename T>
void host_matrix_mul(T alpha,
    T        beta,
    int      M,
    int      N,
    int      K,
    const T* A,
    int      As1,
    int      As2,
    const T* B,
    int      Bs1,
    int      Bs2,
    T*       C,
    int      Cs1,
    int      Cs2)
{
    for(int i1 = 0; i1 < M; i1++)
    {
        for(int i2 = 0; i2 < N; i2++)
        {
            T t = 0.0;
            for(int i3 = 0; i3 < K; i3++)
            {
                t += A[i1 * As1 + i3 * As2] * B[i3 * Bs1 + i2 * Bs2];
            }
            C[i1 * Cs1 + i2 * Cs2] = beta * C[i1 * Cs1 + i2 * Cs2] + alpha * t;
        }
    }
}

/*! \brief  lapack_xlassq computes the scale and sumsq */
template <typename T>
void lapack_xlassq(int64_t n, T* X, int64_t incx, double& scale, double& sumsq)
{
    if(n > 0)
    {
        double abs_X = 0.0;
        for(int64_t i = 0; i < n; i++)
        {
            abs_X = std::abs(X[i * incx]);
            if(abs_X > 0 || std::isnan(abs_X))
            {
                if(scale < abs_X)
                {
                    sumsq = 1 + sumsq * std::sqrt(scale / abs_X);
                    scale = abs_X;
                }
                else
                {
                    sumsq = sumsq + std::sqrt(abs_X / scale);
                }
            }
        }
    }
}

/* LAPACK library functionality */
template <typename T>
void lapack_xcombssq(T* ssq, T* colssq)
{
    if(ssq[0] >= colssq[0])
    {
        if(ssq[0] != 0)
        {
            ssq[1] = ssq[1] + std::sqrt(colssq[0] / ssq[0]) * colssq[1];
        }
        else
        {
            ssq[1] = ssq[1] + colssq[1];
        }
    }
    else
    {
        ssq[1] = colssq[1] + std::sqrt(ssq[0] / colssq[0]) * ssq[1];
        ssq[0] = colssq[0];
    }
    return;
}

/*! \brief calculate_norm - returns the value of the one norm, infinity norm or
  the Frobenius norm of the matrix A. */
template <typename T>
double calculate_norm(char norm_type, int64_t m, int64_t n, T* A, int64_t lda, double* work)
{
    double value = 0.0;
    double sum   = 0.0;

    if(std::min(m, n) == 0)
        return value;

    int64_t a_offset = lda >= 0 ? 0 : lda * (1 - n); // e.g. vectors with negative inc
    if(norm_type == 'O' || norm_type == 'o' || norm_type == '1')
    {
        //Find the one norm of Matrix A.
        for(int64_t j = 0; j < n; j++)
        {
            sum = 0.0;
            for(int64_t i = 0; i < m; i++)
                sum = sum + std::abs(A[a_offset + i + j * lda]);

            if(value < sum || std::isnan(sum))
                value = sum;
        }
    }
    else if(norm_type == 'I' || norm_type == 'i')
    {
        //Find the infinity norm of Matrix A.
        for(int64_t j = 0; j < n; j++)
            for(int64_t i = 0; i < m; i++)
            {
                work[i] = work[i] + std::abs(A[a_offset + i + j * lda]);
            }
        for(int64_t i = 0; i < m; i++)
            if(value < work[i] || std::isnan(work[i]))
                value = work[i];
    }
    else if(norm_type == 'F' || norm_type == 'f')
    {
        //Find the Frobenius norm of Matrix A.
        //SSQ(1) is scale
        //SSQ(2) is sum-of-squares
        //For better accuracy, sum each column separately.
        std::vector<double> ssq(2);
        std::vector<double> colssq(2);

        ssq[0] = 0.0;
        ssq[1] = 1.0;
        for(int64_t j = 0; j < n; j++)
        {
            colssq[0] = 0.0;
            colssq[1] = 1.0;
            lapack_xlassq(m, A + a_offset + j * lda, 1, colssq[0], colssq[1]);
            lapack_xcombssq(ssq.data(), colssq.data());
        }
        value = ssq[0] * std::sqrt(ssq[1]);
    }
    return value;
}

template <typename T>
void m_axpy_64(int64_t N, T* alpha, T* x, int64_t incx, T* y, int64_t incy)
{
    int64_t x_offset = incx >= 0 ? 0 : incx * (1 - N);
    int64_t y_offset = incy >= 0 ? 0 : incy * (1 - N);
    for(int64_t i = 0; i < N; i++)
    {
        y[y_offset + i * incy] = (*alpha) * x[x_offset + i * incx] + y[y_offset + i * incy];
    }
}

template <
    typename T,
    std::enable_if<(std::is_same<T, float>{} || std::is_same<T, double>{}),int>::type = 0>
double norm_check_general(char norm_type, int64_t M, int64_t N, int64_t lda, T* hCPU, T* hGPU)
{
    // norm type can be 'O', 'I', 'F', 'o', 'i', 'f' for one, infinity or Frobenius norm
    // one norm is max column sum
    // infinity norm is max row sum
    // Frobenius is l2 norm of matrix entries

    std::vector<double> work(std::max(int64_t(1), M));
    int64_t             incx  = 1;
    double              alpha = -1.0;

    size_t size = M * size_t(N); // copying data so lda is M

    std::vector<double> hCPU_double(size);
    std::vector<double> hGPU_double(size);

    for(int64_t i = 0; i < N; i++)
    {
        int64_t src_col = i * int64_t(lda);
        int64_t dst_col = i * int64_t(M);
        for(int64_t j = 0; j < M; j++)
        {
            hCPU_double[size_t(dst_col + j)] = double(hCPU[src_col + j]);
            hGPU_double[size_t(dst_col + j)] = double(hGPU[src_col + j]);
        }
    }

    double cpu_norm = calculate_norm(norm_type, M, N, hCPU_double.data(), M, work.data());
    m_axpy_64(size, &alpha, hCPU_double.data(), incx, hGPU_double.data(), incx);
    double error = calculate_norm(norm_type, M, N, hGPU_double.data(), M, work.data()) / cpu_norm;
    return error;
}

// For F8, we convert the results to float to double first
template <
    typename T,
    std::enable_if<(std::is_same<T, rocblas_f8>{} || std::is_same<T, rocblas_bf8>{}), int>::type = 0>
double norm_check_general(char norm_type, int64_t M, int64_t N, int64_t lda, T* hCPU, T* hGPU)
{
    // norm type can be 'O', 'I', 'F', 'o', 'i', 'f' for one, infinity or Frobenius norm
    // one norm is max column sum
    // infinity norm is max row sum
    // Frobenius is l2 norm of matrix entries
    size_t size = M * size_t(N); // copying data so lda is M

    std::vector<double> hCPU_double(size);
    std::vector<double> hGPU_double(size);

    for(int64_t i = 0; i < N; i++)
    {
        int64_t src_col = i * int64_t(lda);
        int64_t dst_col = i * int64_t(M);
        for(int64_t j = 0; j < M; j++)
        {
            hCPU_double[size_t(dst_col + j)] = double(float(hCPU[src_col + j]));
            hGPU_double[size_t(dst_col + j)] = double(float(hGPU[src_col + j]));
        }
    }

    std::vector<double> work(std::max(int64_t(1), M));
    int64_t             incx  = 1;
    double              alpha = -1.0;

    double cpu_norm = calculate_norm(norm_type, M, N, hCPU_double.data(), M, work.data());
    m_axpy_64(size, &alpha, hCPU_double.data(), incx, hGPU_double.data(), incx);
    double error = calculate_norm(norm_type, M, N, hGPU_double.data(), M, work.data()) / cpu_norm;
    return error;
}

// For BF16, we convert the results to double first
template <typename T,
         std::enable_if<(std::is_same<T, rocblas_bfloat16>{}), int>::type = 0>
double norm_check_general(char norm_type, int64_t M, int64_t N, int64_t lda, T* hCPU, T* hGPU)
{
    size_t              size = N * (size_t)lda;
    std::vector<double> hCPU_double(size);
    std::vector<double> hGPU_double(size);

    for(int64_t i = 0; i < N; i++)
    {
        for(int64_t j = 0; j < M; j++)
        {
            size_t idx       = j + i * (size_t)lda;

            // zero extend lower 16 bits of bfloat16 to convert to IEEE float/double
            hCPU_double[idx] = double(float((uint32_t)hCPU[idx].data << 16));
            hGPU_double[idx] = double(float((uint32_t)hGPU[idx].data << 16));
        }
    }

    return norm_check_general<double>(norm_type, M, N, lda, hCPU_double.data(), hGPU_double.data());
}

// For half, we convert the results to double first
template <typename T,
         std::enable_if<(std::is_same<T, rocblas_half>{}), int>::type = 0>
double norm_check_general(char norm_type, int64_t M, int64_t N, int64_t lda, T* hCPU, T* hGPU)
{
    size_t              size = N * (size_t)lda;
    std::vector<double> hCPU_double(size);
    std::vector<double> hGPU_double(size);

    for(int64_t i = 0; i < N; i++)
    {
        for(int64_t j = 0; j < M; j++)
        {
            size_t idx       = j + i * (size_t)lda;
            hCPU_double[idx] = double(hCPU[idx]);
            hGPU_double[idx] = double(hGPU[idx]);
        }
    }

    return norm_check_general<double>(norm_type, M, N, lda, hCPU_double.data(), hGPU_double.data());
}

template <typename T>
bool rvs_blas::check_result_consistency(void * dout, uint64_t size, double &error) {

  // Allocate gemm output in host memory

  static int i = 1;

  printf("Start CONSI check !!!! \n\n");

  // current result
  if (!hco) {
    if (hipHostMalloc(&hco, size * sizeof(T), 0) != hipSuccess)
      return false;

    if (hipMemset(hco, 0, size * sizeof(T)) != hipSuccess)
      return false;
  }

  if (hipMemcpy(hco, dout, sizeof(T) * size, hipMemcpyDeviceToHost) != hipSuccess)
    return false;

  // previous result
  if (!hpo) {
    if (hipHostMalloc(&hpo, size * sizeof(T), 0) != hipSuccess)
      return false;

    if (hipMemset(hpo, 0, size * sizeof(T)) != hipSuccess)
      return false;

    if (hipMemcpy(hpo, dout, sizeof(T) * size, hipMemcpyDeviceToHost) != hipSuccess)
      return false;

    i++;
    printf("End CONSI check !!!! \n\n");
    // first iteration gemm outputs equal no checking required
    return true;
  }

    printf("error_freq -> %lu \n", error_freq);
    printf("error_count -> %lu \n", error_count);

#if 1
    if(error_freq && error_count) {

      if(i%error_freq == 0) {

        hipError_t error;

        if(error_count <= size) {

          printf("Error insertion for CONSI !!!\n");
          if ((error = hipMemset(hco, 0,  sizeof(T) * error_count)) != hipSuccess) {

            printf("hipMemset error for CONSI !!! -> %d  %p %lu\n", error, hco,  sizeof(T) * error_count);
            return false;
          }
        }
      }
    }
#endif

  // Norm checking

  T * fp = (T *)hpo;
  T * fc = (T *)hco;

  int64_t M = (int64_t)m;
  int64_t N = (int64_t)n;
  int64_t _ldc = (int64_t) blas_ldc_offset;

  error = std::abs(norm_check_general('F', M, N, _ldc, fp, fc));

  std::cout << "norm error ->" << std::setprecision(std::numeric_limits<double>::max_digits10) << error << std::endl;

  if (hipMemcpy(hpo, dout, sizeof(T) * size, hipMemcpyDeviceToHost) != hipSuccess)
    return false;

  i++;
  printf("End CONSI check !!!! \n\n");
  return true;
}

template <typename T>
bool rvs_blas::check_result_accuracy(void * dout, uint64_t size, double &error) {

  int a_stride_1 = 1,
      a_stride_2 = blas_lda_offset,
      b_stride_1 = 1,
      b_stride_2 = blas_ldb_offset;

  static int j  = 1;

  printf("\n\nStart ACCU check !!!! \n");

  if(transa == rocblas_operation_transpose) {
    a_stride_1 = blas_lda_offset;
    a_stride_2 = 1;
  }

  if(transb == rocblas_operation_transpose) {
    b_stride_1 = blas_ldb_offset;
    b_stride_2 = 1;
  }

  printf("size -> %d \n", size);

  T* hout;
  T* hdout;

  if(hipHostMalloc(&hout, size * sizeof(T), 0) != hipSuccess)
    return false;

  if(hipHostMalloc(&hdout, size * sizeof(T), 0) != hipSuccess)
    return false;

  T * _ha;
  T * _hb;
  T * _hc;
  T alpha = (T) blas_alpha_val;
  T beta = (T) blas_beta_val;

  if (std::is_same<T, float>{}) {
    _ha = (T *)ha;
    _hb = (T *)hb;
    _hc = (T *)hc;

    printf("type is float \n");

  }
  else {
    _ha = (T *)hdbla;
    _hb = (T *)hdblb;
    _hc = (T *)hdblc;

    printf("type is double \n");
  }

  hipMemcpy(hout, _hc, sizeof(T) * size, hipMemcpyHostToHost);

  std::cout << "m, n, k, lda, ldb, ldc = " << m << ", " << n << ", " << k << ", " << blas_lda_offset
    << ", " << blas_ldb_offset << ", " << blas_ldc_offset << std::endl;

  host_matrix_mul<T>(alpha,
      beta,
      m,
      n,
      k,
      _ha,
      a_stride_1,
      a_stride_2,
      _hb,
      b_stride_1,
      b_stride_2,
      hout,
      1,
      blas_ldc_offset);

    if (hipMemcpy(hdout, dout, sizeof(T) * size, hipMemcpyDeviceToHost)!= hipSuccess) {
      printf("hipMemcpy error for ACCU !!!\n");
      return false;
    }

#if 1

    printf("error_freq -> %lu \n", error_freq);
    printf("error_count -> %lu \n", error_count);

    if(error_freq && error_count) {

      if(j%error_freq == 0) {

        hipError_t error;

        if(error_count <= size) {

          printf("Error insertion for ACCU !!!\n");
          if ((error = hipMemset(hdout, 0,  sizeof(T) * error_count)) != hipSuccess) {

            printf("hipMemset error for ACCU !!! -> %d  %p %lu\n", error, hdout,  sizeof(T) * error_count);
            return false;
          }
        }
      }
    }
#endif

//  T max_relative_error = std::numeric_limits<T>::min();
  T max_relative_error = 0.0;

  for(int i = 0; i < size; i++)
  {

    if(hout[i] != hdout[i]) {
      printf("hout[%d] -> %f  hdout[%d] -> %f \n", i, hout[i], i, hdout[i]);
    }

    T relative_error = (hout[i] - hdout[i]) / hout[i];

    relative_error = relative_error > 0 ? relative_error : -relative_error;

    max_relative_error
      = relative_error < max_relative_error ? max_relative_error : relative_error;
  }

  T eps = std::numeric_limits<T>::epsilon();
  T tolerance = 10;

  if(max_relative_error > eps * tolerance)
  {
    std::cout << "FAIL: max_relative_error = " << max_relative_error << std::endl;
  }
  else
  {
//    std::cout << "PASS: max_relative_error = " << max_relative_error << std::endl;
  }

  error = max_relative_error;

  hipHostFree(hout);
  hipHostFree(hdout);

  j++;
  printf("End ACCU check !!!! \n\n");
  return true;
}

#if 0
template <typename T>
bool rvs_blas::check_result_accuracy(void * dout, uint64_t size, double &error) {

  int a_stride_1 = 1,
      a_stride_2 = blas_lda_offset,
      b_stride_1 = 1,
      b_stride_2 = blas_ldb_offset;

  static int j  = 1;

  printf("\n\nStart ACCU check !!!! \n");

  if(transa == rocblas_operation_transpose) {
    a_stride_1 = blas_lda_offset;
    a_stride_2 = 1;
  }

  if(transb == rocblas_operation_transpose) {
    b_stride_1 = blas_ldb_offset;
    b_stride_2 = 1;
  }

  printf("size -> %d \n", size);

  std::vector<T> hout(size);
  std::vector<T> hdout(size);

  T * _ha;
  T * _hb;
  T * _hc;
  T alpha = (T) blas_alpha_val;
  T beta = (T) blas_beta_val;

  if (std::is_same<T, float>{}) {
    _ha = (T *)ha;
    _hb = (T *)hb;
    _hc = (T *)hc;

    printf("type is float \n");

  }
  else {
    _ha = (T *)hdbla;
    _hb = (T *)hdblb;
    _hc = (T *)hdblc;

    printf("type is double \n");
  }

  hipMemcpy(hout.data(), _hc, sizeof(T) * size, hipMemcpyHostToHost);

  std::cout << "m, n, k, lda, ldb, ldc = " << m << ", " << n << ", " << k << ", " << blas_lda_offset
    << ", " << blas_ldb_offset << ", " << blas_ldc_offset << std::endl;

  host_matrix_mul<T>(alpha,
      beta,
      m,
      n,
      k,
      _ha,
      a_stride_1,
      a_stride_2,
      _hb,
      b_stride_1,
      b_stride_2,
      (T *)hout.data(),
      1,
      blas_ldc_offset);

    if (hipMemcpy(hdout.data(), dout, sizeof(T) * size, hipMemcpyDeviceToHost)!= hipSuccess) {
      printf("hipMemcpy error for ACCU !!!\n");
      return false;
    }

#if 1
  if(j%3 == 0) {

    hipError_t error;
    printf("Error insertion for ACCU !!!\n");



    if ((error = hipMemset(hdout.data(), 0,  sizeof(T) * 1000)) != hipSuccess) {

      printf("hipMemset error for ACCU !!! -> %d  %p %lu\n", error, hdout.data(),  sizeof(T) * 1000);
      return false;
    }
  }
#endif

  T max_relative_error = std::numeric_limits<T>::min();

  for(int i = 0; i < size; i++)
  {

    if(hout[i] != hdout[i]) {
      printf("hout[%d] -> %f  hdout[%d] -> %f \n", i, hout[i], i, hdout[i]);
    }

    T relative_error = (hout[i] - hdout[i]) / hout[i];

    relative_error = relative_error > 0 ? relative_error : -relative_error;

    max_relative_error
      = relative_error < max_relative_error ? max_relative_error : relative_error;
  }

  T eps = std::numeric_limits<T>::epsilon();
  T tolerance = 10;

  if(max_relative_error > eps * tolerance)
  {
    std::cout << "FAIL: max_relative_error = " << max_relative_error << std::endl;
  }
  else
  {
    std::cout << "PASS: max_relative_error = " << max_relative_error << std::endl;
  }

  error = max_relative_error;

  j++;
  printf("End ACCU check !!!! \n\n");
  return true;
}
#endif

bool rvs_blas::validate_gemm(bool self_check, bool accu_check, double &self_error, double &accu_error) {

  /* Gemm output checked for consistency/repeatability
     by comparing current output with previous output */
  if(self_check) {

    if(ops_type == "sgemm") {
      check_result_consistency<float>(dc, size_c, self_error);
    }
    else if(ops_type == "dgemm") {
      check_result_consistency<double>(ddblc, size_c, self_error);
    }
    else if(data_type == "fp8_r") {
      check_result_consistency<rocblas_f8>(ddd, size_d, self_error);
    }
    else if(data_type == "fp16_r") {
      check_result_consistency<rocblas_half>(ddd, size_d, self_error);
    }
    else if(data_type == "bf16_r") {
      check_result_consistency<rocblas_bfloat16>(ddd, size_d, self_error);
    }
    else {
      /* Do Nothing */
    }
  }

  /* Gemm output checked for accuracy/correctness by comparing
     host(CPU) output with device(GPU) output */
  if(accu_check) {

    if(ops_type == "sgemm") {
      check_result_accuracy<float>(dc, size_c, accu_error);
    }
    else if(ops_type == "dgemm") {
      check_result_accuracy<double>(ddblc, size_c, accu_error);
    }
    else {
      /* Do Nothing */
    }
  }

  return true;
}

void rvs_blas::set_gemm_error(uint64_t _error_freq, uint64_t _error_count) {

  error_freq = _error_freq;
  error_count = _error_count;

}

