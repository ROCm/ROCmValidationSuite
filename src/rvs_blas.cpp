/********************************************************************************
 *
 * Copyright (c) 2018-2025 Advanced Micro Devices, Inc. All rights reserved.
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
    float alpha , float beta, int lda, int ldb, int ldc, int ldd,
    std::string _ops_type, std::string _data_type, std::string _gemm_mode, int _batch_size,
    uint64_t _stride_a, uint64_t _stride_b, uint64_t _stride_c, uint64_t _stride_d,
    std::string _blas_source, std::string _compute_type, std::string _out_data_type)
  : gpu_device_index(_gpu_device_index)
  , ops_type(_ops_type)
  , data_type(_data_type)
  , out_data_type(_out_data_type)
  , m(_m), n(_n), k(_k)
  , matrix_init (_matrix_init)
  , size_a(0), size_b(0), size_c(0), size_d(0)
  , da(nullptr), db(nullptr), dc(nullptr), dd(nullptr)
  , ha(nullptr), hb(nullptr), hc(nullptr)
  , hpo(nullptr), hco(nullptr)
  , hout(nullptr), hdout(nullptr)
  , hip_stream(nullptr)
  , hiprand_generator(nullptr)
  , blas_handle(nullptr)
  , is_handle_init(false)
  , is_error(false)
  , check_count(1)
  , gemm_mode(_gemm_mode)
  , batch_size(_batch_size)
  , stride_a(_stride_a), stride_b(_stride_b), stride_c(_stride_c), stride_d(_stride_d)
  , blas_source(_blas_source)
  , compute_type(_compute_type)
  , hbl_handle(nullptr), hbl_workspace(nullptr)
  , hbl_layout_a(nullptr), hbl_layout_b(nullptr)
  , hbl_layout_c(nullptr), hbl_layout_d(nullptr)
  , hbl_matmul(nullptr)
{

  if (blas_source == "rocblas") {

    // Matrix a & b transpose
    transa = (transA == 0) ? rocblas_operation_none : rocblas_operation_transpose;
    transb = (transB == 0) ? rocblas_operation_none : rocblas_operation_transpose;

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

    // Setting matrix a, b & c sizes
    size_a = (transa == rocblas_operation_none) ? size_t(k) * blas_lda_offset : size_t(m) * blas_lda_offset;
    size_b = (transb == rocblas_operation_none) ? size_t(n) * blas_ldb_offset : size_t(k) * blas_ldb_offset;
    size_c = size_t(n) * blas_ldc_offset;

    // gemm based on data type, size of output matrix d.
    if (!data_type.empty()) {
      size_d = size_t(n) * blas_ldd_offset;
    }

    if(gemm_mode == "strided_batched") {

      if(stride_a == 0)
        stride_a = (transA == rocblas_operation_none) ? blas_lda_offset * k : blas_lda_offset * m;

      if(stride_b == 0)
        stride_b = (transB == rocblas_operation_none) ? blas_ldb_offset * n : blas_ldb_offset * k;

      if(stride_c == 0)
        stride_c = blas_ldc_offset * n;

      if(stride_d == 0)
        stride_d = blas_ldd_offset * n;

      size_a = (batch_size == 0) ? size_a : size_a + stride_a * (batch_size - 1);
      size_b = (batch_size == 0) ? size_b : size_b + stride_b * (batch_size - 1);
      size_c = (batch_size == 0) ? size_c : size_c + stride_c * (batch_size - 1);

      if (!data_type.empty()) {
        size_d = (batch_size == 0) ? size_d : size_d + stride_d * (batch_size - 1);
      }
    }
  }
  else if (blas_source == "hipblaslt") {

    // Matrix a & b transpose
    hbl_trans_a = (transA == 0) ? HIPBLAS_OP_N  : HIPBLAS_OP_T;
    hbl_trans_b = (transB == 0) ? HIPBLAS_OP_N : HIPBLAS_OP_T;

    // minimum leading dimensions
    int64_t min_lda = (hbl_trans_a == HIPBLAS_OP_N) ? m : k;
    int64_t min_ldb = (hbl_trans_b == HIPBLAS_OP_N) ? k : n;
    int64_t min_ldc = m;
    int64_t min_ldd = m;

    hbl_row_a = (hbl_trans_a == HIPBLAS_OP_N) ? m : k;
    hbl_col_a = (hbl_trans_a == HIPBLAS_OP_N) ? k : m;

    hbl_row_b = (hbl_trans_b == HIPBLAS_OP_N) ? k : n;
    hbl_col_b = (hbl_trans_b == HIPBLAS_OP_N) ? n : k;

    // setting actual leading dimensions
    hbl_lda_offset = ((int64_t)lda < min_lda) ? min_lda : (int64_t)lda;
    hbl_ldb_offset = ((int64_t)ldb < min_ldb) ? min_ldb : (int64_t)ldb;
    hbl_ldc_offset = ((int64_t)ldc < min_ldc) ? min_ldc : (int64_t)ldc;
    hbl_ldd_offset = ((int64_t)ldd < min_ldd) ? min_ldd : (int64_t)ldd;

    // Setting matrix a, b & c sizes
    size_a = (hbl_trans_a == HIPBLAS_OP_N) ? size_t(k) * hbl_lda_offset : size_t(m) * hbl_lda_offset;
    size_b = (hbl_trans_b == HIPBLAS_OP_N) ? size_t(n) * hbl_ldb_offset : size_t(k) * hbl_ldb_offset;
    size_c = size_t(n) * hbl_ldc_offset;

    // gemm based on data type, size of output matrix d.
    if (!data_type.empty()) {
      size_d = size_t(n) * hbl_ldd_offset;
    }

    // Get hip data type
    hbl_datatype = datatype_to_hip_datatype(data_type);
    if(RVS_BLAS_HIP_DATATYPE_INVALID == hbl_datatype) {
      is_error = true;
      std::cout << "\n Invalid data-type !!!" << "\n";
      return;
    }

    // output hip data type
    if (!out_data_type.empty()) {

      hbl_out_datatype = datatype_to_hip_datatype(out_data_type);
      if(RVS_BLAS_HIP_DATATYPE_INVALID == hbl_out_datatype) {
        is_error = true;
        std::cout << "\n Invalid output data-type !!!" << "\n";
        return;
      }
    }
    else {
      hbl_out_datatype = hbl_datatype;
    }

    // Get hipblas compute type
    hbl_computetype = computetype_to_hipblas_computetype(compute_type);
    if(RVS_BLAS_HIPBLAS_COMPUTETYPE_INVALID == hbl_computetype) {
      is_error = true;
      std::cout << "\n Invalid compute-type !!!" << "\n";
      return;
    }

  }
  else {
    is_error = true;
    std::cout << "\n Invalid blas source !!!" << "\n";
    return;
  }

  //setting alpha and beta val
  blas_alpha_val = alpha;
  blas_beta_val = beta;

  if (allocate_host_matrix_mem()) {
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

  if (hipStreamCreate(&hip_stream) != hipSuccess) {
    std::cout << "\n hipStreamCreate() failed !!!" << "\n";
    return false;
  }

  if (!allocate_gpu_matrix_mem()) {
    std::cout << "\n allocate_gpu_matrix_mem() failed !!!" << "\n";
    return false;
  }

  if (blas_source == "rocblas") {

    // rocblas initialize
    rocblas_initialize();

    if (rocblas_create_handle(&blas_handle) != rocblas_status_success) {
      std::cout << "\n rocblas_create_handle() failed !!!" << "\n";
      return false;
    }

    if (rocblas_set_stream(blas_handle, hip_stream) != rocblas_status_success) {
      std::cout << "\n rocblas_set_stream() failed !!!" << "\n";
      return false;
    }
  }
  else if (blas_source == "hipblaslt") {

    // Create hipblaslt handle
    if(hipblasLtCreate(&hbl_handle) != HIPBLAS_STATUS_SUCCESS) {
      std::cout << "\n hipblasLtCreate() failed !!!" << "\n";
      return false;
    }

    // Create Matrix Layouts
    if(hipblasLtMatrixLayoutCreate(&hbl_layout_a, hbl_datatype, hbl_row_a, hbl_col_a, hbl_lda_offset) != HIPBLAS_STATUS_SUCCESS) {
      std::cout << "\nLayout_A hipblasLtMatrixLayoutCreate() failed !!!" << "\n";
      return false;
    }

    if(hipblasLtMatrixLayoutCreate(&hbl_layout_b, hbl_datatype, hbl_row_b, hbl_col_b, hbl_ldb_offset) != HIPBLAS_STATUS_SUCCESS) {
      std::cout << "\nLayout_B hipblasLtMatrixLayoutCreate() failed !!!" << "\n";
      return false;
    }

    if(hipblasLtMatrixLayoutCreate(&hbl_layout_c, hbl_out_datatype, m, n, hbl_ldc_offset) != HIPBLAS_STATUS_SUCCESS) {
      std::cout << "\nLayout_C hipblasLtMatrixLayoutCreate() failed !!!" << "\n";
      return false;
    }

    if(hipblasLtMatrixLayoutCreate(&hbl_layout_d, hbl_out_datatype, m, n, hbl_ldd_offset) != HIPBLAS_STATUS_SUCCESS) {
      std::cout << "\nLayout_D hipblasLtMatrixLayoutCreate() failed !!!" << "\n";
      return false;
    }

    // Create Matrix Multiplication descriptor & set attributes
    if(hipblasLtMatmulDescCreate(&hbl_matmul, hbl_computetype, HIP_R_32F) != HIPBLAS_STATUS_SUCCESS) {
      std::cout << "\nhipblasLtMatmulDescCreate() failed !!!" << "\n";
      return false;
    }

    if(hipblasLtMatmulDescSetAttribute(hbl_matmul, HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT, &hbl_datatype, sizeof(void*)) != HIPBLAS_STATUS_SUCCESS) {
      std::cout << "\nhipblasLtMatmulDescSetAttribute() failed !!!" << "\n";
      return false;
    }

    if(hipblasLtMatmulDescSetAttribute(hbl_matmul, HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT, &hbl_datatype, sizeof(void*)) != HIPBLAS_STATUS_SUCCESS) {
      std::cout << "\nhipblasLtMatmulDescSetAttribute() failed !!!" << "\n";
      return false;
    }

    if(hipblasLtMatmulDescSetAttribute(hbl_matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &hbl_trans_a, sizeof(int32_t)) != HIPBLAS_STATUS_SUCCESS) {
      std::cout << "\nhipblasLtMatmulDescSetAttribute() failed !!!" << "\n";
      return false;
    }

    if(hipblasLtMatmulDescSetAttribute(hbl_matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &hbl_trans_b, sizeof(int32_t)) != HIPBLAS_STATUS_SUCCESS) {
      std::cout << "\nhipblasLtMatmulDescSetAttribute() failed !!!" << "\n";
      return false;
    }

    // Request only 1 algorithm
    constexpr int request_algo_count = 1;
    int returned_algo_count = 0;

    // Set max. workspace size to 32MB
    constexpr size_t max_workspace_size = 32 * 1024 * 1024;

    hipblasLtMatmulPreference_t pref;

    if(hipblasLtMatmulPreferenceCreate(&pref) != HIPBLAS_STATUS_SUCCESS) {
      std::cout << "\nhipblasLtMatmulPreferenceCreate() failed !!!" << "\n";
      return false;
    }

    if(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
          &max_workspace_size, sizeof(max_workspace_size)) != HIPBLAS_STATUS_SUCCESS) {
      std::cout << "\nhipblasLtMatmulPreferenceSetAttribute() failed !!!" << "\n";
      return false;
    }

    if(hipblasLtMatmulAlgoGetHeuristic(hbl_handle,
          hbl_matmul,
          hbl_layout_a,
          hbl_layout_b,
          hbl_layout_c,
          hbl_layout_d,
          pref,
          request_algo_count,
          &hbl_heuristic_result,
          &returned_algo_count) != HIPBLAS_STATUS_SUCCESS) {

      std::cout << "\nError in hipblasLtMatmulAlgoGetHeuristic() !!!" << "\n";
      hipblasLtMatmulPreferenceDestroy(pref);
      return false;
    }

    hipblasLtMatmulPreferenceDestroy(pref);

    if(returned_algo_count != request_algo_count) {
      std::cout << "\nIncorrect Heuristic algo. count !!!" << "\n";
      return false;
    }

    if(hbl_heuristic_result.workspaceSize) {
      // Allocate workspace for matrix multiplication
      hipMalloc(&hbl_workspace, hbl_heuristic_result.workspaceSize);
    }
  }
  else {
    std::cout << "\n Invalid blas source type !!!" << "\n";
    return false;
  }

  if("hiprand" == matrix_init) {

    // Create hipRAND generator, assign stream.
    if(hiprandCreateGenerator(&hiprand_generator, HIPRAND_RNG_PSEUDO_DEFAULT) != HIPRAND_STATUS_SUCCESS) {
      std::cout << "\n hiprandCreateGenerator() failed !!!" << "\n";
      return false;
    }

    if(hiprandSetStream(hiprand_generator, hip_stream) != HIPRAND_STATUS_SUCCESS) {
      std::cout << "\n hiprandSetStream() failed !!!" << "\n";
      return false;
    }
  }

  is_handle_init = true;
  return true;
}

template <typename Ti, typename To>
bool rvs_blas::copy_data_to_gpu(void) {

  if (da) {
    if (hipMemcpy(da, ha, sizeof(Ti) * size_a, hipMemcpyHostToDevice)
        != hipSuccess) {
      is_error = true;
      return false;
    }
  }

  if (db) {
    if (hipMemcpy(db, hb, sizeof(Ti) * size_b, hipMemcpyHostToDevice)
        != hipSuccess) {
      is_error = true;
      return false;
    }
  }

  if (dc) {
    if (hipMemcpy(dc, hc, sizeof(To) * size_c, hipMemcpyHostToDevice)
        != hipSuccess) {
      is_error = true;
      return false;
    }
  }

  is_error = false;
  return true;
}

/**
 * @brief copy data matrix from host to gpu
 * @return true if everything went fine, otherwise false
 */
bool rvs_blas::copy_data_to_gpu(void) {

  if("hiprand" == matrix_init) {

    // hipRAND no need for allocation in host memory, so no host to device copy !
    return true;
  }

  if(ops_type == "sgemm") {
    return copy_data_to_gpu<float, float>();
  }

  if(ops_type == "dgemm") {
    return copy_data_to_gpu<double, double>();
  }

  if(ops_type == "hgemm") {
    return copy_data_to_gpu<rocblas_half, rocblas_half>();
  }

  if(data_type == "fp8_r") {
    if (blas_source == "rocblas") {
      return copy_data_to_gpu<rocblas_f8, rocblas_f8>();
    }
    else if (blas_source == "hipblaslt") {
      return copy_data_to_gpu<hipblaslt_f8, float>();
    }
  }

  if(data_type == "fp8_e4m3_r") {
    return copy_data_to_gpu<hipblaslt_f8, float>();
  }

  if(data_type == "fp8_e5m2_r") {
    return copy_data_to_gpu<hipblaslt_bf8, float>();
  }

  if(data_type == "fp16_r") {
    if (blas_source == "rocblas") {
      return copy_data_to_gpu<rocblas_half, rocblas_half>();
    }
    else if (blas_source == "hipblaslt") {
      return copy_data_to_gpu<hipblasLtHalf, hipblasLtHalf>();
    }
  }

  if(data_type == "bf16_r") {
    if (blas_source == "rocblas") {
      return copy_data_to_gpu<rocblas_bfloat16, rocblas_bfloat16>();
    }
    else if (blas_source == "hipblaslt") {
      return copy_data_to_gpu<hipblasLtBfloat16, hipblasLtBfloat16>();
    }
  }

  if(data_type == "i8_r") {
    return copy_data_to_gpu<int8_t, int8_t>();
  }

  if(data_type == "fp32_r") {
    return copy_data_to_gpu<float, float>();
  }

  if(data_type == "fp64_r") {
    return copy_data_to_gpu<double, double>();
  }

  is_error = false;
  return true;
}

template <typename Ti, typename To>
bool rvs_blas::allocate_gpu_matrix_mem(void) {

  if (hipMalloc(&da, size_a * sizeof(Ti)) != hipSuccess)
    return false;
  if (hipMalloc(&db, size_b * sizeof(Ti)) != hipSuccess)
    return false;
  if (hipMalloc(&dc, size_c * sizeof(To)) != hipSuccess)
    return false;

  if(size_d)
    if (hipMalloc(&dd, size_d * sizeof(To)) != hipSuccess)
      return false;

  return true;
}

/**
 * @brief allocates memory (for matrix multiplication) on the selected GPU
 * @return true if everything went fine, otherwise false
 */
bool rvs_blas::allocate_gpu_matrix_mem(void) {

  if(ops_type == "sgemm") {
    return allocate_gpu_matrix_mem<float, float>();
  }

  if(ops_type == "dgemm") {
    return allocate_gpu_matrix_mem<double, double>();
  }

  if(ops_type == "hgemm") {
    return allocate_gpu_matrix_mem<rocblas_half, rocblas_half>();
  }

  if(data_type == "fp8_r") {
    if (blas_source == "rocblas") {
      return allocate_gpu_matrix_mem<rocblas_f8, rocblas_f8>();
    }
    else if (blas_source == "hipblaslt") {
      return allocate_gpu_matrix_mem<hipblaslt_f8, float>();
    }
  }

  if(data_type == "fp8_e4m3_r") {
    return allocate_gpu_matrix_mem<hipblaslt_f8, float>();
  }

  if(data_type == "fp8_e5m2_r") {
    return allocate_gpu_matrix_mem<hipblaslt_bf8, float>();
  }

  if(data_type == "fp16_r") {
    if (blas_source == "rocblas") {
      return allocate_gpu_matrix_mem<rocblas_half, rocblas_half>();
    }
    else if (blas_source == "hipblaslt") {
      return allocate_gpu_matrix_mem<hipblasLtHalf, hipblasLtHalf>();
    }
  }

  if(data_type == "bf16_r") {
    if (blas_source == "rocblas") {
      return allocate_gpu_matrix_mem<rocblas_bfloat16, rocblas_bfloat16>();
    }
    else if (blas_source == "hipblaslt") {
      return allocate_gpu_matrix_mem<hipblasLtBfloat16, hipblasLtBfloat16>();
    }
  }

  if(data_type == "i8_r") {
    return allocate_gpu_matrix_mem<int8_t, int8_t>();
  }

  if(data_type == "fp32_r") {
    return allocate_gpu_matrix_mem<float, float>();
  }

  if(data_type == "fp64_r") {
    return allocate_gpu_matrix_mem<double, double>();
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
  if (dd)
    hipFree(dd);

  if (is_handle_init) {

    if(blas_handle)
      rocblas_destroy_handle(blas_handle);

    if(hiprand_generator)
      hiprandDestroyGenerator(hiprand_generator);

    hipStreamDestroy(hip_stream);

    if(hbl_layout_a)
      hipblasLtMatrixLayoutDestroy(hbl_layout_a);
    if(hbl_layout_b)
      hipblasLtMatrixLayoutDestroy(hbl_layout_b);
    if(hbl_layout_c)
      hipblasLtMatrixLayoutDestroy(hbl_layout_c);
    if(hbl_layout_d)
      hipblasLtMatrixLayoutDestroy(hbl_layout_d);

    if(hbl_matmul)
      hipblasLtMatmulDescDestroy(hbl_matmul);

    if(hbl_workspace)
      hipFree(hbl_workspace);

    if(hbl_handle)
      hipblasLtDestroy(hbl_handle);
  }
}

/**
 * @brief allocate host matrix memory
 * @return true if everything went fine, otherwise false
 */
bool rvs_blas::allocate_host_matrix_mem(void) {

  if("hiprand" == matrix_init) {

    // hipRAND no need for allocation in host memory
    return true;
  }

  try {

    if(ops_type == "sgemm") {

      ha = new float[size_a];
      hb = new float[size_b];
      hc = new float[size_c];
    }

    if(ops_type == "dgemm") {

      ha = new double[size_a];
      hb = new double[size_b];
      hc = new double[size_c];
    }

    if(ops_type == "hgemm") {

      ha = new rocblas_half[size_a];
      hb = new rocblas_half[size_b];
      hc = new rocblas_half[size_c];
    }

    if(data_type == "fp8_r") {

      if (blas_source == "rocblas") {
        ha = new rocblas_f8[size_a];
        hb = new rocblas_f8[size_b];
        hc = new rocblas_f8[size_c];
      }
      else if (blas_source == "hipblaslt") {
        ha = new hipblaslt_f8[size_a];
        hb = new hipblaslt_f8[size_b];
        hc = new float[size_c];
      }
    }

    if(data_type == "fp8_e4m3_r") {

      ha = new hipblaslt_f8[size_a];
      hb = new hipblaslt_f8[size_b];
      hc = new float[size_c];
    }

    if(data_type == "fp8_e5m2_r") {

      ha = new hipblaslt_bf8[size_a];
      hb = new hipblaslt_bf8[size_b];
      hc = new float[size_c];
    }

    if(data_type == "fp16_r") {

      if (blas_source == "rocblas") {
        ha = new rocblas_half[size_a];
        hb = new rocblas_half[size_b];
        hc = new rocblas_half[size_c];
      }
      else if (blas_source == "hipblaslt") {
        ha = new hipblasLtHalf[size_a];
        hb = new hipblasLtHalf[size_b];
        hc = new hipblasLtHalf[size_c];
      }
    }

    if(data_type == "bf16_r") {

      if (blas_source == "rocblas") {
        ha = new rocblas_bfloat16[size_a];
        hb = new rocblas_bfloat16[size_b];
        hc = new rocblas_bfloat16[size_c];
      }
      else if (blas_source == "hipblaslt") {
        ha = new hipblasLtBfloat16[size_a];
        hb = new hipblasLtBfloat16[size_b];
        hc = new hipblasLtBfloat16[size_c];
      }
    }

    if(data_type == "i8_r") {

      ha = new int8_t[size_a];
      hb = new int8_t[size_b];
      hc = new int8_t[size_c];
    }

    if(data_type == "fp32_r") {

      ha = new float[size_a];
      hb = new float[size_b];
      hc = new float[size_c];
    }

    if(data_type == "fp64_r") {

      ha = new double[size_a];
      hb = new double[size_b];
      hc = new double[size_c];
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

  if (hpo)
    hipHostFree(hpo);
  if (hco)
    hipHostFree(hco);
  if(hout)
    hipHostFree(hout);
  if(hdout)
    hipHostFree(hdout);
}

/**
 * @brief checks whether all the gemm operations enqueued in the stream is completed
 * @return true if GPU finished with matrix multiplication, otherwise false
 */
bool rvs_blas::is_gemm_op_complete(void) {

  if (is_error)
    return false;

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
bool rvs_blas::run_blas_gemm(void) {

  if (is_error)
    return false;

  if(blas_source == "rocblas") {

    if(ops_type == "sgemm") {

      float alpha = blas_alpha_val, beta = blas_beta_val;

      if(gemm_mode == "strided_batched") {

        if (rocblas_sgemm_strided_batched(blas_handle, transa, transb,
              rvs_blas::m, rvs_blas::n, rvs_blas::k,
              &alpha, (float *)da, blas_lda_offset, stride_a,
              (float *)db, blas_ldb_offset, stride_b, &beta,
              (float *)dc, blas_ldc_offset, stride_c, batch_size) != rocblas_status_success) {
          is_error = true;  // GPU cannot enqueue the gemm
          std::cout << "\nError in rocblas_sgemm_strided_batched() !!!" << "\n";
          return false;
        } else {
          return true;
        }
      }
      else {

        if (rocblas_sgemm(blas_handle, transa, transb,
              rvs_blas::m, rvs_blas::n, rvs_blas::k,
              &alpha, (float *)da, blas_lda_offset,
              (float *)db, blas_ldb_offset, &beta,
              (float *)dc, blas_ldc_offset) != rocblas_status_success) {
          is_error = true;  // GPU cannot enqueue the gemm
          std::cout << "\nError in rocblas_sgemm() !!!" << "\n";
          return false;
        } else {
          return true;
        }
      }
    }

    if(ops_type == "dgemm") {

      double alpha = blas_alpha_val, beta = blas_beta_val;

      if(gemm_mode == "strided_batched") {

        if (rocblas_dgemm_strided_batched(blas_handle, transa, transb,
              rvs_blas::m, rvs_blas::n, rvs_blas::k,
              &alpha, (double *)da, blas_lda_offset, stride_a,
              (double *)db, blas_ldb_offset, stride_b, &beta,
              (double *)dc, blas_ldc_offset, stride_c, batch_size) != rocblas_status_success) {
          is_error = true;  // GPU cannot enqueue the gemm
          std::cout << "\nError in rocblas_dgemm_strided_batched() !!!" << "\n";
          return false;
        } else {
          return true;
        }
      }
      else {
        if (rocblas_dgemm(blas_handle, transa, transb,
              rvs_blas::m, rvs_blas::n, rvs_blas::k,
              &alpha, (double *)da, blas_lda_offset,
              (double *)db, blas_ldb_offset, &beta,
              (double *)dc, blas_ldc_offset) != rocblas_status_success) {
          is_error = true;  // GPU cannot enqueue the gemm
          std::cout << "\nError in rocblas_dgemm() !!!" << "\n";
          return false;
        } else {
          return true;
        }
      }
    }

    if(ops_type == "hgemm") {

      _Float16 alpha = (float)blas_alpha_val;
      _Float16 beta = (float)blas_beta_val;

      if(gemm_mode == "strided_batched") {

        if (rocblas_hgemm_strided_batched(blas_handle, transa, transb,
              rvs_blas::m, rvs_blas::n, rvs_blas::k,
              &alpha, (rocblas_half *)da, blas_lda_offset, stride_a,
              (rocblas_half *)db, blas_ldb_offset, stride_b, &beta,
              (rocblas_half *)dc, blas_ldc_offset, stride_c, batch_size) != rocblas_status_success) {
          is_error = true;  // GPU cannot enqueue the gemm
          std::cout << "\nError in rocblas_hgemm_strided_batched() !!!" << "\n";
          return false;
        } else {
          return true;
        }
      }
      else {

        if (rocblas_hgemm(blas_handle, transa, transb,
              rvs_blas::m, rvs_blas::n, rvs_blas::k,
              &alpha, (rocblas_half *)da, blas_lda_offset,
              (rocblas_half *)db, blas_ldb_offset, &beta,
              (rocblas_half *)dc, blas_ldc_offset) != rocblas_status_success) {
          is_error = true;  // GPU cannot enqueue the gemm
          std::cout << "\nError in rocblas_hgemm() !!!" << "\n";
          return false;
        } else {
          return true;
        }
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

      if(gemm_mode == "strided_batched") {

        if (rocblas_gemm_strided_batched_ex3(blas_handle, transa, transb,
              rvs_blas::m, rvs_blas::n, rvs_blas::k, &alpha,
              da, a_type, blas_lda_offset, stride_a,
              db, b_type, blas_ldb_offset, stride_b, &beta,
              dc, c_type, blas_ldc_offset, stride_c,
              dd, d_type, blas_ldd_offset, stride_d, batch_size,
              compute_type, algo, sol_index, flags) != rocblas_status_success) {

          is_error = true;  // GPU cannot enqueue the gemm
          std::cout << "\nError in rocblas_gemm_strided_batched_ex3() !!! " << "\n";
          return false;
        } else {
          return true;
        }
      }
      else {

        if (rocblas_gemm_ex3(blas_handle, transa, transb,
              rvs_blas::m, rvs_blas::n, rvs_blas::k, &alpha,
              da, a_type, blas_lda_offset,
              db, b_type, blas_ldb_offset, &beta,
              dc, c_type, blas_ldc_offset,
              dd, d_type, blas_ldd_offset,
              compute_type, algo, sol_index, flags) != rocblas_status_success) {

          is_error = true;  // GPU cannot enqueue the gemm
          std::cout << "\nError in rocblas_gemm_ex3() !!! " << "\n";
          return false;
        } else {
          return true;
        }
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

      if(gemm_mode == "strided_batched") {

        if (rocblas_gemm_strided_batched_ex(blas_handle, transa, transb,
              rvs_blas::m, rvs_blas::n, rvs_blas::k, &alpha,
              da, a_type, blas_lda_offset, stride_a,
              db, b_type, blas_ldb_offset, stride_b, &beta,
              dc, c_type, blas_ldc_offset, stride_c,
              dd, d_type, blas_ldd_offset, stride_d, batch_size,
              compute_type, algo, sol_index, flags) != rocblas_status_success) {

          is_error = true;  // GPU cannot enqueue the gemm
          std::cout << "\nError in rocblas_gemm_strided_batched_ex() !!!" << "\n";
          return false;
        } else {
          return true;
        }
      }
      else {

        if (rocblas_gemm_ex(blas_handle, transa, transb,
              rvs_blas::m, rvs_blas::n, rvs_blas::k, &alpha,
              da, a_type, blas_lda_offset,
              db, b_type, blas_ldb_offset, &beta,
              dc, c_type, blas_ldc_offset,
              dd, d_type, blas_ldd_offset,
              compute_type, algo, sol_index, flags) != rocblas_status_success) {

          is_error = true;  // GPU cannot enqueue the gemm
          std::cout << "\nError in rocblas_gemm_ex() !!!" << "\n";
          return false;
        } else {
          return true;
        }
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

      if(gemm_mode == "strided_batched") {

        if (rocblas_gemm_strided_batched_ex(blas_handle, transa, transb,
              rvs_blas::m, rvs_blas::n, rvs_blas::k, &alpha,
              da, a_type, blas_lda_offset, stride_a,
              db, b_type, blas_ldb_offset, stride_b, &beta,
              dc, c_type, blas_ldc_offset, stride_c,
              dd, d_type, blas_ldd_offset, stride_d, batch_size,
              compute_type, algo, sol_index, flags) != rocblas_status_success) {

          is_error = true;  // GPU cannot enqueue the gemm
          std::cout << "\nError in rocblas_gemm_strided_batched_ex() !!!" << "\n";
          return false;
        } else {
          return true;
        }
      }
      else {

        if (rocblas_gemm_ex(blas_handle, transa, transb,
              rvs_blas::m, rvs_blas::n, rvs_blas::k, &alpha,
              da, a_type, blas_lda_offset,
              db, b_type, blas_ldb_offset, &beta,
              dc, c_type, blas_ldc_offset,
              dd, d_type, blas_ldd_offset,
              compute_type, algo, sol_index, flags) != rocblas_status_success) {

          is_error = true;  // GPU cannot enqueue the gemm
          std::cout << "\nError in rocblas_gemm_ex() !!!" << "\n";
          return false;
        } else {
          return true;
        }
      }
    }
  }
  else if(blas_source == "hipblaslt") {

    if (hipblasLtMatmul(hbl_handle, hbl_matmul,
          &blas_alpha_val, da, hbl_layout_a,
          db, hbl_layout_b, &blas_beta_val,
          dc, hbl_layout_c,
          dd, hbl_layout_d,
          &hbl_heuristic_result.algo, hbl_workspace, hbl_heuristic_result.workspaceSize,
          hip_stream) != HIPBLAS_STATUS_SUCCESS) {

      is_error = true;  // GPU cannot enqueue the gemm
      std::cout << "\nError in hipblasLtMatmul() !!!" << "\n";
      return false;
    }
  }
  else {
    return false;
  }

  return true;
}

/**
 * @brief generate matrix random data
 * it should be called before rocBlas GEMM
 */
void rvs_blas::generate_random_matrix_data(void) {

  if (!is_error) {

    if("hiprand" == matrix_init) {

      if((ops_type == "dgemm") || (data_type == "fp64_r")) {

        if(hiprandGenerateUniformDouble(hiprand_generator, (double *)da, size_a) != HIPRAND_STATUS_SUCCESS) {
          std::cout << "\n hiprandGenerateUniformDouble() failed !!!" << "\n";
          is_error = true;
          return;
        }

        if(hiprandGenerateUniformDouble(hiprand_generator, (double *)db, size_b) != HIPRAND_STATUS_SUCCESS) {
          std::cout << "\n hiprandGenerateUniformDouble() failed !!!" << "\n";
          is_error = true;
          return;
        }

        if(hiprandGenerateUniformDouble(hiprand_generator, (double *)dc, size_c) != HIPRAND_STATUS_SUCCESS) {
          std::cout << "\n hiprandGenerateUniformDouble() failed !!!" << "\n";
          is_error = true;
          return;
        }

        if(hipStreamSynchronize(hip_stream) != hipSuccess) {
          std::cout << "hipStreamSynchronize() failed !!! for stream " << hip_stream << std::endl;
          is_error = true;
          return;
        }
      }
      else if((ops_type == "sgemm") || (data_type == "fp32_r")) {

        if(hiprandGenerateUniform(hiprand_generator, (float *)da, size_a) != HIPRAND_STATUS_SUCCESS) {
          std::cout << "\n hiprandGenerateUniform() failed !!!" << "\n";
          is_error = true;
          return;
        }

        if(hiprandGenerateUniform(hiprand_generator, (float *)db, size_b) != HIPRAND_STATUS_SUCCESS) {
          std::cout << "\n hiprandGenerateUniform() failed !!!" << "\n";
          is_error = true;
          return;
        }

        if(hiprandGenerateUniform(hiprand_generator, (float *)dc, size_c) != HIPRAND_STATUS_SUCCESS) {
          std::cout << "\n hiprandGenerateUniform() failed !!!" << "\n";
          is_error = true;
          return;
        }

        if(hipStreamSynchronize(hip_stream) != hipSuccess) {
          std::cout << "hipStreamSynchronize() failed !!! for stream " << hip_stream << std::endl;
          is_error = true;
          return;
        }
      }
    }
    else {

      size_t i;
      uint64_t nextr = (uint64_t) time(NULL);

      //SGEMM (float fp32_r)
      if((ops_type == "sgemm") || (data_type == "fp32_r")) {

        for (i = 0; i < size_a; ++i)
         ((float *) ha)[i] = fast_pseudo_rand(&nextr, i);

        for (i = 0; i < size_b; ++i)
          ((float *) hb)[i] = fast_pseudo_rand(&nextr, i);

        for (int i = 0; i < size_c; ++i)
          ((float *) hc)[i] = fast_pseudo_rand(&nextr, i);
      }

      //DGEMM (double fp64_r)
      if((ops_type == "dgemm") || (data_type == "fp64_r")) {

        for (i = 0; i < size_a; ++i)
          ((double *) ha)[i] = (double)fast_pseudo_rand(&nextr, i);

        for (i = 0; i < size_b; ++i)
          ((double *) hb)[i] = (double)fast_pseudo_rand(&nextr, i);

        for (int i = 0; i < size_c; ++i)
          ((double *) hc)[i] = (double)fast_pseudo_rand(&nextr, i);
      }

      //HGEMM (half-float fp16_r)
      if(ops_type == "hgemm") {

        for (i = 0; i < size_a; ++i)
          ((rocblas_half* )ha)[i] = fast_pseudo_rand(&nextr, i);

        for (i = 0; i < size_b; ++i)
          ((rocblas_half* )hb)[i] = fast_pseudo_rand(&nextr, i);

        for (int i = 0; i < size_c; ++i)
          ((rocblas_half* )hc)[i] = fast_pseudo_rand(&nextr, i);
      }

      // 8-bit floating point real (fp8_r) format
      if(data_type == "fp8_r") {

        if (blas_source == "rocblas") {
          for (i = 0; i < size_a; ++i)
            ((rocblas_f8* )ha)[i] = rocblas_f8(fast_pseudo_rand(&nextr, i));

          for (i = 0; i < size_b; ++i)
            ((rocblas_f8* )hb)[i] = rocblas_f8(fast_pseudo_rand(&nextr, i));

          for (i = 0; i < size_c; ++i)
            ((rocblas_f8* )hc)[i] = rocblas_f8(fast_pseudo_rand(&nextr, i));
        }
        else if (blas_source == "hipblaslt") {

          for (i = 0; i < size_a; ++i)
            ((hipblaslt_f8* )ha)[i] = hipblaslt_f8(fast_pseudo_rand(&nextr, i));

          for (i = 0; i < size_b; ++i)
            ((hipblaslt_f8* )hb)[i] = hipblaslt_f8(fast_pseudo_rand(&nextr, i));

          for (i = 0; i < size_c; ++i)
            ((float* )hc)[i] = float(fast_pseudo_rand(&nextr, i));
        }
      }

      // 8-bit floating point real OCP E4M3 (fp8_e4m3_r) format
      if(data_type == "fp8_e4m3_r") {

        for (i = 0; i < size_a; ++i)
          ((hipblaslt_f8* )ha)[i] = hipblaslt_f8(fast_pseudo_rand(&nextr, i));

        for (i = 0; i < size_b; ++i)
          ((hipblaslt_f8* )hb)[i] = hipblaslt_f8(fast_pseudo_rand(&nextr, i));

        for (i = 0; i < size_c; ++i)
          ((float* )hc)[i] = float(fast_pseudo_rand(&nextr, i));
      }

      // 8-bit floating point real OCP E5M2 (fp8_e5m2_r) format
      if(data_type == "fp8_e5m2_r") {

        for (i = 0; i < size_a; ++i)
          ((hipblaslt_bf8* )ha)[i] = hipblaslt_bf8(fast_pseudo_rand(&nextr, i));

        for (i = 0; i < size_b; ++i)
          ((hipblaslt_bf8* )hb)[i] = hipblaslt_bf8(fast_pseudo_rand(&nextr, i));

        for (i = 0; i < size_c; ++i)
          ((float* )hc)[i] = float(fast_pseudo_rand(&nextr, i));
      }

      // 16-bit floating point real (fp16_r) format
      if(data_type == "fp16_r") {

        if (blas_source == "rocblas") {
          for (i = 0; i < size_a; ++i)
            ((rocblas_half* )ha)[i] = rocblas_half(fast_pseudo_rand(&nextr, i));

          for (i = 0; i < size_b; ++i)
            ((rocblas_half* )hb)[i] = rocblas_half(fast_pseudo_rand(&nextr, i));

          for (i = 0; i < size_c; ++i)
            ((rocblas_half* )hc)[i] = rocblas_half(fast_pseudo_rand(&nextr, i));
        }
        else if (blas_source == "hipblaslt") {
          for (i = 0; i < size_a; ++i)
            ((hipblasLtHalf* )ha)[i] = hipblasLtHalf(fast_pseudo_rand(&nextr, i));

          for (i = 0; i < size_b; ++i)
            ((hipblasLtHalf* )hb)[i] = hipblasLtHalf(fast_pseudo_rand(&nextr, i));

          for (i = 0; i < size_c; ++i)
            ((hipblasLtHalf* )hc)[i] = hipblasLtHalf(fast_pseudo_rand(&nextr, i));
        }
      }

      // 16-bit brain floating point real (bf16_r) format
      if(data_type == "bf16_r") {

        if (blas_source == "rocblas") {
          for (i = 0; i < size_a; ++i)
            ((rocblas_bfloat16* )ha)[i] = rocblas_bfloat16(fast_pseudo_rand(&nextr, i));

          for (i = 0; i < size_b; ++i)
            ((rocblas_bfloat16* )hb)[i] = rocblas_bfloat16(fast_pseudo_rand(&nextr, i));

          for (i = 0; i < size_c; ++i)
            ((rocblas_bfloat16* )hc)[i] = rocblas_bfloat16(fast_pseudo_rand(&nextr, i));
        }
        else if (blas_source == "hipblaslt") {

          for (i = 0; i < size_a; ++i)
            ((hipblasLtBfloat16* )ha)[i] = hipblasLtBfloat16(fast_pseudo_rand(&nextr, i));

          for (i = 0; i < size_b; ++i)
            ((hipblasLtBfloat16* )hb)[i] = hipblasLtBfloat16(fast_pseudo_rand(&nextr, i));

          for (i = 0; i < size_c; ++i)
            ((hipblasLtBfloat16* )hc)[i] = hipblasLtBfloat16(fast_pseudo_rand(&nextr, i));
        }
      }

      // 8-bit integer real (i8_r) format
      if(data_type == "i8_r") {

        for (i = 0; i < size_a; ++i)
          ((int8_t* )ha)[i] = int8_t(fast_pseudo_rand(&nextr, i));

        for (i = 0; i < size_b; ++i)
          ((int8_t* )hb)[i] = int8_t(fast_pseudo_rand(&nextr, i));

        for (i = 0; i < size_c; ++i)
          ((int8_t* )hc)[i] = int8_t(fast_pseudo_rand(&nextr, i));
      }
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

    if(("fp8_r" == data_type) && (blas_source == "rocblas")) {
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
    else if("i8_r" == data_type)
    {
      return (float)std::uniform_int_distribution<unsigned short>(1, 3)(rvsblas_t_rng);
    }
    else { /* sgemm (fp32), dgemm (fp64), fp8_e4m3_r, fp8_e5m2_r */
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

/**
 * Host(CPU) based Matrix multiplication -> C = alpha (A*B) + beta (C).
 * @param[in] alpha scalar for matrix A*B
 * @param[in] beta scalar for matrix C
 * @param[in] M matrix A rows
 * @param[in] N matrix B cols
 * @param[in] K matrix A/B cols/rows respectively
 * @param[in] A matrix A
 * @param[in] B matrix B
 * @param[in,out] C matrix C
 */
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

/*! \brief F-norm utility function that computes the scale
  (largest absolute element in the column) and sum sqrt of the matric column */
  template <typename T>
void lapack_xlassq(int64_t n, T* X, int64_t incx, double& scale, double& sumsq) {

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

/*! \brief F-norm utility function that acculatively computes
  the sum sqrt of the matrix from sum sqrt of the matric column */
template <typename T>
void lapack_xcombssq(T* ssq, T* colssq) {

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

/*! \brief matrix norm function calculates the one norm,
  infinity norm or the frobenius norm of the matrix A */
template <typename T>
double calculate_norm(char norm_type, int64_t m, int64_t n, T* A, int64_t lda, double* work) {

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

/*! \brief Matrix utility function to create difference matrix from two matrices */
template <typename T>
void m_axpy_64(int64_t N, T* alpha, T* x, int64_t incx, T* y, int64_t incy) {

  int64_t x_offset = incx >= 0 ? 0 : incx * (1 - N);
  int64_t y_offset = incy >= 0 ? 0 : incy * (1 - N);
  for(int64_t i = 0; i < N; i++)
  {
    y[y_offset + i * incy] = (*alpha) * x[x_offset + i * incx] + y[y_offset + i * incy];
  }
}

/**
 * Get relative norm error for float/double data type matrices.
 * @param[in] norm_type matrix norm type to execute.
 * @param[in] M matrix rows
 * @param[in] N matrix columns
 * @param[in] Ida matrix leading dimension
 * @param[in] hA host memory matrix A
 * @param[in] hB host memory matrix B
 */
template <
    typename T,
    std::enable_if<(std::is_same<T, float>{} || std::is_same<T, double>{}),int>::type = 0>
double check_norm_error(char norm_type, int64_t M, int64_t N, int64_t lda, T* hA, T* hB) {

  // norm type can be 'O', 'I', 'F', 'o', 'i', 'f' for one, infinity or Frobenius norm
  // one norm is max column sum
  // infinity norm is max row sum
  // Frobenius is l2 norm of matrix entries

  std::vector<double> work(std::max(int64_t(1), M));
  int64_t             incx  = 1;
  double              alpha = -1.0;

  size_t size = M * size_t(N); // copying data so lda is M

  std::vector<double> hA_double(size);
  std::vector<double> hB_double(size);

  for(int64_t i = 0; i < N; i++)
  {
    int64_t src_col = i * int64_t(lda);
    int64_t dst_col = i * int64_t(M);
    for(int64_t j = 0; j < M; j++)
    {
      hA_double[size_t(dst_col + j)] = double(hA[src_col + j]);
      hB_double[size_t(dst_col + j)] = double(hB[src_col + j]);
    }
  }

  double a_norm = calculate_norm(norm_type, M, N, hA_double.data(), M, work.data());
  m_axpy_64(size, &alpha, hA_double.data(), incx, hB_double.data(), incx);
  double error = calculate_norm(norm_type, M, N, hB_double.data(), M, work.data()) / a_norm;

  return error;
}

/**
 * Get relative norm error for fp8 data type matrices.
 * @param[in] norm_type matrix norm type to execute.
 * @param[in] M matrix rows
 * @param[in] N matrix columns
 * @param[in] Ida matrix leading dimension
 * @param[in] hA host memory matrix A
 * @param[in] hB host memory matrix B
 */
template <
    typename T,
    std::enable_if<std::is_same<T, rocblas_f8>{}, int>::type = 0>
double check_norm_error(char norm_type, int64_t M, int64_t N, int64_t lda, T* hA, T* hB) {

  // norm type can be 'O', 'I', 'F', 'o', 'i', 'f' for one, infinity or Frobenius norm
  // one norm is max column sum
  // infinity norm is max row sum
  // Frobenius is l2 norm of matrix entries
  size_t size = M * size_t(N); // copying data so lda is M

  std::vector<double> hA_double(size);
  std::vector<double> hB_double(size);

  for(int64_t i = 0; i < N; i++)
  {
    int64_t src_col = i * int64_t(lda);
    int64_t dst_col = i * int64_t(M);
    for(int64_t j = 0; j < M; j++)
    {
      hA_double[size_t(dst_col + j)] = double(float(hA[src_col + j]));
      hB_double[size_t(dst_col + j)] = double(float(hB[src_col + j]));
    }
  }

  std::vector<double> work(std::max(int64_t(1), M));
  int64_t             incx  = 1;
  double              alpha = -1.0;

  double a_norm = calculate_norm(norm_type, M, N, hA_double.data(), M, work.data());
  m_axpy_64(size, &alpha, hA_double.data(), incx, hB_double.data(), incx);
  double error = calculate_norm(norm_type, M, N, hB_double.data(), M, work.data()) / a_norm;

  return error;
}

/**
 * Get relative norm error for bf16 data type matrices.
 * @param[in] norm_type matrix norm type to execute.
 * @param[in] M matrix rows
 * @param[in] N matrix columns
 * @param[in] Ida matrix leading dimension
 * @param[in] hA host memory matrix A
 * @param[in] hB host memory matrix B
 */
template <typename T,
         std::enable_if<(std::is_same<T, rocblas_bfloat16>{}), int>::type = 0>
double check_norm_error(char norm_type, int64_t M, int64_t N, int64_t lda, T* hA, T* hB) {

  size_t              size = N * (size_t)lda;
  std::vector<double> hA_double(size);
  std::vector<double> hB_double(size);

  for(int64_t i = 0; i < N; i++)
  {
    for(int64_t j = 0; j < M; j++)
    {
      size_t idx = j + i * (size_t)lda;

      // zero extend lower 16 bits of bfloat16 to convert to IEEE float/double
      hA_double[idx] = double(float((uint32_t)hA[idx].data << 16));
      hB_double[idx] = double(float((uint32_t)hB[idx].data << 16));
    }
  }

  return check_norm_error<double>(norm_type, M, N, lda, hA_double.data(), hB_double.data());
}

/**
 * Get relative norm error for fp16 (half) data type matrices.
 * @param[in] norm_type matrix norm type to execute.
 * @param[in] M matrix rows
 * @param[in] N matrix columns
 * @param[in] Ida matrix leading dimension
 * @param[in] hA host memory matrix A
 * @param[in] hB host memory matrix B
 */
template <typename T,
         std::enable_if<(std::is_same<T, rocblas_half>{}), int>::type = 0>
double check_norm_error(char norm_type, int64_t M, int64_t N, int64_t lda, T* hA, T* hB) {

  size_t              size = N * (size_t)lda;
  std::vector<double> hA_double(size);
  std::vector<double> hB_double(size);

  for(int64_t i = 0; i < N; i++)
  {
    for(int64_t j = 0; j < M; j++)
    {
      size_t idx       = j + i * (size_t)lda;
      hA_double[idx] = double(hA[idx]);
      hB_double[idx] = double(hB[idx]);
    }
  }

  return check_norm_error<double>(norm_type, M, N, lda, hA_double.data(), hB_double.data());
}

/**
 * Check gemm output for consistency (current output vs previous output).
 * @param[in] dout Device (GPU) matrix output.
 * @param[in] size No of elements in matrix output.
 * @param[out] error Relative F-norm self error.
 */
template <typename T>
bool rvs_blas::check_result_consistency(void * dout, size_t size, double &error) {

  /* Allocate host memory for current gemm output */
  if (!hco) {
    if (hipHostMalloc(&hco, size * sizeof(T), 0) != hipSuccess)
      return false;

    if (hipMemset(hco, 0, size * sizeof(T)) != hipSuccess)
      return false;
  }

  /* Copy current device gemm output to host memory */
  if (hipMemcpy(hco, dout, sizeof(T) * size, hipMemcpyDeviceToHost) != hipSuccess)
    return false;

  /* Allocate host memory for previous gemm output */
  if (!hpo) {
    if (hipHostMalloc(&hpo, size * sizeof(T), 0) != hipSuccess)
      return false;

    if (hipMemset(hpo, 0, size * sizeof(T)) != hipSuccess)
      return false;

    /* Copy current device gemm output to host memory */
    if (hipMemcpy(hpo, dout, sizeof(T) * size, hipMemcpyDeviceToHost) != hipSuccess)
      return false;

    /* Exit first iteration of self-check as there is no previous result yet ! */
    return true;
  }

  /* If error injection is enabled, insert error in gemm output */
  if(error_freq && error_count && check_count) {

    /* Insert error at set error frequency */
    if(check_count%error_freq == 0) {

      if(error_count <= size) {

        if (hipMemset(hco, 0,  sizeof(T) * error_count) != hipSuccess)
          return false;
      }
    }
  }

  /* Norm checking */

  T * fp = (T *)hpo;
  T * fc = (T *)hco;

  int64_t M = (int64_t)m;
  int64_t N = (int64_t)n;
  int64_t _ldc = (int64_t) blas_ldc_offset;

  /* Set norm error if any by checking current vs previous gemm outputs */
  error = std::abs(check_norm_error('F', M, N, _ldc, fp, fc));

  /* Copy current device gemm output to host previous gemm output memory */
  if (hipMemcpy(hpo, dout, sizeof(T) * size, hipMemcpyDeviceToHost) != hipSuccess)
    return false;

  return true;
}

/**
 * Check gemm output for accuracy (GPU output vs CPU output).
 * @param[in] dout Device (GPU) matrix output.
 * @param[in] size No of elements in matrix output.
 * @param[out] error Relative accuracy error.
 */
template <typename T>
bool rvs_blas::check_result_accuracy(void * dout, size_t size, double &error) {

  int a_stride_1 = 1,
      a_stride_2 = blas_lda_offset,
      b_stride_1 = 1,
      b_stride_2 = blas_ldb_offset;

  if(transa == rocblas_operation_transpose) {
    a_stride_1 = blas_lda_offset;
    a_stride_2 = 1;
  }

  if(transb == rocblas_operation_transpose) {
    b_stride_1 = blas_ldb_offset;
    b_stride_2 = 1;
  }

  /* Allocate host memory for host (CPU) gemm output */
  if(!hout) {
    if(hipHostMalloc(&hout, size * sizeof(T), 0) != hipSuccess)
      return false;

    if (hipMemset(hout, 0, size * sizeof(T)) != hipSuccess)
      return false;
  }

  /* Allocate host memory for device (GPU) gemm output */
  if(!hdout) {
    if(hipHostMalloc(&hdout, size * sizeof(T), 0) != hipSuccess)
      return false;

    if (hipMemset(hdout, 0, size * sizeof(T)) != hipSuccess)
      return false;
  }

  T * _ha;
  T * _hb;
  T * _hc;
  T alpha = (T) blas_alpha_val;
  T beta = (T) blas_beta_val;

  if (std::is_same<T, float>{}) {
    _ha = (T *)ha;
    _hb = (T *)hb;
    _hc = (T *)hc;
  }
  else {
    _ha = (T *)ha;
    _hb = (T *)hb;
    _hc = (T *)hc;
  }

  /* Copy Matrix C to host gemm output memory */
  if(hipMemcpy(hout, _hc, sizeof(T) * size, hipMemcpyHostToHost) != hipSuccess)
    return false;

  /* Host (CPU) based matrix multiplication */
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
      (T *)hout,
      1,
      blas_ldc_offset);

  /* Copy device gemm output to host memory */
  if (hipMemcpy(hdout, dout, sizeof(T) * size, hipMemcpyDeviceToHost) != hipSuccess)
    return false;

  /* If error injection is enabled, insert error in gemm output */
  if(error_freq && error_count && check_count) {

    /* Insert error at set error frequency */
    if(check_count%error_freq == 0) {

      if(error_count <= size) {

        if (hipMemset(hdout, 0,  sizeof(T) * error_count) != hipSuccess)
          return false;
      }
    }
  }

  /* Calculate max. relative error */

  T max_relative_error = 0.0;

  for(size_t i = 0; i < size; i++)
  {
    T relative_error = (((T *)hout)[i] - ((T *)hdout)[i]) / ((T *)hout)[i];

    relative_error = relative_error > 0 ? relative_error : -relative_error;

    max_relative_error
      = relative_error < max_relative_error ? max_relative_error : relative_error;
  }

  T eps = std::numeric_limits<T>::epsilon();
  T tolerance = 10;

  /* Set error if max. relative error greater than tolerance level */
  if(max_relative_error > eps * tolerance)
  {
    error = max_relative_error;
  }

  return true;
}

/**
 * Validate gemm output for consistency and accuracy.
 * @param[in] self_check Enable self checking of gemm outputs (previous vs current).
 * @param[in] accu_check Enable accuracy checking of gemm outputs (GPU vs CPU).
 * @param[out] self_error Relative F-norm self error.
 * @param[out] accu_error Relative accuracy error.
 */
bool rvs_blas::validate_gemm(bool self_check, bool accu_check, double &self_error, double &accu_error) {

  /* Gemm output checked for consistency/repeatability
     by comparing current output with previous output */
  if(self_check) {

    if(ops_type == "sgemm") {
      check_result_consistency<float>(dc, size_c, self_error);
    }
    else if(ops_type == "dgemm") {
      check_result_consistency<double>(dc, size_c, self_error);
    }
    else if(data_type == "fp8_r") {
      check_result_consistency<rocblas_f8>(dd, size_d, self_error);
    }
    else if(data_type == "fp16_r") {
      check_result_consistency<rocblas_half>(dd, size_d, self_error);
    }
    else if(data_type == "bf16_r") {
      check_result_consistency<rocblas_bfloat16>(dd, size_d, self_error);
    }
    else {
      return false;
    }
  }

  /* Gemm output checked for accuracy/correctness by comparing
     host(CPU) output with device(GPU) output */
  if(accu_check) {

    if(ops_type == "sgemm") {
      check_result_accuracy<float>(dc, size_c, accu_error);
    }
    else if(ops_type == "dgemm") {
      check_result_accuracy<double>(dc, size_c, accu_error);
    }
    else {
      return false;
    }
  }

  /* Error injection is enabled */
  if(error_freq && error_count) {
    /* Increment the gemm check counter */
    check_count++;
  }

  return true;
}

/**
 * Set gemm error stimulation parameters.
 * Note: This function is meant only for test purpose !!!
 * @param[in] _error_freq gemm calls per error injection.
 * @param[in] _error_count no. of errors injected in gemm result.
 */
void rvs_blas::set_gemm_error(uint64_t _error_freq, uint64_t _error_count) {

  error_freq = _error_freq;
  error_count = _error_count;
}

