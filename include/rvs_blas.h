/********************************************************************************
 *
 * Copyright (c) 2018-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <map>
#include <vector>
using std::map;

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
       std::string _blas_source, std::string _compute_type, std::string _out_data_type,
       std::string _scale_a, std::string _scale_b, uint32_t rotating, uint64_t _hot_calls);
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
    template <typename Ti, typename To> bool copy_data_to_gpu(void);
    bool run_blas_gemm(uint64_t num_calls);
    bool is_gemm_op_complete(void);
    bool validate_gemm(bool self_check, bool accu_check, double &self_error, double &accu_error);
    void set_gemm_error(uint64_t _error_freq, uint64_t _error_count);

    /**
     * @brief Override the matrix generation seed so that multiple rvs_blas
     *        instances (one per GPU) produce identical input matrices.
     *
     * Must be called BEFORE generate_random_matrix_data().
     *
     * For hiprand mode this calls hiprandSetPseudoRandomGeneratorSeed().
     * For CPU-side modes this replaces the time(NULL) seed used in
     * generate_random_matrix_data().
     *
     * @param seed  Deterministic seed value shared across all workers.
     */
    void set_matrix_seed(uint64_t seed);

    /**
     * @brief Compute CRC-32 over the GEMM output buffer and compare against
     *        the stored reference.  On the first call the CRC is saved as the
     *        reference and 0 (no mismatch) is returned.  On every subsequent
     *        call a non-zero value is returned when the CRC differs from the
     *        reference, signalling a potential silent data corruption event.
     * @return 0  – output matches reference (or this is the first call)
     *         1  – CRC mismatch detected (SDC candidate)
     *        -1  – internal error (e.g. GPU→host copy failed)
     */
    int compute_output_crc();

    //! Returns the CRC-32 value computed during the most recent
    //! compute_output_crc() call.  Valid only when compute_output_crc()
    //! has been called at least once (i.e. crc_ref_valid is true).
    uint32_t get_last_output_crc() const { return last_crc; }

    //! Returns a single 32-bit digest that represents the ordered sequence of
    //! every per-iteration CRC produced so far.  Each call to
    //! compute_output_crc() chains the iteration's CRC-32 value into this
    //! accumulator, so the digest is sensitive to both the value and the
    //! position of any individual iteration's output.  Compare this across
    //! GPUs after a full run to detect any transient or sustained SDC event.
    uint32_t get_run_crc_digest() const {
      return accumulated_crc ^ 0xFFFFFFFFu;
    }

    //! Number of per-iteration CRC values recorded so far.
    size_t get_crc_iter_count() const { return crc_per_iter.size(); }

    //! Compute the run digest over only the first @p n iterations.
    //! Used to normalize cross-GPU digest comparison when GPUs complete
    //! different numbers of iterations (sequential mode, time-based duration).
    uint32_t compute_digest_for_n_iters(size_t n) const;

    //! Called only when compute_output_crc() returns 1 (mismatch).
    //! Fills:
    //!   damaged_bytes  – number of raw bytes that differ from the reference
    //!   total_bytes    – total bytes in the compared buffer
    //!   fnorm          – relative Frobenius norm ||current-ref||_F / ||ref||_F,
    //!                    or -1.0 when the element type is not supported for norm.
    //! Both hcrc_buf (current) and ref_buf (reference snapshot from iteration 1)
    //! must be valid — only safe to call after at least two compute_output_crc() calls.
    void compute_mismatch_magnitude(uint32_t &damaged_bytes,
                                    uint32_t &total_bytes_out,
                                    double   &fnorm) const;

    /**
     * @brief Copy the current host matrix bytes into caller-owned vectors.
     *
     * Must be called after generate_random_matrix_data() so that ha/hb/hc
     * are populated.  Used by the first GSTWorker to snapshot the shared
     * matrix pool for all subsequent workers.
     */
    void get_host_matrix_bytes(std::vector<uint8_t> &a,
                               std::vector<uint8_t> &b,
                               std::vector<uint8_t> &c) const;

    /**
     * @brief Overwrite ha/hb/hc from caller-provided byte vectors.
     *
     * Used by subsequent GSTWorkers instead of generate_random_matrix_data()
     * to ensure byte-identical input matrices across all GPUs.
    /**
     * Inject previously-snapshotted matrix bytes into this instance's host
     * buffers.  Sizes must match what was allocated for this instance — ensured
     * by s_host_matrix_byte_sizes() mirroring allocate_host_matrix_mem().
     */
    void inject_host_matrix_data(const std::vector<uint8_t> &a,
                                 const std::vector<uint8_t> &b,
                                 const std::vector<uint8_t> &c);

    /**
     * @brief Compute CRC-32 over each of the three host input matrices.
     *
     * Uses the same CRC-32/ISO-HDLC polynomial as compute_output_crc().
     * Returns zeros when host buffers are not allocated (e.g. hiprand mode).
     * Intended for diagnostic logging to confirm all GPUs received identical
     * input data.
     */
    void get_host_matrix_input_crcs(uint32_t &crc_a,
                                    uint32_t &crc_b,
                                    uint32_t &crc_c) const;

    //! CRC-32 over the scale matrices hsa and hsb.
    //! Returns zero for each when the corresponding buffer is null.
    void get_host_scale_crcs(uint32_t &crc_sa, uint32_t &crc_sb) const;

    //! Copy hsa/hsb into caller-owned vectors (empty when not allocated).
    void get_host_scale_bytes(std::vector<uint8_t> &sa,
                              std::vector<uint8_t> &sb) const;

    //! Overwrite hsa/hsb from caller-provided byte vectors.
    void inject_host_scale_data(const std::vector<uint8_t> &sa,
                                const std::vector<uint8_t> &sb);

    bool set_callback(rvsBlasCallback_t callback, void *user_data);

    static void hip_stream_callback (hipStream_t stream, hipError_t status, void *user_data);

    rvsBlasCallback_t callback;
    void * user_data;

 protected:
    //! GPU device index
    int gpu_device_index;
    //! Type of operation
    std::string ops_type;
    //! Type of input data
    std::string data_type;
    //! Type of output data
    std::string out_data_type;
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

    //Data type Declaration
    //! pointer to device (GPU) memory
    void *da;
    //! pointer to device (GPU) memory
    void *db;
    //! pointer to device (GPU) memory
    void *dc;
    //! pointer to device (GPU) memory
    void *dd;
    //! pointer to host memory
    void *ha;
    //! pointer to host memory
    void *hb;
    //! pointer to host memory
    void *hc;

    //! pointer to device scale A memory
    void *dsa;
    //! pointer to device scale B memory
    void *dsb;
    //! pointer to host scale A memory
    void *hsa;
    //! pointer to host scale B memory
    void *hsb;

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

    // Matrix generation seed (0 = use time(NULL) / library default)
    uint64_t matrix_seed;

    // CRC-based SDC detection state
    //! Reference CRC-32 captured from the first GEMM output
    uint32_t crc_ref;
    //! CRC-32 from the most recent compute_output_crc() call
    uint32_t last_crc;
    //! Running CRC-32 accumulator (pre-finalized state) built by chaining every
    //! per-iteration CRC into it.  Finalized value is exposed via
    //! get_run_crc_digest().  Initialized to 0xFFFFFFFFu (standard pre-condition).
    uint32_t accumulated_crc;
    //! True once the reference CRC has been stored
    bool     crc_ref_valid;
    //! Per-iteration CRC history: crc_per_iter[i] is the raw CRC-32 of iteration i.
    //! Used by compute_digest_for_n_iters() to normalize cross-GPU comparisons.
    std::vector<uint32_t> crc_per_iter;
    //! Scratch host buffer used to copy GPU output for CRC computation
    void    *hcrc_buf;
    //! Byte length of hcrc_buf (tracks reallocation need)
    size_t   hcrc_buf_bytes;
    //! Host-pinned snapshot of the iteration-1 output (the reference output).
    //! Allocated and filled on the first compute_output_crc() call, then frozen.
    //! Used by compute_mismatch_magnitude() for byte-diff and Frobenius norm.
    void    *ref_buf;
    size_t   ref_buf_bytes;
    //! Byte size of the last CRC computation region — set in compute_output_crc(),
    //! read by compute_mismatch_magnitude() for the byte-diff loop.
    size_t   crc_total_bytes;

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

    //! Matrix A rows
    uint64_t hbl_row_a;
    //! Matrix A columns
    uint64_t hbl_col_a;
    //! Matrix B row
    uint64_t hbl_row_b;
    //! Matrix B columns
    uint64_t hbl_col_b;

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

    //! hipblaslt matrix output data-type
    hipDataType hbl_out_datatype;

    //! hipblaslt compute-type
    hipblasComputeType_t hbl_computetype;

    //! Create hipblaslt matrix multiply descriptor
    std::vector <hipblasLtMatmulDesc_t> hbl_matmul;

    uint64_t block_count;

    //! Transpose matrix A
    hipblasOperation_t hbl_trans_a;
    //! Transpose matrix B
    hipblasOperation_t hbl_trans_b;

    //! Scale matrix A
    std::string hbl_scale_a;
    //! Scale matrix B
    std::string hbl_scale_b;
    //! Scale matrix A size
    size_t hbl_scale_a_size;
    //! Scale matrix B size
    size_t hbl_scale_b_size;
    //! Scale matrix A block row size
    const size_t hbl_scale_a_block_row = 32;
    //! Scale matrix A block column size
    const size_t hbl_scale_a_block_col = 1;
    //! Scale matrix B block row size
    const size_t hbl_scale_b_block_row = 1;
    //! Scale matrix B block column size
    const size_t hbl_scale_b_block_col = 32;

    //! Workspace buffer for matrix multiplication
    void* hbl_workspace;

    //! hipblaslt matrix A leading dimension
    int64_t hbl_lda_offset;
    //! hipblaslt matrix B leading dimension
    int64_t hbl_ldb_offset;
    //! hipblaslt matrix C leading dimension
    int64_t hbl_ldc_offset;
    //! hipblaslt matrix D leading dimension
    int64_t hbl_ldd_offset;

    //! hipblaslt heuristic algorithm result
    hipblasLtMatmulHeuristicResult_t hbl_heuristic_result;

    //! number of gemm calls at once
    uint64_t hot_calls;

    bool init_gpu_device(void);
    bool allocate_gpu_matrix_mem(void);
    template <typename Ti, typename To> bool allocate_gpu_matrix_mem(void);
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
        (datatype == "fp4_r")      ? HIP_R_4F_E2M1 :
        (datatype == "fp6_e3m2_r") ? HIP_R_6F_E3M2 :
        (datatype == "fp6_e2m3_r") ? HIP_R_6F_E2M3 :
        (datatype == "i8_r")       ? HIP_R_8I  :
        (datatype == "fp8_r")      ? HIP_R_8F_E4M3_FNUZ : // FP8-FNUZ
        (datatype == "fp8_e4m3_r" || datatype == "mxfp8_e4m3_r") ? HIP_R_8F_E4M3  : // FP8-OCP E4M3
        (datatype == "fp8_e5m2_r" || datatype == "mxfp8_e5m2_r") ? HIP_R_8F_E5M2  : // FP8-OCP E5M2
        (datatype == "bf16_r")     ? HIP_R_16BF :
        (datatype == "fp16_r")     ? HIP_R_16F  :
        (datatype == "fp32_r")     ? HIP_R_32F  :
        (datatype == "fp64_r")     ? HIP_R_64F  :
        RVS_BLAS_HIP_DATATYPE_INVALID;
    }


    hipblasComputeType_t computetype_to_hipblas_computetype(const std::string& computetype)
    {
      return
        computetype == "fp16_r" ? HIPBLAS_COMPUTE_16F  :
        computetype == "fp32_r" ? HIPBLAS_COMPUTE_32F  :
        computetype == "xf32_r" ? HIPBLAS_COMPUTE_32F_FAST_TF32 :
        computetype == "fp64_r" ? HIPBLAS_COMPUTE_64F :
        computetype == "i32_r"  ? HIPBLAS_COMPUTE_32I :
        RVS_BLAS_HIPBLAS_COMPUTETYPE_INVALID;
    }

    inline size_t get_hipdatatype_size(hipDataType hipdatatype)
    {
      static const std::map<hipDataType, size_t> hipdatatype_sizemap {
        {HIP_R_32F, 4},
        {HIP_R_64F, 8},
        {HIP_R_16F, 2},
        {HIP_R_8I, 1},
        {HIP_R_8U, 1},
        {HIP_R_32I, 4},
        {HIP_R_32U, 4},
        {HIP_R_16BF, 2},
        {HIP_R_4I, 1},
        {HIP_R_4U, 1},
        {HIP_R_16I, 2},
        {HIP_R_16U, 2},
        {HIP_R_64I, 8},
        {HIP_R_64U, 8},
        {HIP_R_8F_E4M3_FNUZ, 1},
        {HIP_R_8F_E5M2_FNUZ, 1},
        {HIP_R_8F_E4M3, 1},
        {HIP_R_8F_E5M2, 1},
        {HIP_R_4F_E2M1, 1},
        {HIP_R_6F_E2M3, 1},
        {HIP_R_6F_E3M2, 1}
      };

      return hipdatatype_sizemap.at(hipdatatype);
    }

std::vector<float> generateMXInput(hipDataType            dataType,
                                   void*                  data,
                                   void*                  scale,
                                   int                    row,
                                   int                    col,
                                   int                    stride,
                                   bool                   isTranspose,
                                   int const              scaleBlockRowSize,
                                   int const              scaleBlockColSize,
                                   bool                   isMatrixA,
                                   std::string_view const initMethod = "Bounded",
                                   float                  min_val    = -1.0f,
                                   float                  max_val    = 1.0f);
};

#endif  // INCLUDE_RVS_BLAS_H_
