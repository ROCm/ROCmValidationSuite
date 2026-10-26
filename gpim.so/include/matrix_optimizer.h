/********************************************************************************
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
 #ifndef GPIM_SO_INCLUDE_MATRIX_OPTIMIZER_H_
 #define GPIM_SO_INCLUDE_MATRIX_OPTIMIZER_H_
 
 #include <string>
 #include <cstdint>
 
 namespace gpim {
 
 /**
  * @class MatrixOptimizer
  * @brief GPU-architecture-aware matrix size optimization
  *
  * Calculates optimal matrix sizes for GEMM and memory bandwidth probes
  * based on GPU architecture characteristics:
  * - Compute Unit count
  * - MFMA instruction granularity
  * - Memory capacity and bandwidth
  * - SIMD width and occupancy
  */
 class MatrixOptimizer {
  public:
   /**
    * @struct GPUArchInfo
    * @brief GPU architecture characteristics
    */
   struct GPUArchInfo {
     uint32_t compute_units;
     uint32_t simd_width;            //!< 64 for CDNA
     uint32_t wavefronts_per_cu;
     uint64_t memory_bandwidth_bps;  //!< Bytes per second
     uint64_t memory_size_bytes;
     uint32_t matrix_cores_per_cu;   //!< MI250X: 4 matrix cores per CU
     bool has_mfma_instructions;     //!< Matrix Fused Multiply-Add support
 
     GPUArchInfo() : compute_units(0), simd_width(64), wavefronts_per_cu(32),
                     memory_bandwidth_bps(0), memory_size_bytes(0),
                     matrix_cores_per_cu(0), has_mfma_instructions(false) {}
   };
 
   /**
    * @struct OptimalMatrixSize
    * @brief Optimal matrix dimensions with rationale
    */
   struct OptimalMatrixSize {
     uint64_t M, N, K;               //!< Matrix dimensions (M x K) * (K x N) = (M x N)
     std::string rationale;          //!< Why this size is optimal
     double expected_efficiency;     //!< 0.0-1.0, GPU utilization estimate
     uint64_t actual_tiles;          //!< Actual number of tiles
     double tiles_per_wave;          //!< Tiles per wavefront (diagnostic)
 
     OptimalMatrixSize() : M(0), N(0), K(0), expected_efficiency(0.0),
                           actual_tiles(0), tiles_per_wave(0.0) {}
   };
 
   /**
    * @struct GEMMConfig
    * @brief Complete GEMM configuration for hipBLASLt
    */
   struct GEMMConfig {
     // Matrix dimensions
     uint64_t M, N, K;               //!< Matrix dimensions (M×K) * (K×N) = (M×N)
 
     // Data types (hipBLASLt supports mixed precision)
     std::string input_type_a;       //!< A matrix type: "fp32", "fp16", "bf16", "fp8"
     std::string input_type_b;       //!< B matrix type: "fp32", "fp16", "bf16", "fp8"
     std::string output_type_c;      //!< C matrix type: "fp32", "fp16", "bf16", "fp8"
     std::string output_type_d;      //!< D matrix type (result): "fp32", "fp16", "bf16", "fp8"
     std::string compute_type;       //!< Compute precision: "fp32", "fp16", "bf16"
 
     size_t element_size_a;          //!< Bytes per element for A
     size_t element_size_b;          //!< Bytes per element for B
     size_t element_size_c;          //!< Bytes per element for C
     size_t element_size_d;          //!< Bytes per element for D
 
     // Memory layout (row-major recommended for hipBLASLt)
     uint64_t lda;                   //!< Leading dimension of A
     uint64_t ldb;                   //!< Leading dimension of B
     uint64_t ldc;                   //!< Leading dimension of C
     uint64_t ldd;                   //!< Leading dimension of D (output)
     bool row_major;                 //!< true = row-major (recommended), false = column-major
     char transa;                    //!< Transpose A: 'N' (no), 'T' (yes)
     char transb;                    //!< Transpose B: 'N' (no), 'T' (yes)
 
     // Alpha/Beta (hipBLASLt: D = alpha * (A * B) + beta * C)
     double alpha;
     double beta;
 
     // Batch parameters
     uint64_t batch_count;           //!< Number of matrices in batch
     uint64_t stride_a;              //!< Stride between batches for A
     uint64_t stride_b;              //!< Stride between batches for B
     uint64_t stride_c;              //!< Stride between batches for C
     uint64_t stride_d;              //!< Stride between batches for D
 
     // hipBLASLt specific
     bool use_autotune;              //!< Enable solution auto-tuning
     int max_workspace_bytes;        //!< Max workspace for autotuning (0 = no workspace)
     bool activation_relu;           //!< Fused ReLU activation
     bool activation_gelu;           //!< Fused GELU activation
     bool bias_enabled;              //!< Enable bias vector
     std::string epilogue;           //!< "none", "relu", "gelu", "bias", "bias_relu"
 
     // Performance hints
     double expected_tflops;         //!< Expected TFLOPS for this config
     double expected_efficiency;     //!< GPU utilization estimate (0.0-1.0)
     std::string rationale;          //!< Why this config is optimal
 
     GEMMConfig() : M(0), N(0), K(0),
                    input_type_a("fp16"), input_type_b("fp16"),
                    output_type_c("fp16"), output_type_d("fp16"),
                    compute_type("fp32"),
                    element_size_a(2), element_size_b(2),
                    element_size_c(2), element_size_d(2),
                    lda(0), ldb(0), ldc(0), ldd(0), row_major(true),
                    transa('N'), transb('N'),
                    alpha(1.0), beta(0.0),
                    batch_count(1), stride_a(0), stride_b(0), stride_c(0), stride_d(0),
                    use_autotune(true), max_workspace_bytes(256*1024*1024),
                    activation_relu(false), activation_gelu(false), bias_enabled(false),
                    epilogue("none"),
                    expected_tflops(0.0), expected_efficiency(0.0) {}
   };
 
   /**
    * @brief Detect GPU architecture information
    * @param gpu_id HIP device ID
    * @return Architecture characteristics
    */
   static GPUArchInfo detect_architecture(uint16_t gpu_id);
 
   /**
    * @brief Calculate optimal matrix size for GEMM probe
    *
    * Considers:
    * - CU count for parallelism
    * - MFMA instruction granularity (16x16 for FP32, 32x32 for FP16/BF16)
    * - Memory capacity for avoiding spills
    * - Memory bandwidth saturation
    *
    * @param arch GPU architecture info
    * @param data_type "fp32", "fp16", "bf16", "fp8"
    * @param tiles_per_wave_target Target tiles per wavefront (default: 4 for 95% efficiency)
    *                              1 = 60% efficiency, 2 = 80%, 4 = 95%, 8 = 98%
    * @return Optimal matrix dimensions
    */
   static OptimalMatrixSize calculate_optimal_gemm_size(
       const GPUArchInfo& arch,
       const std::string& data_type,
       uint32_t tiles_per_wave_target = 4);
 
   /**
    * @brief Calculate optimal size for memory bandwidth probe
    * @param arch GPU architecture info
    * @return Buffer size in bytes
    */
   static size_t calculate_optimal_membw_size(const GPUArchInfo& arch);
 
   /**
    * @brief Get optimal GEMM configuration (complete)
    *
    * This provides everything needed for rocBLAS/hipBLAS calls:
    * - Recommended data type based on ISA features
    * - Optimal matrix dimensions
    * - Leading dimensions (lda, ldb, ldc)
    * - Transpose flags
    * - Alpha/beta values
    * - Batch configuration
    *
    * @param gpu_id HIP device ID
    * @param prefer_batched If true, configure for batched GEMM
    * @return Complete GEMM configuration
    */
   static GEMMConfig get_optimal_gemm_config(uint16_t gpu_id,
                                             bool prefer_batched = false);
 
   /**
    * @brief Get recommended data type for architecture
    *
    * Based on ISA features:
    * - CDNA3+ (gfx940+): FP8 (best perf/watt)
    * - CDNA2+ (gfx90a): BF16 (good perf, wider support)
    * - CDNA1 (gfx908): FP16 or FP32
    * - RDNA: FP16 or FP32
    *
    * @param gpu_id HIP device ID
    * @return Recommended data type string
    */
   static std::string get_recommended_data_type(uint16_t gpu_id);
 
  private:
   /**
    * @brief Round up to nearest multiple of MFMA instruction size
    * MI250X MFMA: 16x16 for FP32, 32x32 for FP16/BF16
    */
   static uint64_t round_to_mfma_multiple(uint64_t size,
                                         const std::string& data_type);
 
   /**
    * @brief Check if matrix fits in GPU memory with headroom
    */
   static bool fits_in_memory(uint64_t M, uint64_t N, uint64_t K,
                             uint64_t memory_size,
                             const std::string& data_type);
 
   /**
    * @brief Get data type size in bytes
    */
   static size_t get_data_type_size(const std::string& data_type);
 
   /**
    * @brief Get MFMA tile size for data type
    */
   static uint64_t get_mfma_tile_size(const std::string& data_type);
 };
 
 } // namespace gpim
 
 #endif  // GPIM_SO_INCLUDE_MATRIX_OPTIMIZER_H_
 