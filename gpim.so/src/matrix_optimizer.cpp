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
 #include "include/matrix_optimizer.h"
 #include "include/gpu_capability_detector.h"
 #include <hip/hip_runtime.h>
 #include <cmath>
 #include <sstream>
 #include <algorithm>
 #include <cstring>
 #include <iomanip>
 
 namespace gpim {
 
 MatrixOptimizer::GPUArchInfo MatrixOptimizer::detect_architecture(uint16_t gpu_id) {
   GPUArchInfo arch;
 
   hipDeviceProp_t prop;
   if (hipGetDeviceProperties(&prop, gpu_id) != hipSuccess) {
     return arch;
   }
 
   // Get CU count
   arch.compute_units = prop.multiProcessorCount;
 
   // Use capability detector for programmatic detection
   std::string gcn_arch_name(prop.gcnArchName);
   auto arch_class = GPUCapabilityDetector::detect_architecture_class(gcn_arch_name);
   uint32_t cdna_gen = GPUCapabilityDetector::get_cdna_generation(gcn_arch_name);
   auto isa_features = GPUCapabilityDetector::detect_isa_features(gpu_id);
 
   // Detect SIMD width programmatically
   arch.simd_width = GPUCapabilityDetector::detect_wavefront_size(gpu_id);
 
   // Detect wavefronts per CU based on architecture
   arch.wavefronts_per_cu = GPUCapabilityDetector::estimate_wavefronts_per_cu(
       arch_class, cdna_gen);
 
   // Detect MFMA support programmatically
   arch.has_mfma_instructions = isa_features.has_mfma;
   arch.matrix_cores_per_cu = GPUCapabilityDetector::detect_matrix_cores_per_cu(isa_features);
 
   // Calculate theoretical memory bandwidth from HIP properties
   arch.memory_bandwidth_bps = GPUCapabilityDetector::calculate_theoretical_bandwidth(
       prop.memoryClockRate, prop.memoryBusWidth);
 
   // Debug: Print detected capabilities (can be disabled in production)
   #ifdef GPIM_DEBUG_ARCH_DETECTION
   std::cerr << "[MatrixOptimizer] GPU Name: " << prop.name << std::endl;
   std::cerr << "[MatrixOptimizer] Arch Name: " << gcn_arch_name << std::endl;
   std::cerr << "[MatrixOptimizer] Architecture Class: ";
   switch (arch_class) {
     case GPUCapabilityDetector::ArchitectureClass::CDNA:
       std::cerr << "CDNA (gen " << cdna_gen << ")" << std::endl;
       break;
     case GPUCapabilityDetector::ArchitectureClass::RDNA:
       std::cerr << "RDNA" << std::endl;
       break;
     case GPUCapabilityDetector::ArchitectureClass::GCN:
       std::cerr << "GCN" << std::endl;
       break;
     default:
       std::cerr << "UNKNOWN" << std::endl;
   }
   std::cerr << "[MatrixOptimizer] SIMD Width: " << arch.simd_width << std::endl;
   std::cerr << "[MatrixOptimizer] Wavefronts/CU: " << arch.wavefronts_per_cu << std::endl;
   std::cerr << "[MatrixOptimizer] Has MFMA: " << (arch.has_mfma_instructions ? "Yes" : "No") << std::endl;
   std::cerr << "[MatrixOptimizer] Bandwidth: " << (arch.memory_bandwidth_bps / (1024.0*1024*1024)) << " GB/s" << std::endl;
   #endif
 
   // Get memory info
   size_t free_mem, total_mem;
   hipSetDevice(gpu_id);
   if (hipMemGetInfo(&free_mem, &total_mem) == hipSuccess) {
     arch.memory_size_bytes = total_mem;
   }
 
   return arch;
 }
 
 MatrixOptimizer::OptimalMatrixSize MatrixOptimizer::calculate_optimal_gemm_size(
     const GPUArchInfo& arch,
     const std::string& data_type,
     uint32_t tiles_per_wave_target) {
 
   OptimalMatrixSize optimal;
 
   if (arch.compute_units == 0) {
     // Fallback to reasonable default
     optimal.M = optimal.N = optimal.K = 4096;
     optimal.rationale = "Using default size (GPU info unavailable)";
     optimal.expected_efficiency = 0.5;
     return optimal;
   }
 
   // Step 1: Calculate target parallelism
   // Each CU should have multiple wave tiles for good occupancy
   uint64_t total_waves = arch.compute_units * arch.wavefronts_per_cu;
   uint64_t target_tiles = total_waves * tiles_per_wave_target;
 
   // Step 2: Get MFMA tile size and data type size
   uint64_t mfma_tile_size = get_mfma_tile_size(data_type);
   size_t data_size_bytes = get_data_type_size(data_type);
 
   // Step 3: Calculate base matrix size
   // For square matrices: total_tiles = (N / tile_size)^2
   // N = sqrt(target_tiles) * tile_size
   uint64_t base_size = static_cast<uint64_t>(
       std::sqrt(static_cast<double>(target_tiles)) * mfma_tile_size);
 
   // Round to MFMA multiple
   optimal.M = round_to_mfma_multiple(base_size, data_type);
   optimal.N = optimal.M;
   optimal.K = optimal.M;
 
   // Step 4: Check memory constraints
   // 3 matrices: A (M x K), B (K x N), C (M x N)
   uint64_t total_elements = optimal.M * optimal.K +
                            optimal.K * optimal.N +
                            optimal.M * optimal.N;
   uint64_t memory_required = total_elements * data_size_bytes;
 
   // Use 70% of memory to leave headroom for runtime, other allocations
   uint64_t memory_budget = (arch.memory_size_bytes * 7) / 10;
 
   while (memory_required > memory_budget && optimal.M > mfma_tile_size) {
     // Scale down by 10%
     optimal.M = (optimal.M * 9) / 10;
     optimal.N = (optimal.N * 9) / 10;
     optimal.K = (optimal.K * 9) / 10;
 
     optimal.M = round_to_mfma_multiple(optimal.M, data_type);
     optimal.N = round_to_mfma_multiple(optimal.N, data_type);
     optimal.K = round_to_mfma_multiple(optimal.K, data_type);
 
     total_elements = optimal.M * optimal.K + optimal.K * optimal.N +
                     optimal.M * optimal.N;
     memory_required = total_elements * data_size_bytes;
   }
 
   // Step 5: Calculate expected efficiency
   // Efficiency based on tiles per wave:
   // - 1 tile/wave = ~60% utilization
   // - 2 tiles/wave = ~80% utilization
   // - 4 tiles/wave = ~95% utilization (target)
   // - 8+ tiles/wave = ~98% utilization (diminishing returns)
   uint64_t actual_tiles = (optimal.M / mfma_tile_size) *
                          (optimal.N / mfma_tile_size);
   double tiles_per_wave = static_cast<double>(actual_tiles) / total_waves;
 
   // Store diagnostic info
   optimal.actual_tiles = actual_tiles;
   optimal.tiles_per_wave = tiles_per_wave;
 
   // Non-linear efficiency curve (based on empirical GPU utilization)
   if (tiles_per_wave >= 4.0) {
     optimal.expected_efficiency = 0.95 + std::min(0.05, (tiles_per_wave - 4.0) * 0.01);
   } else if (tiles_per_wave >= 2.0) {
     // Linear interpolation: 2 tiles = 80%, 4 tiles = 95%
     optimal.expected_efficiency = 0.80 + (tiles_per_wave - 2.0) * 0.075;
   } else if (tiles_per_wave >= 1.0) {
     // Linear interpolation: 1 tile = 60%, 2 tiles = 80%
     optimal.expected_efficiency = 0.60 + (tiles_per_wave - 1.0) * 0.20;
   } else {
     // Undersaturated: < 1 tile per wave
     optimal.expected_efficiency = tiles_per_wave * 0.60;
   }
 
   // Step 6: Generate rationale
   std::ostringstream oss;
   oss << "Optimized for " << arch.compute_units << " CUs with "
       << arch.wavefronts_per_cu << " wf/CU (" << total_waves << " total waves), "
       << "targeting " << tiles_per_wave_target << " tiles/wave, "
       << "MFMA " << mfma_tile_size << "x" << mfma_tile_size << " tiles, "
       << "achieved " << std::fixed << std::setprecision(2) << tiles_per_wave << " tiles/wave, "
       << static_cast<int>(optimal.expected_efficiency * 100) << "% efficiency";
   optimal.rationale = oss.str();
 
   return optimal;
 }
 
 size_t MatrixOptimizer::calculate_optimal_membw_size(const GPUArchInfo& arch) {
   // For memory bandwidth, we want to saturate all memory channels
   // Use large buffer to ensure bandwidth saturation
 
   // Target: 512 MB - 1 GB depending on GPU memory
   size_t target_size = 512 * 1024 * 1024;  // 512 MB baseline
 
   if (arch.memory_size_bytes > 64ULL * 1024 * 1024 * 1024) {
     // Large memory GPU (>64GB), use 1GB buffer
     target_size = 1024 * 1024 * 1024;
   } else if (arch.memory_size_bytes < 16ULL * 1024 * 1024 * 1024) {
     // Small memory GPU (<16GB), use 256MB buffer
     target_size = 256 * 1024 * 1024;
   }
 
   return target_size;
 }
 
 uint64_t MatrixOptimizer::round_to_mfma_multiple(uint64_t size,
                                                 const std::string& data_type) {
   uint64_t mfma_size = get_mfma_tile_size(data_type);
   return ((size + mfma_size - 1) / mfma_size) * mfma_size;
 }
 
 bool MatrixOptimizer::fits_in_memory(uint64_t M, uint64_t N, uint64_t K,
                                     uint64_t memory_size,
                                     const std::string& data_type) {
   size_t data_size = get_data_type_size(data_type);
   uint64_t total_elements = M * K + K * N + M * N;
   uint64_t required_bytes = total_elements * data_size;
 
   // Use 70% threshold
   return required_bytes <= (memory_size * 7 / 10);
 }
 
 size_t MatrixOptimizer::get_data_type_size(const std::string& data_type) {
   if (data_type == "fp32") {
     return 4;
   } else if (data_type == "fp16" || data_type == "bf16") {
     return 2;
   } else if (data_type == "fp8") {
     return 1;
   } else {
     return 4;  // Default to FP32
   }
 }
 
 uint64_t MatrixOptimizer::get_mfma_tile_size(const std::string& data_type) {
   if (data_type == "fp32") {
     return 16;  // MFMA 16x16 for FP32
   } else if (data_type == "fp16" || data_type == "bf16" || data_type == "fp8") {
     return 32;  // MFMA 32x32 for FP16/BF16/FP8
   } else {
     return 16;  // Default to FP32 tile size
   }
 }
 
 // ============================================================================
 // Complete GEMM Configuration
 // ============================================================================
 
 std::string MatrixOptimizer::get_recommended_data_type(uint16_t gpu_id) {
   // Detect ISA features
   auto isa_features = GPUCapabilityDetector::detect_isa_features(gpu_id);
 
   // Get architecture info for generation detection
   hipDeviceProp_t prop;
   if (hipGetDeviceProperties(&prop, gpu_id) != hipSuccess) {
     return "fp32";  // Safe fallback
   }
 
   std::string gcn_arch_name(prop.gcnArchName);
   uint32_t cdna_gen = GPUCapabilityDetector::get_cdna_generation(gcn_arch_name);
 
   // Recommend based on ISA features and generation
   if (isa_features.has_fp8 && cdna_gen >= 3) {
     // CDNA3+ (MI300, MI350 series): FP8 gives best perf/watt
     // ~2x throughput vs FP16 on same hardware
     return "fp8";
   } else if (isa_features.has_bfloat16 && cdna_gen >= 2) {
     // CDNA2+ (MI250X, MI210): BF16 good balance of perf and accuracy
     // Better than FP16 for deep learning (wider dynamic range)
     return "bf16";
   } else if (isa_features.has_packed_fp16) {
     // CDNA1, RDNA: FP16 available and faster than FP32
     return "fp16";
   } else {
     // Fallback: FP32 (universally supported)
     return "fp32";
   }
 }
 
 MatrixOptimizer::GEMMConfig MatrixOptimizer::get_optimal_gemm_config(
     uint16_t gpu_id,
     bool prefer_batched) {
 
   GEMMConfig config;
 
   // Step 1: Detect architecture
   GPUArchInfo arch = detect_architecture(gpu_id);
   auto isa_features = GPUCapabilityDetector::detect_isa_features(gpu_id);
 
   // Step 2: Determine optimal data types for hipBLASLt
   // hipBLASLt supports mixed precision: inputs can be lower precision,
   // compute in higher precision, output in medium precision
   std::string recommended_type = get_recommended_data_type(gpu_id);
 
   if (recommended_type == "fp8") {
     // FP8 GEMM: FP8 inputs, FP32 compute, FP16 output (best accuracy)
     config.input_type_a = "fp8";
     config.input_type_b = "fp8";
     config.output_type_c = "fp16";  // Accumulator
     config.output_type_d = "fp16";  // Result
     config.compute_type = "fp32";   // High-precision accumulation
     config.element_size_a = 1;
     config.element_size_b = 1;
     config.element_size_c = 2;
     config.element_size_d = 2;
   } else if (recommended_type == "bf16") {
     // BF16 GEMM: BF16 inputs, FP32 compute, BF16 output
     config.input_type_a = "bf16";
     config.input_type_b = "bf16";
     config.output_type_c = "bf16";
     config.output_type_d = "bf16";
     config.compute_type = "fp32";
     config.element_size_a = 2;
     config.element_size_b = 2;
     config.element_size_c = 2;
     config.element_size_d = 2;
   } else if (recommended_type == "fp16") {
     // FP16 GEMM: FP16 inputs, FP32 compute, FP16 output
     config.input_type_a = "fp16";
     config.input_type_b = "fp16";
     config.output_type_c = "fp16";
     config.output_type_d = "fp16";
     config.compute_type = "fp32";
     config.element_size_a = 2;
     config.element_size_b = 2;
     config.element_size_c = 2;
     config.element_size_d = 2;
   } else {  // fp32
     // FP32 GEMM: FP32 everywhere
     config.input_type_a = "fp32";
     config.input_type_b = "fp32";
     config.output_type_c = "fp32";
     config.output_type_d = "fp32";
     config.compute_type = "fp32";
     config.element_size_a = 4;
     config.element_size_b = 4;
     config.element_size_c = 4;
     config.element_size_d = 4;
   }
 
   // Step 3: Calculate optimal matrix size
   OptimalMatrixSize opt_size = calculate_optimal_gemm_size(arch, recommended_type);
   config.M = opt_size.M;
   config.N = opt_size.N;
   config.K = opt_size.K;
   config.expected_efficiency = opt_size.expected_efficiency;
   config.rationale = opt_size.rationale;
 
   // Step 4: Set memory layout (hipBLASLt prefers row-major)
   // Row-major layout (recommended for hipBLASLt):
   // - Better cache locality for row-major C/C++ arrays
   // - A: M×K, lda = K (elements per row)
   // - B: K×N, ldb = N (elements per row)
   // - C/D: M×N, ldc = ldd = N (elements per row)
   config.row_major = true;
   config.transa = 'N';  // No transpose
   config.transb = 'N';  // No transpose
   config.lda = config.K;   // Row-major: lda = K
   config.ldb = config.N;   // Row-major: ldb = N
   config.ldc = config.N;   // Row-major: ldc = N
   config.ldd = config.N;   // Row-major: ldd = N
 
   // Step 5: Set alpha/beta
   // hipBLASLt: D = alpha * (A * B) + beta * C
   config.alpha = 1.0;
   config.beta = 0.0;  // Pure multiply (no accumulation)
 
   // Step 6: hipBLASLt specific features
   config.use_autotune = true;  // Enable solution auto-tuning
   config.max_workspace_bytes = 256 * 1024 * 1024;  // 256 MB workspace for tuning
 
   // Fused operations (epilogue fusion)
   // For now, disabled - can be enabled for specific use cases
   config.activation_relu = false;
   config.activation_gelu = false;
   config.bias_enabled = false;
   config.epilogue = "none";
 
   // Step 7: Batch configuration
   if (prefer_batched) {
     // Batched GEMM: Process multiple independent GEMMs
     // Batch count should utilize multiple CUs
     config.batch_count = std::min(arch.compute_units / 4, uint32_t(64));
     config.stride_a = config.M * config.K;
     config.stride_b = config.K * config.N;
     config.stride_c = config.M * config.N;
     config.stride_d = config.M * config.N;
 
     // Update rationale
     config.rationale += " | Batched with " +
                         std::to_string(config.batch_count) + " batches";
   } else {
     config.batch_count = 1;
     config.stride_a = 0;
     config.stride_b = 0;
     config.stride_c = 0;
     config.stride_d = 0;
   }
 
   // Step 8: Estimate TFLOPS
   // FLOPS = 2 * M * N * K * batch_count (2 for multiply-add)
   uint64_t total_ops = 2ULL * config.M * config.N * config.K * config.batch_count;
 
   // Estimate peak TFLOPS based on architecture and data type
   double peak_tflops_per_cu = 0.0;
   if (config.input_type_a == "fp8") {
     peak_tflops_per_cu = 0.512;  // ~512 GFLOPS per CU for FP8 MFMA on CDNA3+
   } else if (config.input_type_a == "fp16" || config.input_type_a == "bf16") {
     peak_tflops_per_cu = 0.256;  // ~256 GFLOPS per CU for FP16/BF16 MFMA
   } else {  // fp32
     peak_tflops_per_cu = 0.064;  // ~64 GFLOPS per CU for FP32 MFMA
   }
 
   double peak_tflops = peak_tflops_per_cu * arch.compute_units;
   config.expected_tflops = peak_tflops * config.expected_efficiency;
 
   return config;
 }
 
 } // namespace gpim
 