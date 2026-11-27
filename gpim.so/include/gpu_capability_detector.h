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
 #ifndef GPIM_SO_INCLUDE_GPU_CAPABILITY_DETECTOR_H_
 #define GPIM_SO_INCLUDE_GPU_CAPABILITY_DETECTOR_H_
 
 #include <cstdint>
 #include <string>
 
 namespace gpim {
 
 /**
  * @class GPUCapabilityDetector
  * @brief Programmatically detect GPU capabilities without hardcoding model names
  *
  * Instead of hardcoding "MI300X", "MI355X", etc., we query:
  * 1. ISA features (MFMA, dot product instructions)
  * 2. Architecture ID from gcnArchName (gfx9xx)
  * 3. HIP device properties (warpSize, CU count, memory)
  * 4. HSA agent info (wavefront size, ISA features)
  */
 class GPUCapabilityDetector {
  public:
   /**
    * @struct ISAFeatures
    * @brief Instruction Set Architecture features
    */
   struct ISAFeatures {
     bool has_mfma;              //!< Matrix Fused Multiply-Add instructions
     bool has_dot_product;       //!< Dot product instructions (dp4a, etc.)
     bool has_packed_fp16;       //!< Packed FP16 math
     bool has_bfloat16;          //!< BF16 support
     bool has_fp8;               //!< FP8 support (CDNA3+)
     bool has_xdlops;            //!< Extended Deep Learning Ops (CDNA precursor to MFMA)
     uint32_t mfma_tile_size;    //!< MFMA tile size (16, 32, etc.)
 
     ISAFeatures() : has_mfma(false), has_dot_product(false),
                     has_packed_fp16(false), has_bfloat16(false),
                     has_fp8(false), has_xdlops(false), mfma_tile_size(0) {}
   };
 
   /**
    * @struct ArchitectureClass
    * @brief High-level architecture classification
    */
   enum class ArchitectureClass {
     CDNA,      //!< Data center GPU (MI100, MI200, MI300 series)
     RDNA,      //!< Consumer/gaming GPU
     GCN,       //!< Legacy Graphics Core Next
     UNKNOWN
   };
 
   /**
    * @brief Detect ISA features using HSA runtime
    * @param gpu_id HIP device ID
    * @return ISA features structure
    */
   static ISAFeatures detect_isa_features(uint16_t gpu_id);
 
   /**
    * @brief Detect architecture class from gcnArchName
    * @param gcn_arch_name Architecture name (e.g., "gfx942")
    * @return Architecture class
    */
   static ArchitectureClass detect_architecture_class(const std::string& gcn_arch_name);
 
   /**
    * @brief Get CDNA generation from architecture name
    * @param gcn_arch_name Architecture name
    * @return CDNA generation (1, 2, 3, 4, etc.) or 0 if not CDNA
    */
   static uint32_t get_cdna_generation(const std::string& gcn_arch_name);
 
   /**
    * @brief Detect wavefront size programmatically
    * @param gpu_id HIP device ID
    * @return Wavefront size (32, 64, etc.)
    */
   static uint32_t detect_wavefront_size(uint16_t gpu_id);
 
   /**
    * @brief Estimate wavefronts per CU based on architecture
    * @param arch_class Architecture class
    * @param cdna_gen CDNA generation (if applicable)
    * @return Estimated max wavefronts per CU
    */
   static uint32_t estimate_wavefronts_per_cu(ArchitectureClass arch_class,
                                              uint32_t cdna_gen);
 
   /**
    * @brief Calculate theoretical memory bandwidth from HIP properties
    * @param memory_clock_rate Memory clock rate in kHz
    * @param memory_bus_width Memory bus width in bits
    * @return Theoretical bandwidth in bytes/second
    */
   static uint64_t calculate_theoretical_bandwidth(uint32_t memory_clock_rate,
                                                   uint32_t memory_bus_width);
 
   /**
    * @brief Detect if GPU has matrix core hardware
    * @param isa_features ISA features
    * @return Number of matrix cores per CU (0 if none)
    */
   static uint32_t detect_matrix_cores_per_cu(const ISAFeatures& isa_features);
 
  private:
   /**
    * @brief Query HSA agent for ISA name
    * @param gpu_id HIP device ID
    * @return ISA name string (e.g., "amdgcn-amd-amdhsa--gfx942")
    */
   static std::string get_hsa_isa_name(uint16_t gpu_id);
 
   /**
    * @brief Parse ISA features from ISA name
    * @param isa_name Full ISA name string
    * @return Parsed features
    */
   static ISAFeatures parse_isa_name(const std::string& isa_name);
 
   /**
    * @brief Check for specific instruction support via compilation test
    * @param instruction_name Instruction to test (e.g., "v_mfma_f32_16x16x16f16")
    * @return true if supported
    */
   static bool check_instruction_support(const std::string& instruction_name);
 };
 
 } // namespace gpim
 
 #endif  // GPIM_SO_INCLUDE_GPU_CAPABILITY_DETECTOR_H_
 