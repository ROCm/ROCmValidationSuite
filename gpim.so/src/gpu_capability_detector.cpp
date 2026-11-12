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
 #include "include/gpu_capability_detector.h"
 #include <hip/hip_runtime.h>
 #include <hsa/hsa.h>
 #include <hsa/hsa_ext_amd.h>
 #include <algorithm>
 #include <cctype>
 #include <cstring>
 
 namespace gpim {
 
 // ============================================================================
 // ISA Feature Detection
 // ============================================================================
 
 GPUCapabilityDetector::ISAFeatures GPUCapabilityDetector::detect_isa_features(
     uint16_t gpu_id) {
 
   ISAFeatures features;
 
   // Get HSA ISA name which contains feature flags
   std::string isa_name = get_hsa_isa_name(gpu_id);
   if (!isa_name.empty()) {
     features = parse_isa_name(isa_name);
   }
 
   // If HSA query failed, fall back to HIP device properties
   if (isa_name.empty()) {
     hipDeviceProp_t prop;
     if (hipGetDeviceProperties(&prop, gpu_id) == hipSuccess) {
       std::string arch_name(prop.gcnArchName);
       uint32_t cdna_gen = get_cdna_generation(arch_name);
 
       if (cdna_gen > 0) {
         // All CDNA architectures have MFMA
         features.has_mfma = true;
         features.has_xdlops = true;
         features.has_packed_fp16 = true;
         features.has_dot_product = true;
 
         // CDNA2+ has BF16
         if (cdna_gen >= 2) {
           features.has_bfloat16 = true;
         }
 
         // CDNA3+ has FP8
         if (cdna_gen >= 3) {
           features.has_fp8 = true;
         }
 
         // MFMA tile size: 16 for FP32, 32 for FP16/BF16
         features.mfma_tile_size = 16;  // Conservative default
       }
     }
   }
 
   return features;
 }
 
 std::string GPUCapabilityDetector::get_hsa_isa_name(uint16_t gpu_id) {
   std::string isa_name;
 
   // Initialize HSA if not already initialized
   static bool hsa_initialized = false;
   if (!hsa_initialized) {
     if (hsa_init() == HSA_STATUS_SUCCESS) {
       hsa_initialized = true;
     } else {
       return "";  // HSA not available
     }
   }
 
   // Find HSA agent corresponding to HIP device
   struct agent_info {
     uint16_t target_gpu_id;
     uint16_t current_gpu_id;
     hsa_agent_t agent;
     bool found;
   } info;
 
   info.target_gpu_id = gpu_id;
   info.current_gpu_id = 0;
   info.found = false;
 
   auto callback = [](hsa_agent_t agent, void* data) -> hsa_status_t {
     agent_info* info = static_cast<agent_info*>(data);
 
     hsa_device_type_t device_type;
     hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
 
     if (device_type == HSA_DEVICE_TYPE_GPU) {
       if (info->current_gpu_id == info->target_gpu_id) {
         info->agent = agent;
         info->found = true;
         return HSA_STATUS_INFO_BREAK;
       }
       info->current_gpu_id++;
     }
 
     return HSA_STATUS_SUCCESS;
   };
 
   hsa_iterate_agents(callback, &info);
 
   if (info.found) {
     // Get ISA from agent
     hsa_isa_t isa;
     if (hsa_agent_get_info(info.agent, HSA_AGENT_INFO_ISA, &isa) == HSA_STATUS_SUCCESS) {
       // Get ISA name
       uint32_t name_length = 0;
       hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME_LENGTH, &name_length);
 
       if (name_length > 0) {
         char* name_buffer = new char[name_length + 1];
         hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME, name_buffer);
         name_buffer[name_length] = '\0';
         isa_name = std::string(name_buffer);
         delete[] name_buffer;
       }
     }
   }
 
   return isa_name;
 }
 
 GPUCapabilityDetector::ISAFeatures GPUCapabilityDetector::parse_isa_name(
     const std::string& isa_name) {
 
   ISAFeatures features;
 
   // ISA name format: "amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-"
   // Features are indicated by suffixes like ":sramecc+", ":xnack-"
 
   // Convert to lowercase for easier parsing
   std::string isa_lower = isa_name;
   std::transform(isa_lower.begin(), isa_lower.end(), isa_lower.begin(), ::tolower);
 
   // Extract architecture (gfx9xx part)
   size_t gfx_pos = isa_lower.find("gfx");
   if (gfx_pos != std::string::npos) {
     std::string arch = isa_lower.substr(gfx_pos, 6);  // "gfx942"
     uint32_t cdna_gen = get_cdna_generation(arch);
 
     if (cdna_gen > 0) {
       // CDNA architectures
       features.has_mfma = true;
       features.has_xdlops = true;
       features.has_packed_fp16 = true;
       features.has_dot_product = true;
       features.mfma_tile_size = 16;
 
       if (cdna_gen >= 2) {
         features.has_bfloat16 = true;
       }
 
       if (cdna_gen >= 3) {
         features.has_fp8 = true;
       }
     } else if (arch.find("gfx10") != std::string::npos ||
                arch.find("gfx11") != std::string::npos) {
       // RDNA architectures
       features.has_dot_product = true;
       features.has_packed_fp16 = true;
       features.has_mfma = false;  // RDNA doesn't have MFMA
     }
   }
 
   // Check for SRAMECC (indicates data center GPU)
   if (isa_lower.find("sramecc+") != std::string::npos) {
     // SRAM ECC is typically only on CDNA
     features.has_mfma = true;
   }
 
   return features;
 }
 
 // ============================================================================
 // Architecture Classification
 // ============================================================================
 
 GPUCapabilityDetector::ArchitectureClass
 GPUCapabilityDetector::detect_architecture_class(const std::string& gcn_arch_name) {
 
   std::string arch_lower = gcn_arch_name;
   std::transform(arch_lower.begin(), arch_lower.end(), arch_lower.begin(), ::tolower);
 
   // CDNA: gfx908 (MI100), gfx90a (MI200), gfx94x (MI300), gfx95x (MI350+)
   if (arch_lower.find("gfx908") != std::string::npos ||
       arch_lower.find("gfx90a") != std::string::npos ||
       arch_lower.find("gfx94") != std::string::npos ||
       arch_lower.find("gfx95") != std::string::npos) {
     return ArchitectureClass::CDNA;
   }
 
   // RDNA: gfx10xx (RDNA1), gfx10xx (RDNA2), gfx11xx (RDNA3)
   if (arch_lower.find("gfx10") != std::string::npos ||
       arch_lower.find("gfx11") != std::string::npos) {
     return ArchitectureClass::RDNA;
   }
 
   // GCN: gfx7xx, gfx8xx, gfx9xx (non-CDNA)
   if (arch_lower.find("gfx7") != std::string::npos ||
       arch_lower.find("gfx8") != std::string::npos ||
       (arch_lower.find("gfx9") != std::string::npos &&
        arch_lower.find("gfx90") == std::string::npos)) {
     return ArchitectureClass::GCN;
   }
 
   return ArchitectureClass::UNKNOWN;
 }
 
 uint32_t GPUCapabilityDetector::get_cdna_generation(const std::string& gcn_arch_name) {
   std::string arch_lower = gcn_arch_name;
   std::transform(arch_lower.begin(), arch_lower.end(), arch_lower.begin(), ::tolower);
 
   // CDNA1: gfx908 (MI100)
   if (arch_lower.find("gfx908") != std::string::npos) {
     return 1;
   }
 
   // CDNA2: gfx90a (MI250X, MI210)
   if (arch_lower.find("gfx90a") != std::string::npos) {
     return 2;
   }
 
   // CDNA3: gfx940, gfx941, gfx942 (MI300 series)
   if (arch_lower.find("gfx940") != std::string::npos ||
       arch_lower.find("gfx941") != std::string::npos ||
       arch_lower.find("gfx942") != std::string::npos) {
     return 3;
   }
 
   // CDNA3.5/CDNA4: gfx950+ (MI350+ series)
   if (arch_lower.find("gfx950") != std::string::npos ||
       arch_lower.find("gfx951") != std::string::npos ||
       arch_lower.find("gfx952") != std::string::npos) {
     return 4;  // Treating as generation 4
   }
 
   return 0;  // Not CDNA
 }
 
 // ============================================================================
 // Hardware Property Detection
 // ============================================================================
 
 uint32_t GPUCapabilityDetector::detect_wavefront_size(uint16_t gpu_id) {
   hipDeviceProp_t prop;
   if (hipGetDeviceProperties(&prop, gpu_id) == hipSuccess) {
     // HIP provides warpSize which is the wavefront size
     return prop.warpSize;
   }
 
   // Fallback: CDNA uses 64, RDNA uses 32
   return 64;  // Conservative default
 }
 
 uint32_t GPUCapabilityDetector::estimate_wavefronts_per_cu(
     ArchitectureClass arch_class,
     uint32_t cdna_gen) {
 
   switch (arch_class) {
     case ArchitectureClass::CDNA:
       if (cdna_gen >= 3) {
         // CDNA3+ (MI300, MI350 series): 40 wavefronts per CU
         return 40;
       } else {
         // CDNA1/CDNA2 (MI100, MI200 series): 32 wavefronts per CU
         return 32;
       }
 
     case ArchitectureClass::RDNA:
       // RDNA: 16 wavefronts per CU (dual compute unit = 32 total)
       return 16;
 
     case ArchitectureClass::GCN:
       // GCN: 40 wavefronts per CU
       return 40;
 
     default:
       return 32;  // Conservative default
   }
 }
 
 uint64_t GPUCapabilityDetector::calculate_theoretical_bandwidth(
     uint32_t memory_clock_rate,
     uint32_t memory_bus_width) {
 
   // memory_clock_rate is in kHz
   // memory_bus_width is in bits
   // Formula: 2 × clock_rate × (bus_width / 8) / 1000
   // Factor of 2 for DDR (double data rate)
 
   double bandwidth_gbs = 2.0 * memory_clock_rate * (memory_bus_width / 8.0) / 1.0e6;
   return static_cast<uint64_t>(bandwidth_gbs * 1024 * 1024 * 1024);
 }
 
 uint32_t GPUCapabilityDetector::detect_matrix_cores_per_cu(
     const ISAFeatures& isa_features) {
 
   if (isa_features.has_mfma) {
     // CDNA architectures have 4 matrix cores per CU
     return 4;
   }
 
   return 0;  // No matrix cores
 }
 
 } // namespace gpim
 