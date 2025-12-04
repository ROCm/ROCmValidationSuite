/********************************************************************************
 *
 * Copyright (c) 2018-25 Advanced Micro Devices, Inc. All rights reserved.
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
 

#include <vector>
#include <string>
#include <chrono>
#include <iomanip> 
#include "gtest/gtest.h"

#include "include/gpu_util.h"
#include "include/rvs_unit_testing_defs.h"

using rvs::gpulist;
using rvs::GpuInfo;
using rvs::GpuLookupError;

/**
 * @class GpuUtilTest
 * @brief Test fixture for GPU utility functions
 * 
 * This test fixture sets up a mock GPU environment with known values
 * for testing all lookup and validation functions.
 */
class GpuUtilTest : public ::testing::Test, public rvs::gpulist {
 protected:
  void SetUp() override {
    // Clear any existing data
    clear();
    
    // Set up legacy parallel arrays (for backward compatibility testing)
    location_id = {2, 1, 5, 7, 8, 3};
    gpu_id      = {1, 2, 5, 4, 9, 7};
    device_id   = {3, 0, 2, 7, 5, 1};
    node_id     = {2, 1, 3, 7, 4, 9};
    domain_id   = {0, 0, 0, 0, 0, 0};
    gpu_idx     = {0, 1, 2, 3, 4, 5};
    pci_bdf     = {"0000:01:00.0", "0000:02:00.0", "0000:03:00.0", 
                   "0000:04:00.0", "0000:05:00.0", "0000:06:00.0"};
    
    // Populate new unified structures
    gpu_info_list.clear();
    gpu_id_to_index.clear();
    location_id_to_index.clear();
    node_id_to_index.clear();
    device_id_to_index.clear();
    domain_location_to_index.clear();
    
    for (size_t i = 0; i < gpu_id.size(); ++i) {
      GpuInfo info(
        location_id[i],
        gpu_id[i],
        gpu_idx[i],
        device_id[i],
        node_id[i],
        domain_id[i],
        pci_bdf[i]
      );
      
      gpu_info_list.push_back(info);
      
      // Build index maps
      gpu_id_to_index[gpu_id[i]] = i;
      location_id_to_index[location_id[i]] = i;
      node_id_to_index[node_id[i]] = i;
      device_id_to_index[device_id[i]] = i;
      
      auto domain_loc_pair = std::make_pair(domain_id[i], location_id[i]);
      domain_location_to_index[domain_loc_pair] = i;
    }
  }

  void TearDown() override {
    clear();
  }
};


TEST_F(GpuUtilTest, location2gpu_ValidLookup) {
  uint16_t result_id;
  for (size_t i = 0; i < location_id.size(); i++) {
    int return_value = location2gpu(location_id[i], &result_id);
    EXPECT_EQ(result_id, gpu_id[i]) 
        << "Location ID " << location_id[i] << " should map to GPU ID " << gpu_id[i];
    EXPECT_EQ(return_value, 0) << "Return value should be 0 (success)";
  }
}

TEST_F(GpuUtilTest, location2gpu_InvalidLookup) {
  uint16_t result_id;
  int return_value = location2gpu(9999, &result_id);
  EXPECT_EQ(return_value, -1) << "Should return -1 for non-existent location ID";
}

TEST_F(GpuUtilTest, gpu2location_ValidLookup) {
  uint16_t result_id;
  for (size_t i = 0; i < gpu_id.size(); i++) {
    int return_value = gpu2location(gpu_id[i], &result_id);
    EXPECT_EQ(result_id, location_id[i])
        << "GPU ID " << gpu_id[i] << " should map to Location ID " << location_id[i];
    EXPECT_EQ(return_value, 0);
  }
}

TEST_F(GpuUtilTest, gpu2location_InvalidLookup) {
  uint16_t result_id;
  int return_value = gpu2location(9999, &result_id);
  EXPECT_EQ(return_value, -1);
}

TEST_F(GpuUtilTest, node2gpu_ValidLookup) {
  uint16_t result_id;
  for (size_t i = 0; i < node_id.size(); i++) {
    int return_value = node2gpu(node_id[i], &result_id);
    EXPECT_EQ(result_id, gpu_id[i]);
    EXPECT_EQ(return_value, 0);
  }
}

TEST_F(GpuUtilTest, node2gpu_InvalidLookup) {
  uint16_t result_id;
  int return_value = node2gpu(9999, &result_id);
  EXPECT_EQ(return_value, -1);
}

TEST_F(GpuUtilTest, location2device_ValidLookup) {
  uint16_t result_id;
  for (size_t i = 0; i < location_id.size(); i++) {
    int return_value = location2device(location_id[i], &result_id);
    EXPECT_EQ(result_id, device_id[i]);
    EXPECT_EQ(return_value, 0);
  }
}

TEST_F(GpuUtilTest, location2device_InvalidLookup) {
  uint16_t result_id;
  int return_value = location2device(9999, &result_id);
  EXPECT_EQ(return_value, -1);
}


TEST_F(GpuUtilTest, gpu2node_ValidLookup) {
  uint16_t result_id;
  for (size_t i = 0; i < gpu_id.size(); i++) {
    int return_value = gpu2node(gpu_id[i], &result_id);
    EXPECT_EQ(result_id, node_id[i])
        << "GPU ID " << gpu_id[i] << " should map to Node ID " << node_id[i];
    EXPECT_EQ(return_value, static_cast<int>(GpuLookupError::SUCCESS));
  }
}

TEST_F(GpuUtilTest, gpu2node_InvalidGpuId) {
  uint16_t result_id;
  int return_value = gpu2node(9999, &result_id);
  EXPECT_EQ(return_value, static_cast<int>(GpuLookupError::NOT_FOUND))
      << "Should return NOT_FOUND for non-existent GPU ID";
}

TEST_F(GpuUtilTest, gpu2node_NullPointer) {
  int return_value = gpu2node(gpu_id[0], nullptr);
  EXPECT_EQ(return_value, static_cast<int>(GpuLookupError::NULL_POINTER))
      << "Should return NULL_POINTER error when output pointer is null";
}

TEST_F(GpuUtilTest, gpu2node_Uninitialized) {
  // Clear all data to simulate uninitialized state
  gpu_info_list.clear();
  gpu_id_to_index.clear();
  
  uint16_t result_id;
  int return_value = gpu2node(gpu_id[0], &result_id);
  EXPECT_EQ(return_value, static_cast<int>(GpuLookupError::UNINITIALIZED))
      << "Should return UNINITIALIZED when GPU list is empty";
}

TEST_F(GpuUtilTest, gpu2device_ValidLookup) {
  uint16_t result_id;
  for (size_t i = 0; i < gpu_id.size(); i++) {
    int return_value = gpu2device(gpu_id[i], &result_id);
    EXPECT_EQ(result_id, device_id[i]);
    EXPECT_EQ(return_value, static_cast<int>(GpuLookupError::SUCCESS));
  }
}

TEST_F(GpuUtilTest, gpu2device_NullPointer) {
  int return_value = gpu2device(gpu_id[0], nullptr);
  EXPECT_EQ(return_value, static_cast<int>(GpuLookupError::NULL_POINTER));
}

TEST_F(GpuUtilTest, location2node_ValidLookup) {
  uint16_t result_id;
  for (size_t i = 0; i < location_id.size(); i++) {
    int return_value = location2node(location_id[i], &result_id);
    EXPECT_EQ(result_id, node_id[i]);
    EXPECT_EQ(return_value, static_cast<int>(GpuLookupError::SUCCESS));
  }
}

TEST_F(GpuUtilTest, location2node_InvalidLocation) {
  uint16_t result_id;
  int return_value = location2node(9999, &result_id);
  EXPECT_EQ(return_value, static_cast<int>(GpuLookupError::NOT_FOUND));
}

TEST_F(GpuUtilTest, location2node_NullPointer) {
  int return_value = location2node(location_id[0], nullptr);
  EXPECT_EQ(return_value, static_cast<int>(GpuLookupError::NULL_POINTER));
}


TEST_F(GpuUtilTest, domlocation2gpu_ValidLookup) {
  uint16_t result_id;
  for (size_t i = 0; i < gpu_id.size(); i++) {
    int return_value = domlocation2gpu(domain_id[i], location_id[i], &result_id);
    EXPECT_EQ(result_id, gpu_id[i])
        << "Domain " << domain_id[i] << " + Location " << location_id[i] 
        << " should map to GPU ID " << gpu_id[i];
    EXPECT_EQ(return_value, static_cast<int>(GpuLookupError::SUCCESS));
  }
}

TEST_F(GpuUtilTest, domlocation2gpu_InvalidPair) {
  uint16_t result_id;
  int return_value = domlocation2gpu(9999, 9999, &result_id);
  EXPECT_EQ(return_value, static_cast<int>(GpuLookupError::NOT_FOUND));
}

TEST_F(GpuUtilTest, domlocation2gpu_NullPointer) {
  int return_value = domlocation2gpu(domain_id[0], location_id[0], nullptr);
  EXPECT_EQ(return_value, static_cast<int>(GpuLookupError::NULL_POINTER));
}



TEST_F(GpuUtilTest, get_gpu_info_by_gpu_id_ValidLookup) {
  for (size_t i = 0; i < gpu_id.size(); i++) {
    const GpuInfo* info = get_gpu_info_by_gpu_id(gpu_id[i]);
    ASSERT_NE(info, nullptr) << "Should return valid pointer for GPU ID " << gpu_id[i];
    EXPECT_EQ(info->gpu_id, gpu_id[i]);
    EXPECT_EQ(info->location_id, location_id[i]);
    EXPECT_EQ(info->node_id, node_id[i]);
    EXPECT_EQ(info->device_id, device_id[i]);
    EXPECT_EQ(info->domain_id, domain_id[i]);
    EXPECT_EQ(info->gpu_idx, gpu_idx[i]);
    EXPECT_EQ(info->pci_bdf, pci_bdf[i]);
  }
}

TEST_F(GpuUtilTest, get_gpu_info_by_gpu_id_InvalidLookup) {
  const GpuInfo* info = get_gpu_info_by_gpu_id(9999);
  EXPECT_EQ(info, nullptr) << "Should return nullptr for non-existent GPU ID";
}

TEST_F(GpuUtilTest, get_gpu_info_by_location_ValidLookup) {
  for (size_t i = 0; i < location_id.size(); i++) {
    const GpuInfo* info = get_gpu_info_by_location(location_id[i]);
    ASSERT_NE(info, nullptr);
    EXPECT_EQ(info->location_id, location_id[i]);
    EXPECT_EQ(info->gpu_id, gpu_id[i]);
  }
}

TEST_F(GpuUtilTest, get_gpu_info_by_location_InvalidLookup) {
  const GpuInfo* info = get_gpu_info_by_location(9999);
  EXPECT_EQ(info, nullptr);
}

TEST_F(GpuUtilTest, get_gpu_info_by_node_ValidLookup) {
  for (size_t i = 0; i < node_id.size(); i++) {
    const GpuInfo* info = get_gpu_info_by_node(node_id[i]);
    ASSERT_NE(info, nullptr);
    EXPECT_EQ(info->node_id, node_id[i]);
    EXPECT_EQ(info->gpu_id, gpu_id[i]);
  }
}

TEST_F(GpuUtilTest, get_gpu_info_by_node_InvalidLookup) {
  const GpuInfo* info = get_gpu_info_by_node(9999);
  EXPECT_EQ(info, nullptr);
}

TEST_F(GpuUtilTest, get_all_gpu_info) {
  const std::vector<GpuInfo>& all_gpus = get_all_gpu_info();
  EXPECT_EQ(all_gpus.size(), gpu_id.size())
      << "Should return all " << gpu_id.size() << " GPUs";
  
  // Verify each GPU info is correct
  for (size_t i = 0; i < all_gpus.size(); i++) {
    EXPECT_EQ(all_gpus[i].gpu_id, gpu_id[i]);
    EXPECT_EQ(all_gpus[i].location_id, location_id[i]);
    EXPECT_EQ(all_gpus[i].node_id, node_id[i]);
  }
}

TEST_F(GpuUtilTest, is_valid_gpu_id) {
  // Test valid GPU IDs
  for (const auto& id : gpu_id) {
    EXPECT_TRUE(is_valid_gpu_id(id))
        << "GPU ID " << id << " should be valid";
  }
  
  // Test invalid GPU ID
  EXPECT_FALSE(is_valid_gpu_id(9999))
      << "GPU ID 9999 should be invalid";
}

TEST_F(GpuUtilTest, get_gpu_count) {
  size_t count = get_gpu_count();
  EXPECT_EQ(count, gpu_id.size())
      << "GPU count should match number of GPUs in test data";
}

TEST_F(GpuUtilTest, clear) {
  // Verify data exists
  EXPECT_GT(get_gpu_count(), 0);
  
  // Clear all data
  clear();
  
  // Verify everything is cleared
  EXPECT_EQ(get_gpu_count(), 0);
  EXPECT_EQ(gpu_info_list.size(), 0);
  EXPECT_EQ(gpu_id_to_index.size(), 0);
  EXPECT_EQ(location_id_to_index.size(), 0);
  EXPECT_EQ(node_id_to_index.size(), 0);
  EXPECT_EQ(device_id_to_index.size(), 0);
  EXPECT_EQ(domain_location_to_index.size(), 0);
  
  // Legacy arrays should also be cleared
  EXPECT_EQ(location_id.size(), 0);
  EXPECT_EQ(gpu_id.size(), 0);
  EXPECT_EQ(device_id.size(), 0);
  EXPECT_EQ(node_id.size(), 0);
}


TEST_F(GpuUtilTest, GpuInfo_DefaultConstructor) {
  GpuInfo info;
  EXPECT_EQ(info.location_id, 0);
  EXPECT_EQ(info.gpu_id, 0);
  EXPECT_EQ(info.gpu_idx, 0);
  EXPECT_EQ(info.device_id, 0);
  EXPECT_EQ(info.node_id, 0);
  EXPECT_EQ(info.domain_id, 0);
  EXPECT_EQ(info.pci_bdf, "");
}

TEST_F(GpuUtilTest, GpuInfo_ParameterizedConstructor) {
  GpuInfo info(100, 200, 0, 300, 400, 500, "0000:01:00.0");
  EXPECT_EQ(info.location_id, 100);
  EXPECT_EQ(info.gpu_id, 200);
  EXPECT_EQ(info.gpu_idx, 0);
  EXPECT_EQ(info.device_id, 300);
  EXPECT_EQ(info.node_id, 400);
  EXPECT_EQ(info.domain_id, 500);
  EXPECT_EQ(info.pci_bdf, "0000:01:00.0");
}

TEST_F(GpuUtilTest, GpuInfo_is_valid) {
  GpuInfo valid_info(1, 100, 0, 2, 3, 4, "0000:01:00.0");
  EXPECT_TRUE(valid_info.is_valid()) 
      << "GPU info with non-zero gpu_id should be valid";
  
  GpuInfo invalid_info;
  EXPECT_FALSE(invalid_info.is_valid())
      << "GPU info with zero gpu_id should be invalid";
}

TEST_F(GpuUtilTest, GpuInfo_get_domain_location_pair) {
  GpuInfo info(100, 200, 0, 300, 400, 500, "0000:01:00.0");
  auto pair = info.get_domain_location_pair();
  EXPECT_EQ(pair.first, 500);   // domain_id
  EXPECT_EQ(pair.second, 100);  // location_id
}


TEST_F(GpuUtilTest, Performance_HashMapLookup) {
  const int iterations = 10000;
  uint16_t result_id;
  
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; ++i) {
    gpu2node(gpu_id[i % gpu_id.size()], &result_id);
  }
  auto end = std::chrono::high_resolution_clock::now();
  
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  
  // Should complete 10,000 lookups in less than 10ms 
  EXPECT_LT(duration.count(), 10000)
      << "10,000 hash map lookups should take less than 10ms, took " 
      << duration.count() << "µs";
  
  std::cout << "Performance: " << iterations << " lookups in " 
            << duration.count() << "µs ("
            << (duration.count() / static_cast<double>(iterations)) 
            << "µs per lookup)" << std::endl;
}

TEST_F(GpuUtilTest, Performance_NewAPIvsLegacy) {
  const int iterations = 1000;
  
  // Test new API (get_gpu_info_by_gpu_id)
  auto start_new = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; ++i) {
    const GpuInfo* info = get_gpu_info_by_gpu_id(gpu_id[i % gpu_id.size()]);
    (void)info;  // Suppress unused variable warning
  }
  auto end_new = std::chrono::high_resolution_clock::now();
  auto duration_new = std::chrono::duration_cast<std::chrono::microseconds>(
      end_new - start_new);
  
  // Test legacy API (gpu2node)
  uint16_t result;
  auto start_legacy = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; ++i) {
    gpu2node(gpu_id[i % gpu_id.size()], &result);
  }
  auto end_legacy = std::chrono::high_resolution_clock::now();
  auto duration_legacy = std::chrono::duration_cast<std::chrono::microseconds>(
      end_legacy - start_legacy);
  
  std::cout << "New API: " << iterations << " lookups in " 
            << duration_new.count() << "µs" << std::endl;
  std::cout << "Legacy API: " << iterations << " lookups in " 
            << duration_legacy.count() << "µs" << std::endl;
  
  // Both should be fast (O(1)), but new API might have slight overhead
  EXPECT_LT(duration_new.count(), 5000);
  EXPECT_LT(duration_legacy.count(), 5000);
}

TEST_F(GpuUtilTest, EdgeCase_MultipleLookupsWithSameKey) {
  uint16_t result1, result2, result3;
  
  int ret1 = gpu2node(gpu_id[0], &result1);
  int ret2 = gpu2node(gpu_id[0], &result2);
  int ret3 = gpu2node(gpu_id[0], &result3);
  
  EXPECT_EQ(ret1, static_cast<int>(GpuLookupError::SUCCESS));
  EXPECT_EQ(ret2, static_cast<int>(GpuLookupError::SUCCESS));
  EXPECT_EQ(ret3, static_cast<int>(GpuLookupError::SUCCESS));
  EXPECT_EQ(result1, result2);
  EXPECT_EQ(result2, result3);
}

TEST_F(GpuUtilTest, EdgeCase_ZeroValues) {
  // Add a GPU with zero values (except gpu_id which must be non-zero)
  GpuInfo zero_info(0, 12345, 0, 0, 0, 0, "");
  gpu_info_list.push_back(zero_info);
  gpu_id_to_index[12345] = gpu_info_list.size() - 1;
  
  const GpuInfo* info = get_gpu_info_by_gpu_id(12345);
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->location_id, 0);
  EXPECT_EQ(info->device_id, 0);
  EXPECT_EQ(info->node_id, 0);
  EXPECT_TRUE(info->is_valid());  // Still valid because gpu_id is non-zero
}

TEST_F(GpuUtilTest, EdgeCase_LargeGpuId) {
  uint16_t large_id = 0xFFFF;  // Max uint16_t
  GpuInfo large_info(1, large_id, 0, 2, 3, 4, "0000:ff:00.0");
  gpu_info_list.push_back(large_info);
  gpu_id_to_index[large_id] = gpu_info_list.size() - 1;
  
  uint16_t result;
  int ret = gpu2node(large_id, &result);
  EXPECT_EQ(ret, static_cast<int>(GpuLookupError::SUCCESS));
}

TEST_F(GpuUtilTest, Performance_HashMapVsLinearSearch) {
  const int iterations = 10000;
  uint16_t result_id;
  
  // Simulate old linear search
  auto old_linear_search = [&](uint16_t id) {
    const auto it = std::find(gpu_id.cbegin(), gpu_id.cend(), id);
    if (it != gpu_id.cend()) {
      size_t pos = std::distance(gpu_id.cbegin(), it);
      return node_id[pos];
    }
    return uint16_t(0);
  };
  
  // Test middle element (worst case for linear search)
  uint16_t test_id = gpu_id[gpu_id.size() / 2];
  
  // Benchmark old method
  auto start_old = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; ++i) {
    volatile auto result = old_linear_search(test_id);
  }
  auto end_old = std::chrono::high_resolution_clock::now();
  auto duration_old = std::chrono::duration_cast<std::chrono::microseconds>(
      end_old - start_old);
  
  // Benchmark new method
  auto start_new = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; ++i) {
    gpu2node(test_id, &result_id);
  }
  auto end_new = std::chrono::high_resolution_clock::now();
  auto duration_new = std::chrono::duration_cast<std::chrono::microseconds>(
      end_new - start_new);
  
  // Calculate speedup
  double speedup = static_cast<double>(duration_old.count()) / 
                   static_cast<double>(duration_new.count());
  
  std::cout << "\n=== Performance Comparison ===\n";
  std::cout << "Iterations: " << iterations << "\n";
  std::cout << "Old (Linear): " << duration_old.count() << " μs\n";
  std::cout << "New (Hash Map): " << duration_new.count() << " μs\n";
  std::cout << "Speedup: " << std::setprecision(2) << speedup << "x\n";
  std::cout << "Improvement: " << std::setprecision(1) 
            << ((1.0 - 1.0/speedup) * 100.0) << "%\n\n";
  
  
  
  // Verify the hash map structure exists
  EXPECT_GT(gpu_id_to_index.size(), 0) << "Hash map should be populated";
}
  

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

