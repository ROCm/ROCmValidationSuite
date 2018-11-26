/********************************************************************************
 *
 * Copyright (c) 2018 ROCm Developer Tools
 *
 * MIT LICENSE:
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without result_idtriction, including without limitation the rights to
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

#include "gtest/gtest.h"

#include "gpu_util.h"

#include <vector>

class GpuUtilTest : public ::testing::Test , public rvs::gpulist {
 protected:
  void SetUp() override {
    rvs::gpulist::location_id = {2, 1, 5, 7, 8, 3};
    rvs::gpulist::gpu_id      = {1, 2, 5, 4, 9, 7};
    rvs::gpulist::device_id   = {3, 0, 2, 7, 5, 1};
    rvs::gpulist::node_id     = {2, 1, 3, 7, 4, 9};
  }

  void TearDown() override {
    rvs::gpulist::location_id.clear();
    rvs::gpulist::gpu_id.clear();
    rvs::gpulist::device_id.clear();
    rvs::gpulist::node_id.clear();
  }

};

TEST_F(GpuUtilTest, gpu_util) {
  uint16_t result_id;
  int return_value;

  // location2gpu
  for (int i = 0; i < (int) rvs::gpulist::location_id.size(); i++) {
    return_value = rvs::gpulist::location2gpu(rvs::gpulist::location_id[i], &result_id);
    EXPECT_EQ(result_id, rvs::gpulist::gpu_id[i]);
    EXPECT_EQ(return_value, 0);
  };
  return_value = rvs::gpulist::location2gpu(100, &result_id);
  EXPECT_EQ(return_value, -1);

  // gpu2location
  for (int i = 0; i < (int) rvs::gpulist::gpu_id.size(); i++) {
    return_value = rvs::gpulist::gpu2location(rvs::gpulist::gpu_id[i], &result_id);
    EXPECT_EQ(result_id, rvs::gpulist::location_id[i]);
    EXPECT_EQ(return_value, 0);
  };
  return_value = rvs::gpulist::gpu2location(100, &result_id);
  EXPECT_EQ(return_value, -1);

  // node2gpu
  for (int i = 0; i < (int) rvs::gpulist::node_id.size(); i++) {
    return_value = rvs::gpulist::node2gpu(rvs::gpulist::node_id[i], &result_id);
    EXPECT_EQ(result_id, rvs::gpulist::gpu_id[i]);
    EXPECT_EQ(return_value, 0);
  };
  return_value = rvs::gpulist::node2gpu(100, &result_id);
  EXPECT_EQ(return_value, -1);

  // location2device
  for (int i = 0; i < (int) rvs::gpulist::location_id.size(); i++) {
    return_value = rvs::gpulist::location2device(rvs::gpulist::location_id[i], &result_id);
    EXPECT_EQ(result_id, rvs::gpulist::device_id[i]);
    EXPECT_EQ(return_value, 0);
  };
  return_value = rvs::gpulist::location2device(100, &result_id);
  EXPECT_EQ(return_value, -1);

  // gpu2device
  for (int i = 0; i < (int) rvs::gpulist::gpu_id.size(); i++) {
    return_value = rvs::gpulist::gpu2device(rvs::gpulist::gpu_id[i], &result_id);
    EXPECT_EQ(result_id, rvs::gpulist::device_id[i]);
    EXPECT_EQ(return_value, 0);
  };
  return_value = rvs::gpulist::gpu2device(100, &result_id);
  EXPECT_EQ(return_value, -1);

  // location2node
  for (int i = 0; i < (int) rvs::gpulist::location_id.size(); i++) {
    return_value = rvs::gpulist::location2node(rvs::gpulist::location_id[i], &result_id);
    EXPECT_EQ(result_id, rvs::gpulist::node_id[i]);
    EXPECT_EQ(return_value, 0);
  };
  return_value = rvs::gpulist::location2node(100, &result_id);
  EXPECT_EQ(return_value, -1);

  // gpu2node
  for (int i = 0; i < (int) rvs::gpulist::gpu_id.size(); i++) {
    return_value = rvs::gpulist::gpu2node(rvs::gpulist::gpu_id[i], &result_id);
    EXPECT_EQ(result_id, rvs::gpulist::node_id[i]);
    EXPECT_EQ(return_value, 0);
  };
  return_value = rvs::gpulist::gpu2node(100, &result_id);
  EXPECT_EQ(return_value, -1);

}

