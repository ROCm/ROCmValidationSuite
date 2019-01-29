/********************************************************************************
 *
 * Copyright (c) 2018 ROCm Developer Tools
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
#include "gtest/gtest.h"
#include "include/action.h"
#include "test/unitsmqt.h"

TEST(smqt, action) {
  bar_data* bd = new bar_data;
  bd->on_set_device_gpu_id();
  EXPECT_EQ(bd->get_dev_id(), 123);
  bd->on_bar_data_read();
  ulong bar1_size, bar2_size, bar4_size, bar5_size;
  ulong bar1_base_addr, bar2_base_addr, bar4_base_addr;
  std::tie(bar1_size, bar2_size, bar4_size, bar5_size) = bd->get_bar_sizes();
  std::tie(bar1_base_addr, bar2_base_addr, bar4_base_addr) = bd->get_bar_addr();
  EXPECT_EQ(bar1_size, 2UL);
  EXPECT_EQ(bar2_size, 3UL);
  EXPECT_EQ(bar4_size, 5UL);
  EXPECT_EQ(bar5_size, 4UL);
  EXPECT_EQ(bar1_base_addr, 1UL);
  EXPECT_EQ(bar2_base_addr, 6UL);
  EXPECT_EQ(bar4_base_addr, 7UL);
  delete bd;
}
