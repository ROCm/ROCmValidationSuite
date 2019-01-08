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
#include "test/unitsmqt.h"
#include <tuple>

//! Default constructor
bar_data::bar_data() {
}

//! Default destructor
bar_data::~bar_data() {
  property.clear();
}
void bar_data::on_set_device_gpu_id() {
  dev_id = 123;
}
void bar_data::on_bar_data_read() {
  bar1_size = 2;
  bar2_size = 3;
  bar4_size = 5;
  bar5_size = 4;
  bar1_base_addr = 1;
  bar2_base_addr = 6;
  bar4_base_addr = 7;
}
std::tuple<ulong, ulong, ulong, ulong> bar_data::get_bar_sizes() {
  return std::make_tuple(bar1_size, bar2_size, bar4_size, bar5_size);
}
std::tuple<ulong, ulong, ulong> bar_data::get_bar_addr() {
  return std::make_tuple(bar1_base_addr, bar2_base_addr, bar4_base_addr);
}
int bar_data::get_dev_id() {
  return dev_id;
}
