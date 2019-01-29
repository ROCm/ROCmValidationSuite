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
#include "include/gpu_util.h"

Worker* pworker;

TEST(gm, coverage_rsmi_failure) {
  rvs::gpulist::Initialize();
  pworker = nullptr;
  gm_action* pa = new gm_action;
  ASSERT_NE(pa, nullptr);
  pa->property_set("monitor", "true");
  pa->property_set("name", "unit_test");
  pa->property_set("device", "all");
  pa->property_set("terminate", "true");
  pa->property_set("metrics.temp", "true 30 0");
  pa->property_set("metrics.fan", "true 100 0");
  pa->property_set("metrics.clock", "true 1500 0");
  pa->property_set("metrics.mem_clock", "true 1500 0");
  pa->property_set("duration", "1000");
  pa->run();
  delete pa;
  pworker->stop();
  delete pworker;
}
