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
#ifndef GPIM_SO_INCLUDE_PROBES_CACHE_PERFORMANCE_PROBE_H_
#define GPIM_SO_INCLUDE_PROBES_CACHE_PERFORMANCE_PROBE_H_

#include "include/probes/probe_base.h"
#include "hip/hip_runtime.h"

namespace gpim {

/**
 * @class CachePerformanceProbe
 * @brief Measures GPU cache hierarchy performance
 *
 * Tests cache latency and bandwidth for:
 * - L1 cache (per-CU)
 * - L2 cache (shared across CUs)
 * - L3 cache (Infinity Cache on MI300)
 */
class CachePerformanceProbe : public ProbeBase {
 public:
  CachePerformanceProbe();
  ~CachePerformanceProbe() override;

  std::string name() const override { return "cache_performance"; }
  std::string description() const override {
    return "GPU cache hierarchy latency and bandwidth (L1/L2/L3)";
  }
  std::string unit() const override { return "ns"; }
  MetricDirection direction() const override {
    return MetricDirection::LOWER_IS_BETTER;
  }

  int initialize(uint16_t gpu_id) override;
  ProbeResult execute() override;
  void cleanup() override;

 private:
  void* d_buffer_;
  hipStream_t stream_;

  /**
   * @brief Measure L1 cache latency via pointer chasing
   */
  double measure_l1_latency();

  /**
   * @brief Measure L2 cache latency
   */
  double measure_l2_latency();

  /**
   * @brief Detect cache thrashing
   */
  bool detect_cache_thrashing();
};

}  // namespace gpim

#endif  // GPIM_SO_INCLUDE_PROBES_CACHE_PERFORMANCE_PROBE_H_
