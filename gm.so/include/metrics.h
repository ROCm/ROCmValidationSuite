/********************************************************************************
 *
 * Copyright (c) 2018-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef GM_SO_INCLUDE_METRICS_H_
#define GM_SO_INCLUDE_METRICS_H_

//! monitored metric and its bound values
typedef struct {
  //! true if metric observed
  bool mon_metric;
  //! true if bounds checked
  bool check_bounds;
  //! bound max_val
  uint32_t max_val;
  //! bound min_val
  uint32_t min_val;
} Metric_bound;

//! number of violations for metrics
typedef struct {
  //! gpu_id
  uint32_t gpu_id;
  //! number of temperature violation
  int temp_violation;
  //! number of clock violation
  int clock_violation;
  //! number of mem_clock violation
  int mem_clock_violation;
  //! number of fan violation
  int fan_violation;
  //! number of power violation
  int power_violation;
} Metric_violation;

//! current metric values
typedef struct {
  //! gpu_id
  uint32_t gpu_id;
  //! current temperature value
  int64_t temp;
  //! current clock value
  uint64_t clock;
  //! current mem_clock value
  uint64_t mem_clock;
  //! current fan percentage
  uint32_t fan;
  //! current power value
  uint32_t power;
} Metric_value;

//! average metric values
typedef struct {
  //! gpu_id
  uint32_t gpu_id;
  //! average temperature
  int64_t av_temp;
  //! average clock
  uint64_t av_clock;
  //! average mem_clock
  uint64_t av_mem_clock;
  //! average fan
  uint64_t av_fan;
  //! average power
  float av_power;
} Metric_avg;

#endif  // GM_SO_INCLUDE_METRICS_H_
