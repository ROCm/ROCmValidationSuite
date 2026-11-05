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
#ifndef GPIM_SO_INCLUDE_PROBES_PROBE_BASE_H_
#define GPIM_SO_INCLUDE_PROBES_PROBE_BASE_H_

#include <string>
#include <map>
#include <vector>
#include <cstdint>

#include "amd_smi/amdsmi.h"

namespace gpim {

/**
 * @enum MetricDirection
 * @brief Which direction of change in a metric counts as an improvement
 */
enum class MetricDirection {
  HIGHER_IS_BETTER,  //!< Throughput, bandwidth, TFLOPS
  LOWER_IS_BETTER    //!< Latency, overhead
};

/**
 * @struct Metric
 * @brief One measured quantity from a probe
 *
 * A probe may report several. Regression detection uses only metrics with
 * monitor set. Direction is declared here and is never inferred from the name.
 */
struct Metric {
  std::string name;
  double value;
  std::string unit;
  MetricDirection direction;
  bool monitor;

  Metric() : value(0.0), direction(MetricDirection::HIGHER_IS_BETTER),
             monitor(true) {}
};

/**
 * @struct TelemetrySnapshot
 * @brief Clock, power, and temperature over the probe window
 *
 * The point fields (temp_junction_mc, gpu_clock_mhz, power_usage_uw) hold the
 * window average so existing diagnostic rules keep working. min/avg/max and
 * the residency deltas come from the sampler thread.
 */
struct TelemetrySnapshot {
  bool valid;
  int64_t temp_junction_mc;       //!< average junction/hotspot, millidegrees C
  uint64_t gpu_clock_mhz;         //!< average gfx clock
  uint64_t power_usage_uw;        //!< average socket power, microwatts
  uint32_t gpu_utilization_pct;
  double temp_c_min;
  double temp_c_avg;
  double temp_c_max;
  double clock_mhz_min;
  double clock_mhz_avg;
  double clock_mhz_max;
  double power_w_min;
  double power_w_avg;
  double power_w_max;
  uint64_t ppt_residency_delta;      //!< power-throttle accumulator change
  uint64_t thermal_residency_delta;  //!< thermal-throttle accumulator change
  int sample_count;

  TelemetrySnapshot()
      : valid(false), temp_junction_mc(0), gpu_clock_mhz(0),
        power_usage_uw(0), gpu_utilization_pct(0),
        temp_c_min(0), temp_c_avg(0), temp_c_max(0),
        clock_mhz_min(0), clock_mhz_avg(0), clock_mhz_max(0),
        power_w_min(0), power_w_avg(0), power_w_max(0),
        ppt_residency_delta(0), thermal_residency_delta(0),
        sample_count(0) {}
};

/**
 * @struct ProbeResult
 * @brief Results from a single probe execution
 */
struct ProbeResult {
  std::string probe_name;           //!< Name of the probe
  uint16_t gpu_id;                  //!< GPU device ID
  uint64_t timestamp_us;            //!< Timestamp (microseconds since epoch)
  double value;                     //!< Primary metric value
  std::string unit;                 //!< Unit of measurement
  MetricDirection direction;        //!< Improvement direction of value
  bool passed;                      //!< Did probe pass validation?

  std::vector<Metric> metrics;      //!< All metrics from this execution

  // Secondary metrics (e.g., p95, min, max, std_dev)
  std::map<std::string, double> secondary_metrics;

  TelemetrySnapshot telemetry;

  ProbeResult() : gpu_id(0), timestamp_us(0), value(0.0),
                  direction(MetricDirection::HIGHER_IS_BETTER), passed(true) {}
};

/**
 * @brief Append a metric. The first monitored metric also fills the primary
 *        value fields so a single-metric probe still compares as before.
 */
inline void add_metric(ProbeResult& result, const std::string& name,
                       double value, const std::string& unit,
                       MetricDirection direction, bool monitor = true) {
  Metric metric;
  metric.name = name;
  metric.value = value;
  metric.unit = unit;
  metric.direction = direction;
  metric.monitor = monitor;
  result.metrics.push_back(metric);
  if (monitor && result.probe_name.empty()) {
    result.probe_name = name;
    result.value = value;
    result.unit = unit;
    result.direction = direction;
  }
}

/**
 * @class ProbeBase
 * @brief Abstract base class for all performance probes
 *
 * All probes must implement the core interface:
 * - name(), description(), unit(): Metadata
 * - initialize(): GPU-specific setup
 * - execute(): Run probe and return results
 * - cleanup(): Release resources
 */
class ProbeBase {
 public:
  virtual ~ProbeBase() = default;

  // Core interface (must be implemented by derived classes)

  /**
   * @brief Get probe name (e.g., "kernel_launch_latency")
   */
  virtual std::string name() const = 0;

  /**
   * @brief Get human-readable description
   */
  virtual std::string description() const = 0;

  /**
   * @brief Get unit of measurement (e.g., "us", "TFLOPS", "GB/s")
   */
  virtual std::string unit() const = 0;

  /**
   * @brief Improvement direction of the primary metric
   */
  virtual MetricDirection direction() const = 0;

  /**
   * @brief Initialize probe for a specific GPU
   * @param gpu_id GPU device ID
   * @return 0 on success, non-zero on error
   */
  virtual int initialize(uint16_t gpu_id) = 0;

  /**
   * @brief Execute the probe and return results
   * @return ProbeResult containing measurements
   */
  virtual ProbeResult execute() = 0;

  /**
   * @brief Cleanup and release resources
   */
  virtual void cleanup() = 0;

  // Configuration (optional, has default implementation)

  /**
   * @brief Set number of iterations for probe execution
   * @param count Number of iterations
   */
  virtual void set_iterations(uint32_t count) {
    iterations_ = count;
  }

  /**
   * @brief Set number of warmup iterations
   * @param count Number of warmup iterations
   */
  virtual void set_warmup(uint32_t count) {
    warmup_iterations_ = count;
  }

  /**
   * @brief Set custom configuration parameter
   * @param key Parameter name
   * @param value Parameter value
   */
  virtual void set_config(const std::string& key, const std::string& value) {
    config_[key] = value;
  }

  /**
   * @brief Get current GPU ID
   */
  uint16_t get_gpu_id() const {
    return gpu_id_;
  }

  /**
   * @brief AMD SMI handle for the GPU this probe was initialized on
   */
  amdsmi_processor_handle smi_handle() const {
    return device_handle_;
  }

  /** 
   *@brief Get current timestamp in microseconds (static utility)
   */
  static uint64_t get_timestamp_us();

  static double compute_median(std::vector<double>& values);
  static double compute_percentile(std::vector<double>& values, double percentile);
  static double compute_stddev(const std::vector<double>& values, double mean);

 protected:
  uint16_t gpu_id_ = 0;                         //!< Target GPU device ID
  uint32_t iterations_ = 100;                   //!< Number of measurement iterations
  uint32_t warmup_iterations_ = 10;             //!< Number of warmup iterations
  std::map<std::string, std::string> config_;   //!< Custom configuration parameters
  amdsmi_processor_handle device_handle_ = nullptr;  //!< AMD SMI device handle


  /**
   * @brief Capture telemetry snapshot from GPU
   * @param result ProbeResult to populate with telemetry data
   */
  void capture_telemetry(ProbeResult& result);
};

}  // namespace gpim

#endif  // GPIM_SO_INCLUDE_PROBES_PROBE_BASE_H_
