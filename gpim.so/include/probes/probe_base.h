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

  // Secondary metrics (e.g., p95, min, max, std_dev)
  std::map<std::string, double> secondary_metrics;

  // Telemetry snapshot during probe execution
  struct {
    bool valid;
    int64_t temp_junction_mc;       // millidegrees Celsius
    uint64_t gpu_clock_mhz;
    uint64_t power_usage_uw;        // microwatts
    uint32_t gpu_utilization_pct;
  } telemetry;

  ProbeResult() : gpu_id(0), timestamp_us(0), value(0.0),
                  direction(MetricDirection::HIGHER_IS_BETTER), passed(true) {
    telemetry.valid = false;
    telemetry.temp_junction_mc = 0;
    telemetry.gpu_clock_mhz = 0;
    telemetry.power_usage_uw = 0;
    telemetry.gpu_utilization_pct = 0;
  }
};

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
   *@brief Get current timestamp in microseconds (static utility)
   */
  static uint64_t get_timestamp_us();

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

  /**
   * @brief Compute median from a vector of values
   */
  static double compute_median(std::vector<double>& values);

  /**
   * @brief Compute percentile from a vector of values
   * @param values Vector of values (will be modified/sorted)
   * @param percentile Percentile to compute (0.0 to 1.0)
   */
  static double compute_percentile(std::vector<double>& values, double percentile);

  /**
   * @brief Compute standard deviation
   */
  static double compute_stddev(const std::vector<double>& values, double mean);
};

}  // namespace gpim

#endif  // GPIM_SO_INCLUDE_PROBES_PROBE_BASE_H_
