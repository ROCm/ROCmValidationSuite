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
#ifndef GPIM_SO_INCLUDE_MODULE_ADAPTER_H_
#define GPIM_SO_INCLUDE_MODULE_ADAPTER_H_

#include "include/probes/probe_base.h"
#include <string>
#include <vector>
#include <map>

namespace gpim {

/**
 * @class ModuleAdapter
 * @brief Base class for adapting RVS module outputs to GPIM baseline format
 *
 * Adapters allow GPIM to orchestrate existing RVS modules (GST, IET, BABEL)
 * for comprehensive baseline creation while using lightweight probes for
 * frequent monitoring.
 */
class ModuleAdapter {
 public:
  virtual ~ModuleAdapter() = default;

  /**
   * @brief Execute RVS module and capture results
   * @return ProbeResult in GPIM format
   */
  virtual ProbeResult execute_and_capture() = 0;

  /**
   * @brief Get module name
   * @return Module name (e.g., "gst", "iet", "babel")
   */
  virtual std::string get_module_name() const = 0;

  /**
   * @brief Get baseline key for this module's output
   * @return Baseline key (e.g., "gst_gflops", "iet_power")
   */
  virtual std::string get_baseline_key() const = 0;

  /**
   * @brief Set GPU ID for module execution
   */
  virtual void set_gpu_id(uint16_t gpu_id) = 0;

 protected:
  uint16_t gpu_id_;

  /**
   * @brief Generate temporary configuration file for module
   */
  virtual std::string generate_config() = 0;

  /**
   * @brief Parse module output and extract metrics
   */
  virtual double parse_output(const std::string& output) = 0;
};

/**
 * @class GSTAdapter
 * @brief Adapter for GST module (GPU Stress Test - GFLOPS)
 */
class GSTAdapter : public ModuleAdapter {
 public:
  GSTAdapter();
  ~GSTAdapter();

  ProbeResult execute_and_capture() override;
  std::string get_module_name() const override;
  std::string get_baseline_key() const override;
  void set_gpu_id(uint16_t gpu_id) override;

  /**
   * @brief Set test duration in milliseconds
   */
  void set_duration(uint64_t duration_ms);

  /**
   * @brief Set matrix size
   */
  void set_matrix_size(uint64_t size);

 protected:
  std::string generate_config() override;
  double parse_output(const std::string& output) override;

 private:
  uint64_t duration_ms_;
  uint64_t matrix_size_;
};

/**
 * @class IETAdapter
 * @brief Adapter for IET module (Input EDPp Test - Power)
 */
class IETAdapter : public ModuleAdapter {
 public:
  IETAdapter();
  ~IETAdapter();

  ProbeResult execute_and_capture() override;
  std::string get_module_name() const override;
  std::string get_baseline_key() const override;
  void set_gpu_id(uint16_t gpu_id) override;

  /**
   * @brief Set test duration in milliseconds
   */
  void set_duration(uint64_t duration_ms);

 protected:
  std::string generate_config() override;
  double parse_output(const std::string& output) override;

 private:
  uint64_t duration_ms_;
};

/**
 * @class BABELAdapter
 * @brief Adapter for BABEL module (Memory Bandwidth)
 */
class BABELAdapter : public ModuleAdapter {
 public:
  BABELAdapter();
  ~BABELAdapter();

  ProbeResult execute_and_capture() override;
  std::string get_module_name() const override;
  std::string get_baseline_key() const override;
  void set_gpu_id(uint16_t gpu_id) override;

 protected:
  std::string generate_config() override;
  double parse_output(const std::string& output) override;
};

} // namespace gpim

#endif  // GPIM_SO_INCLUDE_MODULE_ADAPTER_H_
