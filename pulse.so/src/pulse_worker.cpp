/********************************************************************************
 *
 * Pulse worker: alternating GEMM / idle phases, SMI power & temperature,
 * optional end-of-run GEMM verify (verify_mode / tolerance; CPU accuracy
 * skipped for matrix_size > 2048), sample_interval throttling, multi-GPU
 * barriers on the default HIP stream, configurable max_temp_c.
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

#include <unistd.h>
#include <string>
#include <vector>
#include <iostream>
#include <chrono>
#include <memory>
#include <thread>
#include <cmath>
#include <algorithm>
#include <cctype>

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include "include/rvs_module.h"
#include "include/rvsloglp.h"
#include "include/pulse_worker.h"

#define MODULE_NAME                 "pulse"
#define PULSE_PASS_KEY              "pass"
#define PULSE_RESULT_PASS           "TRUE"
#define PULSE_RESULT_FAIL           "FALSE"
#define PULSE_JSON_LOG_GPU_ID_KEY   "gpu_id"
#define PULSE_JSON_POWER_HIGH_KEY   "power_high"
#define PULSE_JSON_POWER_LOW_KEY    "power_low"

using std::string;

bool PulseWorker::bjson = false;

static string pulse_mode_normalize(const string& in) {
  string out;
  for (char c : in) {
    if (!std::isspace(static_cast<unsigned char>(c)))
      out.push_back(static_cast<char>(
          std::tolower(static_cast<unsigned char>(c))));
  }
  return out;
}

/** Map verify_mode to rvs_blas::validate_gemm flags (GST-style semantics). */
static void pulse_verify_flags(const string& verify_mode,
    const string& ops_type, const string& data_type,
    bool& self_check, bool& accu_check) {
  self_check = false;
  accu_check = false;
  const string m = pulse_mode_normalize(verify_mode);
  if (m.empty() || m == "none" || m == "off" || m == "false")
    return;
  if (m == "crc") {
    self_check = true;
    return;
  }
  if (m == "diff") {
    if (ops_type == "sgemm" || ops_type == "dgemm") {
      accu_check = true;
    } else if (data_type == "fp16_r" || data_type == "bf16_r" ||
        data_type == "fp8_r" || ops_type == "hgemm") {
      self_check = true;
    } else {
      self_check = true;
    }
    return;
  }
  if (m == "both" || m == "full") {
    self_check = true;
    if (ops_type == "sgemm" || ops_type == "dgemm")
      accu_check = true;
    return;
  }
}

static uint64_t time_diff(
    std::chrono::time_point<std::chrono::system_clock> t_end,
    std::chrono::time_point<std::chrono::system_clock> t_start) {
  auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
      t_end - t_start);
  return milliseconds.count();
}

// GPU-side barrier kernel for synchronized avalanche release.
// Uses fine-grained coherent system memory allocated via hipHostMallocCoherent.
// All GPUs atomically signal arrival, then spin until the last GPU releases.
__global__ void gpu_sync_barrier_kernel(int32_t* arrival_count,
                                         int32_t* release_flag,
                                         int32_t  target_count) {
  int32_t arrived = atomicAdd_system(arrival_count, 1) + 1;

  if (arrived == target_count) {
    __atomic_store_n(release_flag, 1, __ATOMIC_RELEASE);
  } else {
    while (__atomic_load_n(release_flag, __ATOMIC_ACQUIRE) == 0) {
      // tight spin on GPU hardware — nanosecond resolution
    }
  }
}

PulseWorker::PulseWorker()
    : gpu_device_index(-1),
      smi_device_handle(nullptr),
      gpu_id(0),
      run_duration_ms(0),
      sample_interval(100),
      log_interval(1000),
      pulse_rate(2),
      high_phase_ratio(0.5f),
      tolerance(10.0f),
      max_temp_c(105.0f),
      matrix_size(4096),
      pulse_trans_a(0),
      pulse_trans_b(1),
      pulse_alpha_val(2.0f),
      pulse_beta_val(-1.0f),
      pulse_lda_offset(0),
      pulse_ldb_offset(0),
      pulse_ldc_offset(0),
      pulse_ldd_offset(0),
      workload_iterations(128),
      halt_on_error(false),
      pulse_hot_calls(1),
      num_gpus(1),
      worker_index(0),
      cpu_barrier(nullptr),
      gpu_arrival_count(nullptr),
      gpu_release_flag(nullptr),
      done_flag(nullptr),
      result(false) {
}

PulseWorker::~PulseWorker() {
}

float PulseWorker::read_power(void) {
  amdsmi_power_info_t pwr_info;
  amdsmi_status_t stat = amdsmi_get_power_info(smi_device_handle, &pwr_info);
  if (stat == AMDSMI_STATUS_SUCCESS) {
    return static_cast<float>(pwr_info.socket_power);
  }
  return -1.0f;
}

// amdsmi_get_temp_metric: some stacks return millidegree Celsius (e.g. 43000
// for 43 °C, as in gm.so); others return whole degrees in the int64 (as in
// tst_worker). Values with magnitude above 1000 are treated as millidegrees.
static float amdsmi_temperature_to_celsius(int64_t raw) {
  const int64_t mag = raw >= 0 ? raw : -raw;
  if (mag > 1000)
    return static_cast<float>(raw) / 1000.0f;
  return static_cast<float>(raw);
}

float PulseWorker::read_temperature(void) {
  int64_t temp = 0;
  amdsmi_status_t stat = amdsmi_get_temp_metric(smi_device_handle,
      AMDSMI_TEMPERATURE_TYPE_JUNCTION, AMDSMI_TEMP_CURRENT, &temp);
  if (stat != AMDSMI_STATUS_SUCCESS) {
    stat = amdsmi_get_temp_metric(smi_device_handle,
        AMDSMI_TEMPERATURE_TYPE_EDGE, AMDSMI_TEMP_CURRENT, &temp);
  }
  if (stat == AMDSMI_STATUS_SUCCESS) {
    return amdsmi_temperature_to_celsius(temp);
  }
  return -1.0f;
}

bool PulseWorker::discover_valid_clock_levels(void) {
  amdsmi_frequencies_t freqs{};
  string msg;

  // Discover GFX clock levels — brief sleep after each set_clk_freq so the
  // driver can apply the request before the next probe.
  if (amdsmi_get_clk_freq(smi_device_handle, AMDSMI_CLK_TYPE_SYS, &freqs)
      == AMDSMI_STATUS_SUCCESS) {
    for (uint32_t level = 0; level < freqs.num_supported; ++level) {
      uint64_t test_mask = (1ULL << level);
      if (amdsmi_set_clk_freq(smi_device_handle, AMDSMI_CLK_TYPE_SYS,
            test_mask) == AMDSMI_STATUS_SUCCESS) {
        valid_gfx_levels.push_back(level);
      } else {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    // Restore all valid levels after probing
    if (!valid_gfx_levels.empty()) {
      uint64_t restore = 0;
      for (auto lvl : valid_gfx_levels) restore |= (1ULL << lvl);
      amdsmi_set_clk_freq(smi_device_handle, AMDSMI_CLK_TYPE_SYS, restore);
    }
  }

  // Discover MEM clock levels
  if (amdsmi_get_clk_freq(smi_device_handle, AMDSMI_CLK_TYPE_MEM, &freqs)
      == AMDSMI_STATUS_SUCCESS) {
    for (uint32_t level = 0; level < freqs.num_supported; ++level) {
      uint64_t test_mask = (1ULL << level);
      if (amdsmi_set_clk_freq(smi_device_handle, AMDSMI_CLK_TYPE_MEM,
            test_mask) == AMDSMI_STATUS_SUCCESS) {
        valid_mem_levels.push_back(level);
      } else {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    if (!valid_mem_levels.empty()) {
      uint64_t restore = 0;
      for (auto lvl : valid_mem_levels) restore |= (1ULL << lvl);
      amdsmi_set_clk_freq(smi_device_handle, AMDSMI_CLK_TYPE_MEM, restore);
    }
  }

  msg = "[" + action_name + "] " + MODULE_NAME + " " +
    std::to_string(gpu_id) + " discovered " +
    std::to_string(valid_gfx_levels.size()) + " GFX levels, " +
    std::to_string(valid_mem_levels.size()) + " MEM levels";
  rvs::lp::Log(msg, rvs::loginfo);

  return !valid_gfx_levels.empty();
}

bool PulseWorker::set_highest_clocks(void) {
  bool ok = true;
  amdsmi_frequencies_t freqs{};

  if (!valid_gfx_levels.empty()) {
    if (amdsmi_get_clk_freq(smi_device_handle, AMDSMI_CLK_TYPE_SYS, &freqs)
        == AMDSMI_STATUS_SUCCESS) {
      uint32_t max_idx = valid_gfx_levels[0];
      for (auto level : valid_gfx_levels) {
        if (freqs.frequency[level] > freqs.frequency[max_idx])
          max_idx = level;
      }
      uint64_t mask = (1ULL << max_idx);
      if (amdsmi_set_clk_freq(smi_device_handle, AMDSMI_CLK_TYPE_SYS, mask)
          != AMDSMI_STATUS_SUCCESS) {
        ok = false;
      }
    }
  }

  if (!valid_mem_levels.empty()) {
    if (amdsmi_get_clk_freq(smi_device_handle, AMDSMI_CLK_TYPE_MEM, &freqs)
        == AMDSMI_STATUS_SUCCESS) {
      uint32_t max_idx = valid_mem_levels[0];
      for (auto level : valid_mem_levels) {
        if (freqs.frequency[level] > freqs.frequency[max_idx])
          max_idx = level;
      }
      uint64_t mask = (1ULL << max_idx);
      if (amdsmi_set_clk_freq(smi_device_handle, AMDSMI_CLK_TYPE_MEM, mask)
          != AMDSMI_STATUS_SUCCESS) {
        ok = false;
      }
    }
  }

  return ok;
}

bool PulseWorker::set_lowest_clocks(void) {
  bool ok = true;
  amdsmi_frequencies_t freqs{};

  if (!valid_gfx_levels.empty()) {
    if (amdsmi_get_clk_freq(smi_device_handle, AMDSMI_CLK_TYPE_SYS, &freqs)
        == AMDSMI_STATUS_SUCCESS) {
      uint32_t min_idx = valid_gfx_levels[0];
      for (auto level : valid_gfx_levels) {
        if (freqs.frequency[level] < freqs.frequency[min_idx])
          min_idx = level;
      }
      uint64_t mask = (1ULL << min_idx);
      if (amdsmi_set_clk_freq(smi_device_handle, AMDSMI_CLK_TYPE_SYS, mask)
          != AMDSMI_STATUS_SUCCESS) {
        ok = false;
      }
    }
  }

  if (!valid_mem_levels.empty()) {
    if (amdsmi_get_clk_freq(smi_device_handle, AMDSMI_CLK_TYPE_MEM, &freqs)
        == AMDSMI_STATUS_SUCCESS) {
      uint32_t min_idx = valid_mem_levels[0];
      for (auto level : valid_mem_levels) {
        if (freqs.frequency[level] < freqs.frequency[min_idx])
          min_idx = level;
      }
      uint64_t mask = (1ULL << min_idx);
      if (amdsmi_set_clk_freq(smi_device_handle, AMDSMI_CLK_TYPE_MEM, mask)
          != AMDSMI_STATUS_SUCCESS) {
        ok = false;
      }
    }
  }

  return ok;
}

bool PulseWorker::restore_clocks(void) {
  bool ok = true;

  if (!valid_gfx_levels.empty()) {
    uint64_t mask = 0;
    for (auto lvl : valid_gfx_levels) mask |= (1ULL << lvl);
    if (amdsmi_set_clk_freq(smi_device_handle, AMDSMI_CLK_TYPE_SYS, mask)
        != AMDSMI_STATUS_SUCCESS) {
      ok = false;
    }
  }

  if (!valid_mem_levels.empty()) {
    uint64_t mask = 0;
    for (auto lvl : valid_mem_levels) mask |= (1ULL << lvl);
    if (amdsmi_set_clk_freq(smi_device_handle, AMDSMI_CLK_TYPE_MEM, mask)
        != AMDSMI_STATUS_SUCCESS) {
      ok = false;
    }
  }

  return ok;
}

bool PulseWorker::setup_blas(void) {
  int m = static_cast<int>(matrix_size);
  int n = static_cast<int>(matrix_size);
  int k = static_cast<int>(matrix_size);

  gpu_blas = std::unique_ptr<rvs_blas>(new rvs_blas(
      gpu_device_index,
      m, n, k,
      matrix_init,
      pulse_trans_a, pulse_trans_b,
      pulse_alpha_val, pulse_beta_val,
      pulse_lda_offset, pulse_ldb_offset,
      pulse_ldc_offset, pulse_ldd_offset,
      pulse_ops_type, pulse_data_type,
      "", 0,
      0, 0, 0, 0,
      blas_source, compute_type,
      pulse_out_data_type,
      "", "", 0,
      pulse_hot_calls));

  gpu_blas->generate_random_matrix_data();
  if (!gpu_blas->copy_data_to_gpu()) {
    return false;
  }
  return true;
}

bool PulseWorker::gpu_barrier_sync(bool time_up, bool& test_passed) {
  if (!cpu_barrier || !gpu_arrival_count || !gpu_release_flag)
    return !time_up;

  // Signal done BEFORE barrier so all threads see it atomically
  if (time_up && done_flag)
    done_flag->store(true, std::memory_order_release);

  // Level 1: CPU barrier — all threads MUST arrive even if done,
  // otherwise the remaining threads deadlock here forever.
  cpu_barrier->arrive_and_wait();

  // After barrier, if any thread signaled done, all exit together
  if (done_flag && done_flag->load(std::memory_order_acquire))
    return false;

  // Reset sync counters (first worker resets, others wait)
  if (worker_index == 0) {
    __atomic_store_n(gpu_arrival_count, 0, __ATOMIC_RELEASE);
    __atomic_store_n(gpu_release_flag, 0, __ATOMIC_RELEASE);
  }
  cpu_barrier->arrive_and_wait();

  if (gpu_device_index >= 0)
    hipSetDevice(gpu_device_index);

  // Default stream + device sync (same pattern as pre-stream refactor).
  gpu_sync_barrier_kernel<<<1, 1>>>(
      gpu_arrival_count, gpu_release_flag, num_gpus);
  if (hipDeviceSynchronize() != hipSuccess) {
    string msg = "[" + action_name + "] " + MODULE_NAME + " " +
      std::to_string(gpu_id) + " hipDeviceSynchronize failed after "
      "GPU barrier kernel";
    rvs::lp::Err(msg, "PULSE", action_name);
    test_passed = false;
  }
  return true;
}

bool PulseWorker::run_gemm_verify(bool& test_passed) {
  bool self_check = false;
  bool accu_check = false;
  pulse_verify_flags(verify_mode, pulse_ops_type, pulse_data_type,
      self_check, accu_check);
  if (!self_check && !accu_check)
    return true;

  // CPU accuracy reference is O(n³) — do not run per pulse on large matrices.
  constexpr uint64_t kMaxAccuMatrixDim = 2048ULL;
  if (accu_check && matrix_size > kMaxAccuMatrixDim) {
    accu_check = false;
    if (!self_check)
      self_check = true;
  }

  if (gpu_device_index >= 0)
    hipSetDevice(gpu_device_index);

  double self_error = 0.0;
  double accu_error = 0.0;
  if (!gpu_blas->validate_gemm(self_check, accu_check, self_error,
          accu_error)) {
    string msg = "[" + action_name + "] " + MODULE_NAME + " " +
      std::to_string(gpu_id) + " GEMM validate_gemm failed (unsupported "
      "combination for verify_mode)";
    rvs::lp::Err(msg, "PULSE", action_name);
    test_passed = false;
    return false;
  }

  const double tol = static_cast<double>(tolerance);
  if (self_check && self_error > tol) {
    string msg = "[" + action_name + "] " + MODULE_NAME + " " +
      std::to_string(gpu_id) + " GEMM self-check error " +
      std::to_string(self_error) + " exceeds tolerance " +
      std::to_string(tol);
    rvs::lp::Err(msg, "PULSE", action_name);
    test_passed = false;
    return false;
  }
  if (accu_check && accu_error > tol) {
    string msg = "[" + action_name + "] " + MODULE_NAME + " " +
      std::to_string(gpu_id) + " GEMM accuracy error " +
      std::to_string(accu_error) + " exceeds tolerance " +
      std::to_string(tol);
    rvs::lp::Err(msg, "PULSE", action_name);
    test_passed = false;
    return false;
  }
  return true;
}

bool PulseWorker::do_pulse_stress(void) {
  std::chrono::time_point<std::chrono::system_clock> test_start, phase_start,
    now, last_log_time, last_sample_time;
  string msg;
  char gpuid_buff[12];
  rvs::action_result_t action_result;
  auto desc = action_descriptor{action_name, MODULE_NAME, gpu_id};

  snprintf(gpuid_buff, sizeof(gpuid_buff), "%5d", gpu_id);

  if (!setup_blas()) {
    msg = "[" + action_name + "] " + MODULE_NAME + " " +
      std::to_string(gpu_id) + " BLAS setup failed";
    rvs::lp::Err(msg, "PULSE", action_name);
    result = false;
    return false;
  }

  discover_valid_clock_levels();

  float max_power_high = 0.0f;
  float min_power_low = 999999.0f;
  float total_power_high = 0.0f;
  float total_power_low = 0.0f;
  int high_samples = 0;
  int low_samples = 0;
  int pulse_count = 0;
  bool test_passed = true;

  // Phase durations in milliseconds.
  // For visible power deltas use pulse_rate 1-10 Hz (100ms-1s phases).
  // Sub-10ms phases are too fast for real GPU power state transitions
  // and the SMI reporting window (~100ms averaging).
  double period_ms = 1000.0 / static_cast<double>(pulse_rate);
  double high_phase_ms = period_ms * high_phase_ratio;
  double low_phase_ms = period_ms * (1.0 - high_phase_ratio);

  high_phase_ms = std::max(high_phase_ms, 10.0);
  low_phase_ms = std::max(low_phase_ms, 10.0);

  const uint64_t smpl_ms = std::max<uint64_t>(1ULL, sample_interval);

  msg = "[" + action_name + "] " + MODULE_NAME + " " +
    std::to_string(gpu_id) + " pulse_rate=" + std::to_string(pulse_rate) +
    "Hz period=" + std::to_string(period_ms) +
    "ms high=" + std::to_string(high_phase_ms) +
    "ms low=" + std::to_string(low_phase_ms) + "ms";
  rvs::lp::Log(msg, rvs::loginfo);

  test_start = std::chrono::system_clock::now();
  last_log_time = test_start;

  while (!rvs::lp::Stopping()) {
    now = std::chrono::system_clock::now();
    uint64_t elapsed_ms = time_diff(now, test_start);
    bool time_up = (run_duration_ms > 0 && elapsed_ms >= run_duration_ms);

    // Coordinated shutdown: when using multi-GPU barrier, all GPUs must
    // arrive at the barrier even if their timer expired — otherwise the
    // remaining GPUs deadlock.  After the barrier, if ANY GPU signaled
    // done, ALL GPUs exit together.
    if (cpu_barrier && num_gpus > 1) {
      if (!gpu_barrier_sync(time_up, test_passed))
        break;
    } else if (time_up) {
      break;
    }

    float pulse_peak_high = 0.0f;
    float pulse_trough_low = 999999.0f;
    float pulse_temp = -1.0f;
    int pulse_gemm_count = 0;

    // ═══ HIGH PHASE: Pin clocks to max + sustained GEMM load ═══
    set_highest_clocks();

    phase_start = std::chrono::system_clock::now();
    last_sample_time = phase_start;

    while (true) {
      now = std::chrono::system_clock::now();
      double phase_elapsed = std::chrono::duration<double, std::milli>(
          now - phase_start).count();
      if (phase_elapsed >= high_phase_ms)
        break;

      for (int iter = 0; iter < workload_iterations && !rvs::lp::Stopping();
           ++iter) {
        if (!gpu_blas->run_blas_gemm(true)) {
          test_passed = false;
          if (halt_on_error) goto done;
          break;
        }
        if (!gpu_blas->is_gemm_op_complete()) {
          test_passed = false;
          if (halt_on_error) goto done;
          break;
        }
        pulse_gemm_count++;
        now = std::chrono::system_clock::now();
        if (std::chrono::duration<double, std::milli>(
                now - phase_start).count() >= high_phase_ms)
          break;
      }

      now = std::chrono::system_clock::now();
      if (time_diff(now, last_sample_time) >= smpl_ms) {
        float power = read_power();
        if (power > 0) {
          total_power_high += power;
          high_samples++;
          if (power > max_power_high) max_power_high = power;
          if (power > pulse_peak_high) pulse_peak_high = power;
        }

        float temp = read_temperature();
        if (temp > 0) pulse_temp = temp;
        if (max_temp_c > 0.0f && temp > max_temp_c) {
          msg = "[" + action_name + "] " + MODULE_NAME + " " +
            std::to_string(gpu_id) + " thermal violation: " +
            std::to_string(temp) + "C (limit " + std::to_string(max_temp_c) +
            "C)";
          rvs::lp::Log(msg, rvs::logerror);
          test_passed = false;
          if (halt_on_error) goto done;
        }
        last_sample_time = now;
      }
    }

    // Drain the GPU pipeline so all GEMM work finishes before the
    // low phase — without this, kernels still in flight keep power
    // elevated and the "low" reading is indistinguishable from "high".
    hipDeviceSynchronize();
    now = std::chrono::system_clock::now();
    double actual_high_ms = std::chrono::duration<double, std::milli>(
        now - phase_start).count();

    // ═══ LOW PHASE: Pin clocks to minimum + idle with sampling ═══
    set_lowest_clocks();

    phase_start = std::chrono::system_clock::now();
    last_sample_time = phase_start;
    while (true) {
      now = std::chrono::system_clock::now();
      double phase_elapsed = std::chrono::duration<double, std::milli>(
          now - phase_start).count();
      if (phase_elapsed >= low_phase_ms)
        break;

      double remain_ms = low_phase_ms - phase_elapsed;
      unsigned sleep_chunk = static_cast<unsigned>(
          std::min(remain_ms, static_cast<double>(smpl_ms)));
      if (sleep_chunk < 1u)
        sleep_chunk = 1u;
      std::this_thread::sleep_for(std::chrono::milliseconds(sleep_chunk));

      now = std::chrono::system_clock::now();
      if (time_diff(now, last_sample_time) >= smpl_ms) {
        float power_low = read_power();
        if (power_low > 0) {
          total_power_low += power_low;
          low_samples++;
          if (power_low < min_power_low) min_power_low = power_low;
          if (power_low < pulse_trough_low) pulse_trough_low = power_low;
        }
        last_sample_time = now;
      }
    }

    now = std::chrono::system_clock::now();
    double actual_low_ms = std::chrono::duration<double, std::milli>(
        now - phase_start).count();

    pulse_count++;

    if (PulseWorker::bjson) {
      uint64_t pulse_elapsed_ms = time_diff(now, test_start);
      log_to_json(desc, rvs::logresults,
          "record_type", "pulse",
          "pulse_num", std::to_string(pulse_count),
          "elapsed_ms", std::to_string(pulse_elapsed_ms),
          "power_high_w", std::to_string(pulse_peak_high),
          "power_low_w", std::to_string(
              pulse_trough_low < 999999.0f ? pulse_trough_low : 0.0f),
          "power_delta_w", std::to_string(
              pulse_peak_high - (pulse_trough_low < 999999.0f
                  ? pulse_trough_low : 0.0f)),
          "high_duration_ms", std::to_string(actual_high_ms),
          "low_duration_ms", std::to_string(actual_low_ms),
          "temp_c", std::to_string(pulse_temp),
          "gemm_count", std::to_string(pulse_gemm_count));
    }
    if (time_diff(now, last_log_time) >= log_interval) {
      float avg_high = (high_samples > 0)
        ? total_power_high / high_samples : 0.0f;
      float avg_low = (low_samples > 0)
        ? total_power_low / low_samples : 0.0f;
      msg = "[" + action_name + "] [GPU:: " + gpuid_buff + "] " +
        "pulse #" + std::to_string(pulse_count) +
        " avg_high=" + std::to_string(avg_high) + "W" +
        " avg_low=" + std::to_string(avg_low) + "W" +
        " max_high=" + std::to_string(max_power_high) + "W" +
        " min_low=" + std::to_string(min_power_low) + "W" +
        " delta=" + std::to_string(avg_high - avg_low) + "W";
      rvs::lp::Log(msg, rvs::logresults);
      last_log_time = now;
    }

    if (rvs::lp::Stopping())
      break;
  }

done:
  if (gpu_blas && gpu_device_index >= 0) {
    hipSetDevice(gpu_device_index);
    hipDeviceSynchronize();
    (void)run_gemm_verify(test_passed);
  }
  restore_clocks();

  float avg_power_high = (high_samples > 0)
    ? total_power_high / high_samples : 0.0f;
  float avg_power_low = (low_samples > 0)
    ? total_power_low / low_samples : 0.0f;
  float power_delta = avg_power_high - avg_power_low;

  msg = "[" + action_name + "] [GPU:: " + gpuid_buff + "] " +
    "completed " + std::to_string(pulse_count) + " pulses" +
    " avg_high=" + std::to_string(avg_power_high) + "W" +
    " avg_low=" + std::to_string(avg_power_low) + "W" +
    " delta=" + std::to_string(power_delta) + "W" +
    " max_high=" + std::to_string(max_power_high) + "W" +
    " min_low=" + std::to_string(min_power_low) + "W";
  rvs::lp::Log(msg, rvs::logresults);

  if (mcm_type == mcm_type_t::SECONDARY) {
    msg = "[" + action_name + "] " + MODULE_NAME + " " +
      std::to_string(gpu_id) +
      " secondary MCM, considering pass by default";
    rvs::lp::Log(msg, rvs::loginfo);
    result = true;
  } else {
    result = test_passed && (pulse_count > 0);
  }

  if (PulseWorker::bjson) {
    log_to_json(desc, rvs::logresults,
        "record_type", "summary",
        "pulse_count", std::to_string(pulse_count),
        "avg_power_high_w", std::to_string(avg_power_high),
        "avg_power_low_w", std::to_string(avg_power_low),
        "avg_delta_w", std::to_string(power_delta),
        "max_power_high_w", std::to_string(max_power_high),
        "min_power_low_w", std::to_string(min_power_low),
        "duration_ms", std::to_string(run_duration_ms),
        "pulse_rate_hz", std::to_string(pulse_rate),
        "ops_type", pulse_ops_type,
        "matrix_size", std::to_string(matrix_size),
        PULSE_PASS_KEY, result ? "true" : "false");
  }

  action_result.state = rvs::actionstate::ACTION_RUNNING;
  action_result.status = result
    ? rvs::actionstatus::ACTION_SUCCESS
    : rvs::actionstatus::ACTION_FAILED;
  action_result.output = msg;
  action.action_callback(&action_result);

  return result;
}

void PulseWorker::run() {
  string msg;
  char gpuid_buff[12];

  msg = "[" + action_name + "] " + MODULE_NAME + " " +
    std::to_string(gpu_id) + " start pulse_rate=" +
    std::to_string(pulse_rate);
  rvs::lp::Log(msg, rvs::loginfo);

  bool pass = do_pulse_stress();

  if (rvs::lp::Stopping())
    return;

  snprintf(gpuid_buff, sizeof(gpuid_buff), "%5d", gpu_id);
  msg = "[" + action_name + "] [GPU:: " + gpuid_buff + "] " +
    PULSE_PASS_KEY + ": " +
    (pass ? PULSE_RESULT_PASS : PULSE_RESULT_FAIL);
  rvs::lp::Log(msg, rvs::logresults);

  sleep(2);
}
