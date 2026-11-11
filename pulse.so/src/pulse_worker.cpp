/********************************************************************************
 *
 * Pulse worker: built-in power waveforms, timed GEMM and barrier waits,
 * and explicit pass/fail (workload, thermal, ECC, optional power swing).
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
#include <sstream>

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include "include/rvs_module.h"
#include "include/rvsloglp.h"
#include "include/pulse_worker.h"
#include "include/pulse_pattern.h"

#define MODULE_NAME                 "pulse"
#define PULSE_PASS_KEY              "pass"
#define PULSE_RESULT_PASS           "TRUE"
#define PULSE_RESULT_FAIL           "FALSE"
#define PULSE_JSON_LOG_GPU_ID_KEY   "gpu_id"

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
    if (ops_type == "sgemm" || ops_type == "dgemm")
      accu_check = true;
    else
      self_check = true;
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
  return std::chrono::duration_cast<std::chrono::milliseconds>(
      t_end - t_start).count();
}

static int64_t steady_ms_since(
    std::chrono::time_point<std::chrono::steady_clock> t0) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - t0).count();
}

// Device barrier. Host memory is fine-grained. A timeout stores the release
// flag so every spinner leaves; the host also does this if the kernel is late.
__global__ void gpu_sync_barrier_kernel(int32_t* arrival_count,
                                         int32_t* release_flag,
                                         int32_t* timeout_flag,
                                         int32_t target_count,
                                         uint64_t timeout_cycles) {
  int32_t arrived = atomicAdd_system(arrival_count, 1) + 1;
  if (arrived == target_count) {
    __atomic_store_n(release_flag, 1, __ATOMIC_RELEASE);
    return;
  }
  const uint64_t start = wall_clock64();
  while (__atomic_load_n(release_flag, __ATOMIC_ACQUIRE) == 0) {
    if (timeout_cycles > 0 && (wall_clock64() - start) > timeout_cycles) {
      __atomic_store_n(timeout_flag, 1, __ATOMIC_RELEASE);
      __atomic_store_n(release_flag, 1, __ATOMIC_RELEASE);
      return;
    }
  }
}

__global__ void pulse_read_wall(uint64_t* out) {
  *out = wall_clock64();
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
      min_power_delta_w(0.0f),
      ecc_check(true),
      gpu_sync_wait_ms(10000),
      pattern_duration_ms(60000),
      custom_pattern(false),
      matrix_m(4096),
      matrix_n(4096),
      matrix_k(4096),
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
      pulse_sync(nullptr),
      sync_epoch(0),
      local_abort(false),
      gemm_event(nullptr),
      gemm_event_ready(false),
      device_busy(false),
      clock_warned(false),
      wall_hz(25000000ULL),
      result(false) {
}

PulseWorker::~PulseWorker() {
  if (gemm_event_ready) {
    (void)hipEventDestroy(gemm_event);
    gemm_event_ready = false;
  }
}

bool PulseWorker::judge_results(void) const {
  return mcm_type != mcm_type_t::SECONDARY;
}

void PulseWorker::log_msg(const string& msg, int level) {
  if (level == rvs::logerror)
    rvs::lp::Err(msg, "PULSE", action_name);
  else
    rvs::lp::Log(msg, level);
}

void PulseWorker::note_failure(bool& test_passed) {
  if (judge_results())
    test_passed = false;
}

std::atomic<bool>& PulseWorker::abort_flag(void) {
  if (pulse_sync)
    return pulse_sync->abort_flag;
  return local_abort;
}

float PulseWorker::read_power(void) {
  if (!smi_device_handle)
    return -1.0f;
  amdsmi_power_info_t pwr_info;
  amdsmi_status_t stat = amdsmi_get_power_info(smi_device_handle, &pwr_info);
  if (stat == AMDSMI_STATUS_SUCCESS)
    return static_cast<float>(pwr_info.socket_power);
  return -1.0f;
}

static float amdsmi_temperature_to_celsius(int64_t raw) {
  const int64_t mag = raw >= 0 ? raw : -raw;
  if (mag > 1000)
    return static_cast<float>(raw) / 1000.0f;
  return static_cast<float>(raw);
}

float PulseWorker::read_temperature(void) {
  if (!smi_device_handle)
    return -1.0f;
  int64_t temp = 0;
  amdsmi_status_t stat = amdsmi_get_temp_metric(smi_device_handle,
      AMDSMI_TEMPERATURE_TYPE_JUNCTION, AMDSMI_TEMP_CURRENT, &temp);
  if (stat != AMDSMI_STATUS_SUCCESS) {
    stat = amdsmi_get_temp_metric(smi_device_handle,
        AMDSMI_TEMPERATURE_TYPE_EDGE, AMDSMI_TEMP_CURRENT, &temp);
  }
  if (stat == AMDSMI_STATUS_SUCCESS)
    return amdsmi_temperature_to_celsius(temp);
  return -1.0f;
}

bool PulseWorker::discover_valid_clock_levels(void) {
  if (!smi_device_handle)
    return false;
  amdsmi_frequencies_t freqs{};
  string msg;

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
    if (!valid_gfx_levels.empty()) {
      uint64_t restore = 0;
      for (auto lvl : valid_gfx_levels) restore |= (1ULL << lvl);
      amdsmi_set_clk_freq(smi_device_handle, AMDSMI_CLK_TYPE_SYS, restore);
    }
  }

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

void PulseWorker::warn_clock_once(const char* what) {
  if (clock_warned)
    return;
  clock_warned = true;
  log_msg("[" + action_name + "] " + MODULE_NAME + " " +
      std::to_string(gpu_id) + " clock control failed (" + what +
      "). Continuing without pinned clocks.", rvs::loginfo);
}

bool PulseWorker::set_highest_clocks(void) {
  bool ok = true;
  amdsmi_frequencies_t freqs{};
  if (!smi_device_handle)
    return false;

  if (!valid_gfx_levels.empty() &&
      amdsmi_get_clk_freq(smi_device_handle, AMDSMI_CLK_TYPE_SYS, &freqs)
        == AMDSMI_STATUS_SUCCESS) {
    uint32_t max_idx = valid_gfx_levels[0];
    for (auto level : valid_gfx_levels) {
      if (freqs.frequency[level] > freqs.frequency[max_idx])
        max_idx = level;
    }
    if (amdsmi_set_clk_freq(smi_device_handle, AMDSMI_CLK_TYPE_SYS,
          (1ULL << max_idx)) != AMDSMI_STATUS_SUCCESS)
      ok = false;
  }

  if (!valid_mem_levels.empty() &&
      amdsmi_get_clk_freq(smi_device_handle, AMDSMI_CLK_TYPE_MEM, &freqs)
        == AMDSMI_STATUS_SUCCESS) {
    uint32_t max_idx = valid_mem_levels[0];
    for (auto level : valid_mem_levels) {
      if (freqs.frequency[level] > freqs.frequency[max_idx])
        max_idx = level;
    }
    if (amdsmi_set_clk_freq(smi_device_handle, AMDSMI_CLK_TYPE_MEM,
          (1ULL << max_idx)) != AMDSMI_STATUS_SUCCESS)
      ok = false;
  }
  if (!ok)
    warn_clock_once("set highest");
  return ok;
}

bool PulseWorker::set_lowest_clocks(void) {
  bool ok = true;
  amdsmi_frequencies_t freqs{};
  if (!smi_device_handle)
    return false;

  if (!valid_gfx_levels.empty() &&
      amdsmi_get_clk_freq(smi_device_handle, AMDSMI_CLK_TYPE_SYS, &freqs)
        == AMDSMI_STATUS_SUCCESS) {
    uint32_t min_idx = valid_gfx_levels[0];
    for (auto level : valid_gfx_levels) {
      if (freqs.frequency[level] < freqs.frequency[min_idx])
        min_idx = level;
    }
    if (amdsmi_set_clk_freq(smi_device_handle, AMDSMI_CLK_TYPE_SYS,
          (1ULL << min_idx)) != AMDSMI_STATUS_SUCCESS)
      ok = false;
  }

  if (!valid_mem_levels.empty() &&
      amdsmi_get_clk_freq(smi_device_handle, AMDSMI_CLK_TYPE_MEM, &freqs)
        == AMDSMI_STATUS_SUCCESS) {
    uint32_t min_idx = valid_mem_levels[0];
    for (auto level : valid_mem_levels) {
      if (freqs.frequency[level] < freqs.frequency[min_idx])
        min_idx = level;
    }
    if (amdsmi_set_clk_freq(smi_device_handle, AMDSMI_CLK_TYPE_MEM,
          (1ULL << min_idx)) != AMDSMI_STATUS_SUCCESS)
      ok = false;
  }
  if (!ok)
    warn_clock_once("set lowest");
  return ok;
}

bool PulseWorker::restore_clocks(void) {
  bool ok = true;
  if (!smi_device_handle)
    return false;
  if (!valid_gfx_levels.empty()) {
    uint64_t mask = 0;
    for (auto lvl : valid_gfx_levels) mask |= (1ULL << lvl);
    if (amdsmi_set_clk_freq(smi_device_handle, AMDSMI_CLK_TYPE_SYS, mask)
        != AMDSMI_STATUS_SUCCESS)
      ok = false;
  }
  if (!valid_mem_levels.empty()) {
    uint64_t mask = 0;
    for (auto lvl : valid_mem_levels) mask |= (1ULL << lvl);
    if (amdsmi_set_clk_freq(smi_device_handle, AMDSMI_CLK_TYPE_MEM, mask)
        != AMDSMI_STATUS_SUCCESS)
      ok = false;
  }
  return ok;
}

bool PulseWorker::setup_blas(void) {
  gpu_blas = std::unique_ptr<rvs_blas>(new rvs_blas(
      gpu_device_index,
      static_cast<int>(matrix_m),
      static_cast<int>(matrix_n),
      static_cast<int>(matrix_k),
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
  if (!gpu_blas->copy_data_to_gpu())
    return false;
  return true;
}

void PulseWorker::calibrate_wall_clock(void) {
  wall_hz = 25000000ULL;
  if (gpu_device_index < 0)
    return;
  uint64_t* ticks = nullptr;
  if (hipHostMalloc(&ticks, sizeof(uint64_t), hipHostMallocCoherent) != hipSuccess)
    return;
  (void)hipSetDevice(gpu_device_index);
  pulse_read_wall<<<1, 1>>>(ticks);
  if (hipDeviceSynchronize() != hipSuccess) {
    (void)hipHostFree(ticks);
    return;
  }
  uint64_t t0 = *ticks;
  auto h0 = std::chrono::steady_clock::now();
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  pulse_read_wall<<<1, 1>>>(ticks);
  if (hipDeviceSynchronize() != hipSuccess) {
    (void)hipHostFree(ticks);
    return;
  }
  uint64_t t1 = *ticks;
  double sec = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - h0).count();
  (void)hipHostFree(ticks);
  if (sec > 0.0 && t1 > t0) {
    double hz = static_cast<double>(t1 - t0) / sec;
    if (hz > 1.0e6 && hz < 2.0e9)
      wall_hz = static_cast<uint64_t>(hz);
  }
}

PulseWorker::SyncStatus PulseWorker::host_rendezvous(void) {
  if (!pulse_sync || num_gpus < 2) {
    if (abort_flag().load(std::memory_order_acquire))
      return SyncStatus::kAbort;
    return SyncStatus::kGo;
  }

  const int my_epoch = sync_epoch;
  bool hit_deadline = false;
  int arrived = pulse_sync->arrival.fetch_add(1, std::memory_order_acq_rel) + 1;
  if (arrived >= pulse_sync->num_gpus) {
    pulse_sync->arrival.store(0, std::memory_order_release);
    pulse_sync->epoch.fetch_add(1, std::memory_order_acq_rel);
  } else {
    auto t0 = std::chrono::steady_clock::now();
    while (pulse_sync->epoch.load(std::memory_order_acquire) == my_epoch) {
      if (steady_ms_since(t0) >= pulse_sync->wait_ms) {
        hit_deadline = true;
        pulse_sync->abort_flag.store(true, std::memory_order_release);
        int expected = my_epoch;
        if (pulse_sync->epoch.compare_exchange_strong(expected, my_epoch + 1,
              std::memory_order_acq_rel)) {
          pulse_sync->arrival.store(0, std::memory_order_release);
        }
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  sync_epoch = pulse_sync->epoch.load(std::memory_order_acquire);
  if (hit_deadline)
    return SyncStatus::kTimeout;
  if (pulse_sync->abort_flag.load(std::memory_order_acquire))
    return SyncStatus::kAbort;
  return SyncStatus::kGo;
}

bool PulseWorker::wait_event(hipEvent_t ev, bool& test_passed, bool gemm_wait) {
  auto t0 = std::chrono::steady_clock::now();
  while (true) {
    hipError_t st = hipEventQuery(ev);
    if (st == hipSuccess)
      return true;
    if (st != hipErrorNotReady) {
      log_msg("[" + action_name + "] " + MODULE_NAME + " " +
          std::to_string(gpu_id) + " HIP event query failed", rvs::logerror);
      note_failure(test_passed);
      return false;
    }
    if (steady_ms_since(t0) >= gpu_sync_wait_ms) {
      if (gemm_wait)
        device_busy = true;
      log_msg("[" + action_name + "] " + MODULE_NAME + " " +
          std::to_string(gpu_id) + (gemm_wait
            ? " GEMM wait timed out"
            : " GPU barrier wait timed out"), rvs::logerror);
      note_failure(test_passed);
      if (judge_results())
        abort_flag().store(true, std::memory_order_release);
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

bool PulseWorker::wait_gemm(bool& test_passed) {
  if (!gemm_event_ready || !gpu_blas)
    return false;
  hipError_t st = hipEventRecord(gemm_event, gpu_blas->get_hip_stream());
  if (st != hipSuccess) {
    log_msg("[" + action_name + "] " + MODULE_NAME + " " +
        std::to_string(gpu_id) + " hipEventRecord failed", rvs::logerror);
    note_failure(test_passed);
    return false;
  }
  return wait_event(gemm_event, test_passed, true);
}

bool PulseWorker::gpu_barrier_sync(bool& test_passed) {
  if (!pulse_sync || !pulse_sync->gpu_arrival_count || num_gpus < 2)
    return true;

  if (worker_index == 0) {
    __atomic_store_n(pulse_sync->gpu_arrival_count, 0, __ATOMIC_RELEASE);
    __atomic_store_n(pulse_sync->gpu_release_flag, 0, __ATOMIC_RELEASE);
    __atomic_store_n(pulse_sync->gpu_timeout_flag, 0, __ATOMIC_RELEASE);
  }
  SyncStatus sr = host_rendezvous();
  if (sr == SyncStatus::kTimeout) {
    log_msg("[" + action_name + "] " + MODULE_NAME + " " +
        std::to_string(gpu_id) + " host rendezvous timed out", rvs::logerror);
    note_failure(test_passed);
    return false;
  }
  if (sr == SyncStatus::kAbort)
    return false;

  if (gpu_device_index >= 0)
    (void)hipSetDevice(gpu_device_index);

  const uint64_t timeout_cycles =
      wall_hz * static_cast<uint64_t>(std::max(gpu_sync_wait_ms, 1)) / 1000ULL;
  gpu_sync_barrier_kernel<<<1, 1>>>(
      pulse_sync->gpu_arrival_count,
      pulse_sync->gpu_release_flag,
      pulse_sync->gpu_timeout_flag,
      num_gpus,
      timeout_cycles);

  hipEvent_t barrier_ev = nullptr;
  if (hipEventCreateWithFlags(&barrier_ev, hipEventDisableTiming) != hipSuccess) {
    note_failure(test_passed);
    return false;
  }
  (void)hipEventRecord(barrier_ev, nullptr);
  bool done = wait_event(barrier_ev, test_passed, false);
  if (!done) {
    __atomic_store_n(pulse_sync->gpu_timeout_flag, 1, __ATOMIC_RELEASE);
    __atomic_store_n(pulse_sync->gpu_release_flag, 1, __ATOMIC_RELEASE);
    auto extra = std::chrono::steady_clock::now();
    while (steady_ms_since(extra) < 1000) {
      if (hipEventQuery(barrier_ev) == hipSuccess)
        break;
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
  }
  (void)hipEventDestroy(barrier_ev);
  if (!done)
    return false;
  if (__atomic_load_n(pulse_sync->gpu_timeout_flag, __ATOMIC_ACQUIRE) != 0) {
    log_msg("[" + action_name + "] " + MODULE_NAME + " " +
        std::to_string(gpu_id) + " GPU barrier timed out", rvs::logerror);
    note_failure(test_passed);
    if (judge_results())
      abort_flag().store(true, std::memory_order_release);
    return false;
  }
  return true;
}

bool PulseWorker::run_gemm_verify(bool& test_passed) {
  if (!gpu_blas || device_busy)
    return true;
  bool self_check = false;
  bool accu_check = false;
  pulse_verify_flags(verify_mode, pulse_ops_type, pulse_data_type,
      self_check, accu_check);
  if (!self_check && !accu_check)
    return true;

  constexpr uint64_t kMaxAccuMatrixDim = 2048ULL;
  uint64_t max_dim = std::max(matrix_m, std::max(matrix_n, matrix_k));
  if (accu_check && max_dim > kMaxAccuMatrixDim) {
    accu_check = false;
    if (!self_check)
      self_check = true;
  }

  if (gpu_device_index >= 0)
    (void)hipSetDevice(gpu_device_index);

  double self_error = 0.0;
  double accu_error = 0.0;
  if (!gpu_blas->validate_gemm(self_check, accu_check, self_error, accu_error)) {
    log_msg("[" + action_name + "] " + MODULE_NAME + " " +
        std::to_string(gpu_id) + " GEMM validate_gemm failed", rvs::logerror);
    note_failure(test_passed);
    return false;
  }

  const double tol = static_cast<double>(tolerance);
  if (self_check && self_error > tol) {
    log_msg("[" + action_name + "] " + MODULE_NAME + " " +
        std::to_string(gpu_id) + " GEMM self-check error " +
        std::to_string(self_error) + " exceeds tolerance " +
        std::to_string(tol), rvs::logerror);
    note_failure(test_passed);
    return false;
  }
  if (accu_check && accu_error > tol) {
    log_msg("[" + action_name + "] " + MODULE_NAME + " " +
        std::to_string(gpu_id) + " GEMM accuracy error " +
        std::to_string(accu_error) + " exceeds tolerance " +
        std::to_string(tol), rvs::logerror);
    note_failure(test_passed);
    return false;
  }
  return true;
}

bool PulseWorker::read_uncorrectable_ecc(uint64_t& count) {
  if (!smi_device_handle)
    return false;
  amdsmi_error_count_t ec{};
  if (amdsmi_get_gpu_total_ecc_count(smi_device_handle, &ec) != AMDSMI_STATUS_SUCCESS)
    return false;
  count = ec.uncorrectable_count;
  return true;
}

void PulseWorker::check_temperature(float temp, bool& test_passed) {
  if (temp < 0.0f)
    return;
  if (max_temp_c > 0.0f && temp > max_temp_c) {
    log_msg("[" + action_name + "] " + MODULE_NAME + " " +
        std::to_string(gpu_id) + " thermal violation: " +
        std::to_string(temp) + "C (limit " + std::to_string(max_temp_c) + "C)",
        rvs::logerror);
    note_failure(test_passed);
  }
}

bool PulseWorker::do_pulse_stress(void) {
  string msg;
  char gpuid_buff[12];
  rvs::action_result_t action_result;
  auto desc = action_descriptor{action_name, MODULE_NAME, gpu_id};
  snprintf(gpuid_buff, sizeof(gpuid_buff), "%5d", gpu_id);

  if (gpu_device_index >= 0)
    (void)hipSetDevice(gpu_device_index);

  if (!smi_device_handle) {
    log_msg("[" + action_name + "] " + MODULE_NAME + " " +
        std::to_string(gpu_id) + " has no SMI handle", rvs::logerror);
    if (judge_results()) {
      abort_flag().store(true, std::memory_order_release);
      host_rendezvous();
      result = false;
      return false;
    }
  }

  bool blas_ok = setup_blas();
  if (!blas_ok) {
    log_msg("[" + action_name + "] " + MODULE_NAME + " " +
        std::to_string(gpu_id) + " BLAS setup failed", rvs::logerror);
    if (judge_results()) {
      abort_flag().store(true, std::memory_order_release);
      host_rendezvous();
      result = false;
      return false;
    }
  }

  if (hipEventCreateWithFlags(&gemm_event, hipEventDisableTiming) == hipSuccess)
    gemm_event_ready = true;
  else if (judge_results()) {
    log_msg("[" + action_name + "] " + MODULE_NAME + " " +
        std::to_string(gpu_id) + " hipEventCreate failed", rvs::logerror);
    abort_flag().store(true, std::memory_order_release);
    host_rendezvous();
    result = false;
    return false;
  }

  discover_valid_clock_levels();
  if (pulse_sync && num_gpus > 1)
    calibrate_wall_clock();

  bool ecc_supported = false;
  uint64_t ecc_base = 0;
  if (ecc_check && smi_device_handle) {
    if (read_uncorrectable_ecc(ecc_base))
      ecc_supported = true;
    else
      log_msg("[" + action_name + "] " + MODULE_NAME + " " +
          std::to_string(gpu_id) +
          " ECC count is not available; ECC check skipped", rvs::loginfo);
  }

  if (smi_device_handle) {
    float idle_power = read_power();
    float peak_activity = 0.0f;
    bool activity_known = false;
    auto idle_start = std::chrono::steady_clock::now();
    while (steady_ms_since(idle_start) < 1000) {
      amdsmi_engine_usage_t usage{};
      if (amdsmi_get_gpu_activity(smi_device_handle, &usage) == AMDSMI_STATUS_SUCCESS &&
          usage.gfx_activity != 0x0000FFFFu) {
        activity_known = true;
        if (usage.gfx_activity > peak_activity)
          peak_activity = static_cast<float>(usage.gfx_activity);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    log_msg("[" + action_name + "] " + MODULE_NAME + " " +
        std::to_string(gpu_id) + " idle power " +
        std::to_string(idle_power) + "W", rvs::loginfo);
    if (activity_known && peak_activity > 10.0f) {
      log_msg("[" + action_name + "] " + MODULE_NAME + " " +
          std::to_string(gpu_id) + " GPU was not idle before the run (gfx " +
          std::to_string(peak_activity) + "%)", rvs::loginfo);
    }
  }

  std::vector<int> ids = pattern_ids;
  if (custom_pattern)
    ids = std::vector<int>{kPulsePatternCustom};

  bool test_passed = true;
  bool stop_patterns = false;
  int total_cycles = 0;
  std::string patterns_ran;
  const uint64_t smpl_ms = std::max<uint64_t>(1ULL, sample_interval);

  for (int pid : ids) {
    if (stop_patterns || rvs::lp::Stopping())
      break;
    if (abort_flag().load(std::memory_order_acquire))
      break;

    const char* pname = pulse_pattern_name(pid);
    if (!patterns_ran.empty())
      patterns_ran += ",";
    patterns_ran += pname;

    float total_high = 0.0f, total_low = 0.0f;
    float max_high = 0.0f, min_low = 999999.0f;
    float peak_temp = -1.0f;
    int high_samples = 0, low_samples = 0, cycles = 0;
    bool pattern_pass = true;
    auto pattern_start = std::chrono::steady_clock::now();
    auto last_log = std::chrono::system_clock::now();

    log_msg("[" + action_name + "] " + MODULE_NAME + " " +
        std::to_string(gpu_id) + " pattern " + std::to_string(pid) +
        " " + pname + " duration_ms=" +
        std::to_string(pattern_duration_ms), rvs::loginfo);

    while (!rvs::lp::Stopping()) {
      if (steady_ms_since(pattern_start) >=
          static_cast<int64_t>(pattern_duration_ms))
        break;

      SyncStatus sr = host_rendezvous();
      if (sr == SyncStatus::kTimeout) {
        log_msg("[" + action_name + "] " + MODULE_NAME + " " +
            std::to_string(gpu_id) + " host rendezvous timed out",
            rvs::logerror);
        note_failure(pattern_pass);
        stop_patterns = true;
        break;
      }
      if (sr == SyncStatus::kAbort ||
          abort_flag().load(std::memory_order_acquire)) {
        stop_patterns = true;
        break;
      }

      if (!gpu_barrier_sync(pattern_pass)) {
        stop_patterns = true;
        break;
      }

      PulseWave wave = pulse_wave(pid,
          static_cast<double>(steady_ms_since(pattern_start)),
          static_cast<double>(pattern_duration_ms),
          static_cast<double>(pulse_rate),
          static_cast<double>(high_phase_ratio));

      float pulse_peak = 0.0f;
      float pulse_trough = 999999.0f;
      float pulse_temp = -1.0f;
      int gemm_count = 0;
      bool stop_cycle = false;

      set_highest_clocks();
      auto phase_start = std::chrono::steady_clock::now();
      auto last_sample = phase_start;
      while (!stop_cycle) {
        if (steady_ms_since(phase_start) >= static_cast<int64_t>(wave.high_ms))
          break;
        if (!blas_ok || device_busy) {
          double remain = wave.high_ms -
              static_cast<double>(steady_ms_since(phase_start));
          unsigned chunk = static_cast<unsigned>(
              std::min(std::max(remain, 1.0), static_cast<double>(smpl_ms)));
          std::this_thread::sleep_for(std::chrono::milliseconds(chunk));
        } else {
        for (int iter = 0; iter < workload_iterations && !rvs::lp::Stopping();
             ++iter) {
          if (steady_ms_since(phase_start) >= static_cast<int64_t>(wave.high_ms))
            break;
          if (!gpu_blas->run_blas_gemm(1)) {
            log_msg("[" + action_name + "] " + MODULE_NAME + " " +
                std::to_string(gpu_id) + " GEMM enqueue failed", rvs::logerror);
            note_failure(pattern_pass);
            if (halt_on_error && judge_results()) {
              abort_flag().store(true, std::memory_order_release);
              stop_cycle = true;
            } else {
              std::this_thread::sleep_for(std::chrono::milliseconds(smpl_ms));
            }
            break;
          }
          if (!wait_gemm(pattern_pass)) {
            if (judge_results()) {
              stop_cycle = true;
              stop_patterns = true;
            }
            break;
          }
          gemm_count++;
        }
        }
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(
              now - last_sample).count() >= static_cast<int64_t>(smpl_ms)) {
          float power = read_power();
          if (power > 0.0f) {
            total_high += power;
            high_samples++;
            if (power > max_high) max_high = power;
            if (power > pulse_peak) pulse_peak = power;
          }
          float temp = read_temperature();
          if (temp > peak_temp) peak_temp = temp;
          if (temp > pulse_temp) pulse_temp = temp;
          check_temperature(temp, pattern_pass);
          if (!pattern_pass && halt_on_error && judge_results()) {
            abort_flag().store(true, std::memory_order_release);
            stop_cycle = true;
          }
          last_sample = now;
        }
        if (steady_ms_since(phase_start) >= static_cast<int64_t>(wave.high_ms))
          break;
      }

      if (!device_busy && gpu_device_index >= 0) {
        hipError_t herr = hipGetLastError();
        if (herr != hipSuccess) {
          log_msg("[" + action_name + "] " + MODULE_NAME + " " +
              std::to_string(gpu_id) + " HIP error after high phase",
              rvs::logerror);
          note_failure(pattern_pass);
          if (halt_on_error && judge_results()) {
            abort_flag().store(true, std::memory_order_release);
            stop_cycle = true;
            stop_patterns = true;
          }
        }
      }

      double actual_high_ms = static_cast<double>(steady_ms_since(phase_start));
      set_lowest_clocks();
      phase_start = std::chrono::steady_clock::now();
      last_sample = phase_start;
      while (!stop_patterns) {
        double elapsed = static_cast<double>(steady_ms_since(phase_start));
        if (elapsed >= wave.low_ms)
          break;
        double remain = wave.low_ms - elapsed;
        unsigned chunk = static_cast<unsigned>(
            std::min(remain, static_cast<double>(smpl_ms)));
        if (chunk < 1u) chunk = 1u;
        std::this_thread::sleep_for(std::chrono::milliseconds(chunk));
        if (std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - last_sample).count()
            >= static_cast<int64_t>(smpl_ms)) {
          float power_low = read_power();
          if (power_low > 0.0f) {
            total_low += power_low;
            low_samples++;
            if (power_low < min_low) min_low = power_low;
            if (power_low < pulse_trough) pulse_trough = power_low;
          }
          float temp = read_temperature();
          if (temp > peak_temp) peak_temp = temp;
          check_temperature(temp, pattern_pass);
          if (!pattern_pass && halt_on_error && judge_results()) {
            abort_flag().store(true, std::memory_order_release);
            stop_cycle = true;
            stop_patterns = true;
          }
          last_sample = std::chrono::steady_clock::now();
        }
      }
      double actual_low_ms = static_cast<double>(steady_ms_since(phase_start));
      cycles++;
      total_cycles++;

      if (PulseWorker::bjson) {
        log_to_json(desc, rvs::logresults,
            "record_type", "pulse",
            "pattern", pname,
            "pulse_num", std::to_string(cycles),
            "power_high_w", std::to_string(pulse_peak),
            "power_low_w", std::to_string(
                pulse_trough < 999999.0f ? pulse_trough : 0.0f),
            "high_duration_ms", std::to_string(actual_high_ms),
            "low_duration_ms", std::to_string(actual_low_ms),
            "temp_c", std::to_string(pulse_temp),
            "gemm_count", std::to_string(gemm_count));
      }

      auto now_sys = std::chrono::system_clock::now();
      if (time_diff(now_sys, last_log) >= log_interval) {
        float avg_h = high_samples > 0 ? total_high / high_samples : 0.0f;
        float avg_l = low_samples > 0 ? total_low / low_samples : 0.0f;
        log_msg("[" + action_name + "] [GPU:: " + gpuid_buff + "] " +
            pname + " pulse #" + std::to_string(cycles) +
            " avg_high=" + std::to_string(avg_h) + "W" +
            " avg_low=" + std::to_string(avg_l) + "W" +
            " delta=" + std::to_string(avg_h - avg_l) + "W", rvs::logresults);
        last_log = now_sys;
      }

      if (stop_cycle || stop_patterns)
        break;
    }

    if (blas_ok && !device_busy)
      run_gemm_verify(pattern_pass);

    if (ecc_supported) {
      uint64_t ecc_now = ecc_base;
      if (read_uncorrectable_ecc(ecc_now) && ecc_now > ecc_base) {
        log_msg("[" + action_name + "] " + MODULE_NAME + " " +
            std::to_string(gpu_id) + " uncorrectable ECC count rose from " +
            std::to_string(ecc_base) + " to " + std::to_string(ecc_now),
            rvs::logerror);
        note_failure(pattern_pass);
      }
    }

    if (judge_results() && cycles > 0 && high_samples == 0 && low_samples == 0) {
      log_msg("[" + action_name + "] " + MODULE_NAME + " " +
          std::to_string(gpu_id) + " no power samples in pattern " + pname,
          rvs::logerror);
      pattern_pass = false;
    }

    float avg_high = high_samples > 0 ? total_high / high_samples : 0.0f;
    float avg_low = low_samples > 0 ? total_low / low_samples : 0.0f;
    float delta = avg_high - avg_low;
    if (judge_results() && min_power_delta_w > 0.0f && cycles > 0 &&
        delta < min_power_delta_w) {
      log_msg("[" + action_name + "] " + MODULE_NAME + " " +
          std::to_string(gpu_id) + " power swing " + std::to_string(delta) +
          "W is below min_power_delta_w " +
          std::to_string(min_power_delta_w) + "W", rvs::logerror);
      pattern_pass = false;
    }

    if (!pattern_pass)
      test_passed = test_passed && !judge_results();

    if (PulseWorker::bjson) {
      log_to_json(desc, rvs::logresults,
          "record_type", "pattern",
          "pattern", pname,
          "pattern_index", std::to_string(pid),
          "pulse_count", std::to_string(cycles),
          "avg_power_high_w", std::to_string(avg_high),
          "avg_power_low_w", std::to_string(avg_low),
          "avg_delta_w", std::to_string(delta),
          "peak_temp_c", std::to_string(peak_temp),
          PULSE_PASS_KEY, (pattern_pass || !judge_results()) ? "true" : "false");
    }
    log_msg("[" + action_name + "] [GPU:: " + gpuid_buff + "] pattern " +
        pname + " cycles=" + std::to_string(cycles) +
        " avg_high=" + std::to_string(avg_high) + "W" +
        " avg_low=" + std::to_string(avg_low) + "W" +
        " delta=" + std::to_string(delta) + "W", rvs::logresults);

    if (!pattern_pass && judge_results()) {
      abort_flag().store(true, std::memory_order_release);
      host_rendezvous();
      stop_patterns = true;
    }
  }

  restore_clocks();
  if (device_busy && gpu_blas) {
    // A timed-out GEMM may still be running. Freeing device memory would
    // block in the driver and pin this thread in join.
    rvs_blas* abandoned = gpu_blas.release();
    (void)abandoned;
  }

  if (!judge_results()) {
    log_msg("[" + action_name + "] " + MODULE_NAME + " " +
        std::to_string(gpu_id) +
        " secondary MCM, considering pass by default", rvs::loginfo);
    result = true;
  } else {
    result = test_passed && total_cycles > 0 && !device_busy;
  }

  if (PulseWorker::bjson) {
    log_to_json(desc, rvs::logresults,
        "record_type", "summary",
        "patterns", patterns_ran,
        "pulse_count", std::to_string(total_cycles),
        "pattern_duration_ms", std::to_string(pattern_duration_ms),
        "ops_type", pulse_ops_type,
        "matrix_m", std::to_string(matrix_m),
        "matrix_n", std::to_string(matrix_n),
        "matrix_k", std::to_string(matrix_k),
        PULSE_PASS_KEY, result ? "true" : "false");
  }

  msg = "[" + action_name + "] [GPU:: " + gpuid_buff + "] completed " +
    std::to_string(total_cycles) + " pulses patterns=" + patterns_ran;
  log_msg(msg, rvs::logresults);

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
    std::to_string(gpu_id) + " start";
  rvs::lp::Log(msg, rvs::loginfo);

  bool pass = do_pulse_stress();

  if (rvs::lp::Stopping())
    return;

  snprintf(gpuid_buff, sizeof(gpuid_buff), "%5d", gpu_id);
  msg = "[" + action_name + "] [GPU:: " + gpuid_buff + "] " +
    PULSE_PASS_KEY + ": " +
    (pass ? PULSE_RESULT_PASS : PULSE_RESULT_FAIL);
  rvs::lp::Log(msg, rvs::logresults);
}
