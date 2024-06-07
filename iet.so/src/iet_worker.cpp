/********************************************************************************
 *
 * Copyright (c) 2018-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <iostream>
#include <chrono>
#include <memory>
#include <exception>

#include "hip/hip_ext.h"
#include "rocm_smi/rocm_smi.h"
#include "include/rvs_module.h"
#include "include/rvsloglp.h"

#include "include/iet_worker.h"

#define MODULE_NAME                             "iet"
#define POWER_PROCESS_DELAY                     5
#define MAX_MS_TRAIN_GPU                        1000
#define MAX_MS_WAIT_BLAS_THREAD                 10000
#define SGEMM_DELAY_FREQ_DEV                    10

#define IET_RESULT_PASS_MESSAGE                 "TRUE"
#define IET_RESULT_FAIL_MESSAGE                 "FALSE"

#define IET_BLAS_FAILURE                        "BLAS setup failed!"
#define IET_POWER_PROC_ERROR                    "could not get/process the GPU"\
                                                " power!"
#define IET_SGEMM_FAILURE                       "GPU failed to run the SGEMMs!"

#define IET_TARGET_MESSAGE                      "target"
#define IET_DTYPE_MESSAGE                       "dtype"
#define IET_PWR_VIOLATION_MSG                   "power violation"
#define IET_PWR_TARGET_ACHIEVED_MSG             "target achieved"
#define IET_PWR_RAMP_EXCEEDED_MSG               "ramp time exceeded"
#define IET_PASS_KEY                            "pass"

#define IET_JSON_LOG_GPU_ID_KEY                 "gpu_id"
#define IET_MEM_ALLOC_ERROR                     1
#define IET_BLAS_ERROR                          2
#define IET_BLAS_MEMCPY_ERROR                   3
#define IET_BLAS_ITERATIONS                     25
#define IET_LOG_GFLOPS_INTERVAL_KEY             "GFLOPS"
#define IET_AVERAGE_POWER_KEY                   "average power"
using std::string;

bool IETWorker::bjson = false;


/**
 * @brief computes the difference (in milliseconds) between 2 points in time
 * @param t_end second point in time
 * @param t_start first point in time
 * @return time difference in milliseconds
 */
static uint64_t time_diff(
                std::chrono::time_point<std::chrono::system_clock> t_end,
                std::chrono::time_point<std::chrono::system_clock> t_start) {
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
                            t_end - t_start);
    return milliseconds.count();
}

/**
 * @brief logs a message to JSON
 * @param key info type
 * @param value message to log
 * @param log_level the level of log (e.g.: info, results, error)
 */
void IETWorker::log_to_json(const std::string &key, const std::string &value,
                     int log_level) {
	if(!IETWorker::bjson)
		return;
        void *json_node = json_node_create(std::string(MODULE_NAME),
                            action_name.c_str(), log_level);
        if (json_node) {
            rvs::lp::AddString(json_node, IET_JSON_LOG_GPU_ID_KEY,
                            std::to_string(gpu_id));
            rvs::lp::AddString(json_node, key, value);
            rvs::lp::LogRecordFlush(json_node);
        }
}


/**
 * @brief class default constructor
 */
IETWorker::IETWorker():endtest(false) {
}

IETWorker::~IETWorker() {
}



/**
 * @brief logs the Gflops computed over the last log_interval period
 * @param gflops_interval the Gflops that the GPU achieved
 */
void IETWorker::log_interval_gflops(double gflops_interval) {
    string msg;
    msg = " GPU flops :" + std::to_string(gflops_interval);
    rvs::lp::Log(msg, rvs::logtrace);
    log_to_json(IET_LOG_GFLOPS_INTERVAL_KEY, std::to_string(gflops_interval),
                rvs::loginfo);

}

void IETWorker::blasThread(int gpuIdx,  uint64_t matrix_size, std::string  iet_ops_type, 
    bool start, uint64_t run_duration_ms, int transa, int transb, float alpha, float beta,
    int iet_lda_offset, int iet_ldb_offset, int iet_ldc_offset, int iet_ldd_offset) {

    std::chrono::time_point<std::chrono::system_clock> iet_start_time, iet_end_time;
    double timetakenforoneiteration;
    double gflops_interval;
    double duration;
    uint64_t gem_ops;
    std::unique_ptr<rvs_blas> gpu_blas;
    rvs_blas *free_gpublas;
    string msg;

    duration = 0;
    gem_ops = 0;
    // setup rvsBlas
    gpu_blas = std::unique_ptr<rvs_blas>(new rvs_blas(gpuIdx,  matrix_size,  matrix_size,  matrix_size, "default", transa, transb, alpha, beta,
          iet_lda_offset, iet_ldb_offset, iet_ldc_offset, iet_ldd_offset, iet_ops_type, ""));

    //Genreate random matrix data
    gpu_blas->generate_random_matrix_data();

    //Copy data to GPU
    gpu_blas->copy_data_to_gpu(iet_ops_type);

    iet_start_time = std::chrono::system_clock::now();

    //Hit the GPU with load to increase temperature
    while ((duration < run_duration_ms) && (endtest == false)) {

        //call the gemm blas
        gpu_blas->run_blass_gemm(iet_ops_type);

        // Waits for GEMM operation to complete
        if(!gpu_blas->is_gemm_op_complete())
          continue;

        //get the end time
        iet_end_time = std::chrono::system_clock::now();
        //Duration in the call
        duration = time_diff(iet_end_time, iet_start_time);

        //Converting microseconds to seconds
        timetakenforoneiteration = duration/1e6;
        //calculating Gemm count
        gflops_interval = gpu_blas->gemm_gflop_count()/timetakenforoneiteration;
        //Print the gflops interval
        log_interval_gflops(gflops_interval);

        // check end test to avoid unnecessary sleep
        if (endtest)
            break;
    }
}

/**
 * @brief performs the Input EDPp stress (IET) test on the given GPU (attempts to sustain
 * the target power)
 * @return true if EDPp test succeeded, false otherwise
 */
bool IETWorker::do_iet_power_stress(void) {

    std::chrono::time_point<std::chrono::system_clock> iet_start_time, end_time,
        sampling_start_time;
    uint64_t  total_time_ms;
    uint64_t  last_power;
    string    msg;
    float     cur_power_value = 0;
    float     totalpower = 0;
    float     max_power = 0;
    bool      result = true;
    bool      start = true;
    rvs::action_result_t action_result;
    RSMI_POWER_TYPE type = RSMI_INVALID_POWER;
    char gpuid_buff[12];
    std::thread compute_t;
    std::thread bandwidth_t;

    // Start default compute thread
    compute_t = std::thread(&IETWorker::blasThread, this, gpu_device_index, matrix_size, iet_ops_type, start, run_duration_ms,
            iet_trans_a, iet_trans_b, iet_alpha_val, iet_beta_val, iet_lda_offset, iet_ldb_offset, iet_ldc_offset, iet_ldd_offset);

    // Start bandwidth thread if bandwidth workload is enabled
    if (iet_bw_workload) {
      bandwidth_t = std::thread(&IETWorker::bandwidthThread, this, gpu_device_index, run_duration_ms);
    }

    snprintf(gpuid_buff, sizeof(gpuid_buff), "%5d", gpu_id);

    // record EDPp ramp-up start time
    iet_start_time = std::chrono::system_clock::now();

    for (;;) {
        // check if stop signal was received
        if (rvs::lp::Stopping())
            break;

        cur_power_value = 0;

        // get GPU's current/average power
        rsmi_status_t rmsi_stat = rsmi_dev_power_get(smi_device_index, &last_power, &type);
        if (rmsi_stat == RSMI_STATUS_SUCCESS) {
          cur_power_value = static_cast<float>(last_power)/1e6;
        }

        msg = "[" + action_name + "] " + MODULE_NAME + " " +
            std::to_string(gpu_id) + " " + " Target power is : " + " " + std::to_string(target_power);
        rvs::lp::Log(msg, rvs::logtrace);

        //update power to max if it is valid
        if(cur_power_value > 0) {
            max_power = std::max(max_power, cur_power_value);// max of averages
        }

        end_time = std::chrono::system_clock::now();

        total_time_ms = time_diff(end_time, iet_start_time);

        msg = "[" + action_name + "] " + "[GPU:: " + gpuid_buff + "] " +
          "Power(W) " + std::to_string(cur_power_value);
        rvs::lp::Log(msg, rvs::logresults);

        msg = "[" + action_name + "] " + MODULE_NAME + " " +
            std::to_string(gpu_id) + " " + " Total time in ms " + " " + std::to_string(total_time_ms) +
            " Run duration in ms " + " " + std::to_string(run_duration_ms);
        rvs::lp::Log(msg, rvs::logtrace);

        if (total_time_ms > run_duration_ms) {
            break;
        }

        //It doesnt make sense to read power continously so slowing down
        sleep(sample_interval);

        // check if stop signal was received
        if (rvs::lp::Stopping()) {
            result = true;
            goto end;
        }
    }

    // json log the avg power
    log_to_json(IET_AVERAGE_POWER_KEY, std::to_string(max_power),
            rvs::loginfo);
    //check whether we reached the target power
    if(max_power >= target_power) {
        msg = "[" + action_name + "] " + MODULE_NAME + " " +
            std::to_string(gpu_id) + " " + " Average power met the target power :" + " " + std::to_string(max_power);
        rvs::lp::Log(msg, rvs::loginfo);
        result = true;
    }else {
        msg = "[" + action_name + "] " + MODULE_NAME + " " +
            std::to_string(gpu_id) + " " + " Average power could not meet the target power  \
            in the given interval, increase the duration and try again, \
            Average power is :" + " " + std::to_string(max_power);
        rvs::lp::Log(msg, rvs::loginfo);
        result = false;
    }

    action_result.state = rvs::actionstate::ACTION_RUNNING;
    action_result.status = (true == result) ? rvs::actionstatus::ACTION_SUCCESS : rvs::actionstatus::ACTION_FAILED;
    action_result.output = msg.c_str();
    action.action_callback(&action_result);

    msg = "[" + action_name + "] " + MODULE_NAME + " " +
        std::to_string(gpu_id) + " " + " End of worker thread " ;
    rvs::lp::Log(msg, rvs::loginfo);

end:

    endtest = true;

    if (true == compute_t.joinable()) {

        try {
            compute_t.join();
        }
        catch (std::exception& e) {
            std::cout << "Standard exception: " << e.what() << std::endl;
        }
    }
    if (true == bandwidth_t.joinable()) {

        try {
            bandwidth_t.join();
        }
        catch (std::exception& e) {
            std::cout << "Standard exception: " << e.what() << std::endl;
        }
    }
    return result;
}


/**
 * @brief performs the Input EDPp (IET) test on the given GPU
 */
void IETWorker::run() {
    string msg, err_description;
    char gpuid_buff[12];

    msg = "[" + action_name + "] " + MODULE_NAME + " " +
            std::to_string(gpu_id) + " start " + std::to_string(target_power);

    rvs::lp::Log(msg, rvs::loginfo);

    if (run_duration_ms < MAX_MS_TRAIN_GPU)
        run_duration_ms += MAX_MS_TRAIN_GPU;

    bool pass = do_iet_power_stress();

    // check if stop signal was received
    if (rvs::lp::Stopping())
         return;

    snprintf(gpuid_buff, sizeof(gpuid_buff), "%5d", gpu_id);

    msg = "[" + action_name + "] " + "[GPU:: " + gpuid_buff + "] " +
      IET_PASS_KEY + ": " + (pass ? IET_RESULT_PASS_MESSAGE : IET_RESULT_FAIL_MESSAGE);
    rvs::lp::Log(msg, rvs::logresults);

    sleep(5);
}

/**
 * @brief blas callback function upon gemm operation completion
 * @param status gemm operation status
 * @param user_data user data set
 */
void IETWorker::blas_callback (bool status, void *user_data) {

  if(!user_data) {
    return;
  }
  IETWorker* worker = (IETWorker*)user_data;

  /* Notify gst worker thread gemm operation completion */
  std::lock_guard<std::mutex> lk(worker->mutex);
  worker->blas_status = status;
  worker->cv.notify_one();
}

// Bandwidth workload configurations

constexpr uint64_t mall_size = 256 * 1024 * 1024;

constexpr uint64_t mem_size_per_wg = 32 * 1024 * 1024;
const uint64_t wg_mem_pad = 4352;
constexpr uint64_t wg_stride = mem_size_per_wg + wg_mem_pad;
const uint32_t vmem_unroll = 2;
const uint32_t wg_size = 1024;

#define T __uint128_t

const uint32_t wg_count = 80;
const bool fill_zero = false;
const bool nt_loads = false;
const bool report_metric = false;

template<bool NT> __device__ T _load(T* __restrict p)
{
  T v;
  if (NT)
  {
    v = __builtin_nontemporal_load(p);
  }
  else
  {
    v = *p;
  }
  return v;
}

template<bool NT>
__global__ void bw_kernel(T* __restrict p, T* __restrict r, uint32_t iters)
{
  T v{};
  T* wg_p = (T*)((char*)p + blockIdx.x * wg_stride);
  volatile __shared__ uint32_t m[16384];
  uint32_t t = threadIdx.x * 4;
  uint32_t d;

  for (uint32_t i = 0; i < iters; i++)
  {
    uint32_t offs = i * vmem_unroll * wg_size + threadIdx.x;
#pragma unroll
    for (uint32_t j = 0; j < vmem_unroll; j++)
    {
      v |= _load<NT>(&wg_p[offs]);
      offs += wg_size;
    }
  }

  if (v == 10000000)
  {
    *r = v + m[100];
  }
}

bool FillMemory(uint32_t* p, size_t size)
{
  const size_t fitems = 65 * 1024;
  const size_t fsize = fitems * sizeof(uint32_t);

  uint32_t* f{};
  if(hipSuccess != hipHostMalloc(&f, fsize))
    return false;

  for (size_t i = 0; i < fitems; i++)
  {
    f[i] = fill_zero ? 0 : rand();
  }

  while (size > 0)
  {
    if(hipSuccess != hipMemcpy(p, f, std::min(fsize, size), hipMemcpyDefault))
      return false;

    p += fitems;
    size -= std::min(fsize, size);
  }

  hipHostFree(f);
  return true;
}

void RunKernel(hipEvent_t& start, hipEvent_t& stop, T* __restrict p, T* __restrict r,
    uint32_t fetch_iters, hipStream_t stream)
{
  const dim3 grid_dim(wg_count, 1, 1);
  const dim3 block_dim(wg_size, 1, 1);
  uint32_t shared_mem = 0;

  hipExtLaunchKernelGGL(nt_loads ? bw_kernel<true> : bw_kernel<false>,
      grid_dim, block_dim, shared_mem, stream, start, stop, 0,
      p, r, fetch_iters);
}

void IETWorker::bandwidthThread(int gpuIdx, uint64_t run_duration_ms)
{
  hipStream_t stream = 0;
  std::vector<uint32_t*> bufs;

  uint32_t* r{};
  hipEvent_t start;
  hipEvent_t stop;

  int b = 0;
  int test_iters = 0;
  double min_bw = __FLT_MAX__;
  double max_bw = 0.0f;
  double avg_bw = 0.0f;

  double duration = 0.0f;
  std::chrono::time_point<std::chrono::system_clock> start_time, end_time;

  srand(time(NULL));

  // Select GPU device for bandwidth workload
  if (hipSetDevice(gpuIdx) != hipSuccess) {
    // cannot select the given GPU device
    return;
  }

  // Create device specific stream for bandwidth workload
  if (hipStreamCreate(&stream) != hipSuccess) {
    std::cout << "\n hipStreamCreate() failed !!!" << "\n";
    return;
  }

  // Prepare device memory for bandwidth workload

  uint32_t wg_fetch_size = wg_size * sizeof(T) * vmem_unroll;
  uint32_t fetch_iters = mem_size_per_wg / wg_fetch_size;

  // ensure fetch is happening in full blocks per WG
  if(mem_size_per_wg % wg_fetch_size != 0)
    return;

  uint64_t alloc_size = (uint64_t)wg_count * wg_stride;
  uint64_t data_size = (uint64_t)wg_count * mem_size_per_wg;

  int buf_count = std::max((int)2, (int)(mall_size * 2 / data_size + 1));
  for (int i = 0; i < buf_count; i++)
  {
    uint32_t* p{};

    if (hipMalloc(&p, alloc_size) != hipSuccess)
       goto end;

    if (true != FillMemory(p, alloc_size)) {
      hipFree(p);
      goto end;
    }

    bufs.push_back(p);
  }

  if (hipMalloc(&r, 4096) != hipSuccess)
    goto end;

  // Execute bandwidth workload

  if (hipEventCreate(&start) != hipSuccess) {
    goto end;
  }
  if (hipEventCreate(&stop) != hipSuccess) {
    goto end;
  }

  for (int i = 0; i < buf_count; i++)
  {
    RunKernel(start, stop, (T*)bufs[i], (T*)r, fetch_iters, stream);
  }

  start_time = std::chrono::system_clock::now();

  do
  {
    RunKernel(start, stop, (T*)bufs[b], (T*)r, fetch_iters, stream);
    b = (b + 1) % buf_count;

    if(hipSuccess != hipEventSynchronize(stop))
      goto end;

    float t_ms;
    if(hipSuccess != hipEventElapsedTime(&t_ms, start, stop))
      goto end;

    double bw = (double)data_size / (t_ms  / 1000.0) / 1000000000;
    if (!report_metric)
    {
      bw /= 1.024 * 1.024 * 1.024;
    }

    min_bw = std::min(min_bw, bw);
    max_bw = std::max(max_bw, bw);
    avg_bw += bw;

    test_iters++;

    if (endtest)
      break;

    // Get the current time
    end_time = std::chrono::system_clock::now();

    // Duration elasped since start
    duration = time_diff(end_time, start_time);

  } while ((duration < run_duration_ms) && (endtest == false));

  if(hipStreamSynchronize(stream) != hipSuccess)
    std::cout << "hipStreamSynchronize() failed !!! for stream " << stream << std::endl;

  avg_bw /= test_iters;

end:

  // Clean up bandwith workload resources

  if(hipEventDestroy(start) != hipSuccess)
    std::cout << "hipEventDestroy() failed !!! for event " << start << std::endl;

  if(hipEventDestroy(stop) != hipSuccess)
    std::cout << "hipEventDestroy() failed !!! for event " << stop << std::endl;

  if(r)
    hipFree(r);

  for (int i = 0; i < bufs.size(); i++)
  {
    hipFree(bufs[i]);
  }

  if(hipStreamDestroy(stream) != hipSuccess)
    std::cout << "hipStreamDestroy() failed !!! for stream " << stream << std::endl;
}

