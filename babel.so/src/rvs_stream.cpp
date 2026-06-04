// Copyright (c) 2014-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code


#include "include/HIPStream.h"
#include "hip/hip_runtime.h"
#include "include/rvsloglp.h"
#include "hiprand/hiprand.hpp"

#include <cfloat>
#include <random>
#include <thread>

#ifndef TBSIZE
#define TBSIZE 1024
#endif

#ifdef __HCC__
__device__ uint32_t grid_size() {
  return hc_get_grid_size(0);
}
__device__ uint32_t localid() {
  return hc_get_workitem_absolute_id(0);
}
#elif defined(__HIP__)
extern "C" __device__ size_t __ockl_get_global_size(uint);
extern "C" __device__ size_t __ockl_get_global_id(uint);
__device__ uint32_t grid_size() {
  return __ockl_get_global_size(0);
}
__device__ uint32_t localid() {
  return __ockl_get_global_id(0);
}
#else
__device__ uint32_t grid_size() {
  return blockDim.x * gridDim.x;
}
__device__ uint32_t localid() {
  return threadIdx.x + blockIdx.x * blockDim.x;
}
#endif


template<typename T>
__device__ __forceinline__ constexpr T scalar(const T scalar) {
  if constexpr (sizeof(T) == sizeof(float)) {
    return static_cast<float>(scalar);
  } else {
    return static_cast<double>(scalar);
  }
}

#define check_error(status)                                                    \
  do {                                                                         \
    hipError_t err = status;                                                   \
    if (err != hipSuccess) {                                                   \
      std::cerr << "Error: " << hipGetErrorString(err) << std::endl;           \
      exit(err);                                                               \
    }                                                                          \
  } while(0)

// Non-temporal load/store control
//   NT_ALL  (1) : NT load + NT store (default)
//   NT_READ (2) : NT load + normal store
//   NT_WRITE(3) : normal load + NT store
//   NT_NONE (0) : normal load + normal store
enum NTMode { NT_NONE = 0, NT_ALL = 1, NT_READ = 2, NT_WRITE = 3 };

#define NT_KERNEL_LAUNCH_EVENTS(kernel, epl, cpb, T, grid, block, smem, start, stop, ...) \
  switch (nt_mode) {                                                                \
    case NT_NONE:  hipLaunchKernelWithEvents((kernel<NT_NONE, epl, cpb, T>),        \
        grid, block, smem, start, stop, __VA_ARGS__); break;                        \
    case NT_READ:  hipLaunchKernelWithEvents((kernel<NT_READ, epl, cpb, T>),        \
        grid, block, smem, start, stop, __VA_ARGS__); break;                        \
    case NT_WRITE: hipLaunchKernelWithEvents((kernel<NT_WRITE, epl, cpb, T>),       \
        grid, block, smem, start, stop, __VA_ARGS__); break;                        \
    default:       hipLaunchKernelWithEvents((kernel<NT_ALL, epl, cpb, T>),         \
        grid, block, smem, start, stop, __VA_ARGS__); break;                        \
  }

#define NT_KERNEL_LAUNCH_SYNC(kernel, epl, cpb, T, grid, block, smem, stop, ...) \
  switch (nt_mode) {                                                 \
    case NT_NONE:  hipLaunchKernelSynchronous((kernel<NT_NONE, epl, cpb, T>),  \
        grid, block, smem, stop, __VA_ARGS__); break;                          \
    case NT_READ:  hipLaunchKernelSynchronous((kernel<NT_READ, epl, cpb, T>),  \
        grid, block, smem, stop, __VA_ARGS__); break;                          \
    case NT_WRITE: hipLaunchKernelSynchronous((kernel<NT_WRITE, epl, cpb, T>), \
        grid, block, smem, stop, __VA_ARGS__); break;                          \
    default:       hipLaunchKernelSynchronous((kernel<NT_ALL, epl, cpb, T>),   \
        grid, block, smem, stop, __VA_ARGS__); break;                          \
  }

#define NT_KERNEL_LAUNCH_DOT_SYNC(kernel, epl, cpb, T, tbs, grid, block, smem, stop, ...) \
  switch (nt_mode) {                                                           \
    case NT_NONE:  hipLaunchKernelSynchronous((kernel<NT_NONE, epl, cpb, T, tbs>),  \
        grid, block, smem, stop, __VA_ARGS__); break;                                \
    case NT_READ:  hipLaunchKernelSynchronous((kernel<NT_READ, epl, cpb, T, tbs>),  \
        grid, block, smem, stop, __VA_ARGS__); break;                                \
    case NT_WRITE: hipLaunchKernelSynchronous((kernel<NT_WRITE, epl, cpb, T, tbs>), \
        grid, block, smem, stop, __VA_ARGS__); break;                                \
    default:       hipLaunchKernelSynchronous((kernel<NT_ALL, epl, cpb, T, tbs>),   \
        grid, block, smem, stop, __VA_ARGS__); break;                                \
  }

template <typename... Args, typename F = void (*)(Args...)>
static void hipLaunchKernelWithEvents(F kernel, const dim3& numBlocks,
                           const dim3& dimBlocks, hipStream_t stream,
                           hipEvent_t startEvent, hipEvent_t stopEvent,
                           Args... args)
{
  check_error(hipEventRecord(startEvent));
  hipLaunchKernelGGL(kernel, numBlocks, dimBlocks,
                   0, stream, args...);
  check_error(hipGetLastError());
  check_error(hipEventRecord(stopEvent));
}

template <typename... Args, typename F = void (*)(Args...)>
static void hipLaunchKernelSynchronous(F kernel, const dim3& numBlocks,
                           const dim3& dimBlocks, hipStream_t stream,
                           hipEvent_t event, Args... args)
{
#ifdef __HIP_PLATFORM_NVCC__
  hipLaunchKernelGGL(kernel, numBlocks, dimBlocks,
                   0, stream, args...);
  check_error(hipGetLastError());
  check_error(hipDeviceSynchronize());
#else
  hipLaunchKernelGGL(kernel, numBlocks, dimBlocks,
                     0, stream, args...);
  check_error(hipGetLastError());
  check_error(hipEventRecord(event));
  check_error(hipEventSynchronize(event));
#endif
}

  template <class T>
HIPStream<T>::HIPStream(const unsigned int ARRAY_SIZE, const bool event_timing,
    const int device_index, const unsigned int _dwords_per_lane, const unsigned int _chunks_per_block,
    const unsigned int _threads_per_block, const std::string& nontemporal)
  : array_size{ARRAY_SIZE}, evt_timing(event_timing),
  dwords_per_lane(_dwords_per_lane), chunks_per_block(_chunks_per_block), tb_size(_threads_per_block)
{

  std::string msg;

  // Set Non-temporal mode
  if (nontemporal == "none")       nt_mode = 0; // NT_NONE
  else if (nontemporal == "read")  nt_mode = 2; // NT_READ
  else if (nontemporal == "write") nt_mode = 3; // NT_WRITE
  else                             nt_mode = 1; // NT_ALL

  // make sure that either:
  //    DWORDS_PER_LANE is less than sizeof(T), in which case we default to 1 element
  //    or
  //    DWORDS_PER_LANE is divisible by sizeof(T)

  if(!((dwords_per_lane * sizeof(unsigned int) < sizeof(T) ||
        (dwords_per_lane * sizeof(unsigned int) % sizeof(T) == 0)))) {

    std::stringstream ss;
    ss << "dwords_per_lane not divisible by sizeof(element_type)";
    throw std::runtime_error(ss.str());
  }

  // take into account the datatype size
  // that is, if we specify 4 DWORDS_PER_LANE, this is 2 FP64 elements
  // and 4 FP32 elements
  elements_per_lane =
    (dwords_per_lane * sizeof(unsigned int)) < sizeof(T) ? 1 :
    (dwords_per_lane * sizeof(unsigned int) / sizeof(T));

  block_cnt = (array_size / (tb_size * elements_per_lane * chunks_per_block));

  msg = std::string("\nelements per lane ") + std::to_string(elements_per_lane) + "," +
	 std::string("chunks per block ") + std::to_string(chunks_per_block);

  // The array size must be divisible by total number of elements
  // moved per block for kernel launches
  if (ARRAY_SIZE % (tb_size * elements_per_lane * chunks_per_block) != 0)
  {
    std::stringstream ss;
    ss << "Array size must be a multiple of elements operated on per block (" <<
          tb_size * elements_per_lane * chunks_per_block << ").";
    throw std::runtime_error(ss.str());
  }
  msg += ", block count "  + std::to_string(block_cnt);

  // Set device
  int count;
  check_error(hipGetDeviceCount(&count));
  if (device_index >= count)
    throw std::runtime_error("Invalid device index");
  check_error(hipSetDevice(device_index));
  msg += "\nUsing HIP device " + getDeviceName(device_index) + ", " +
	  "Driver: "  + getDeviceDriver(device_index) ;
  
  // Allocate the host array for partial sums for dot kernels
  check_error(hipHostMalloc(&sums, sizeof(T) * block_cnt, hipHostMallocNonCoherent));

  // Check buffers fit on the device
  hipDeviceProp_t props;
  check_error(hipGetDeviceProperties(&props, 0));
  if (props.totalGlobalMem < 3*ARRAY_SIZE*sizeof(T))
    throw std::runtime_error("Device does not have enough memory for all 3 buffers");
  msg += ", pciBusID: " + std::to_string(props.pciBusID);
  rvs::lp::Log(msg, rvs::loginfo);
  // Create device buffers
  check_error(hipMalloc(&d_a, ARRAY_SIZE * sizeof(T)));
  check_error(hipMalloc(&d_b, ARRAY_SIZE * sizeof(T)));
  check_error(hipMalloc(&d_c, ARRAY_SIZE * sizeof(T)));

  check_error(hipEventCreate(&start_ev));
  check_error(hipEventCreate(&stop_ev));
  check_error(hipEventCreateWithFlags(&coherent_ev, hipEventReleaseToSystem));
}


template <class T>
HIPStream<T>::~HIPStream()
{
  check_error(hipHostFree(sums));
  check_error(hipFree(d_a));
  check_error(hipFree(d_b));
  check_error(hipFree(d_c));
  check_error(hipEventDestroy(start_ev));
  check_error(hipEventDestroy(stop_ev));
  check_error(hipEventDestroy(coherent_ev));
}


template <typename T>
__global__ void init_kernel(T * a, T * b, T * c, T initA, T initB, T initC)
{
  const int i = localid();
  a[i] = initA;
  b[i] = initB;
  c[i] = initC;
}

template <class T>
void HIPStream<T>::init_arrays(T initA, T initB, T initC)
{
  hipLaunchKernelGGL(init_kernel<T>, dim3(array_size/tb_size), dim3(tb_size), 0,
                     nullptr, d_a, d_b, d_c, initA, initB, initC);
  check_error(hipGetLastError());
  check_error(hipDeviceSynchronize());
}

template <class T>
void HIPStream<T>::init_arrays_normdist(
    T mean, T stddev, bool gpu_init,
    std::vector<T>& a, std::vector<T>& b, std::vector<T>& c)
{
  if (!gpu_init) {

#if !defined(USE_CPU_THREADS_INIT)
    rvs::lp::Log("Using a Single Thread on CPU to Initialize NORMAL distributed data",
                  rvs::loginfo);

    std::random_device rd{};
    std::mt19937_64 gen{rd()};
    std::normal_distribution<T> dist{mean, stddev};

    auto gen_func = [&]() { return dist(gen); };

    std::generate(a.begin(), a.end(), gen_func);
    std::generate(b.begin(), b.end(), gen_func);
    std::generate(c.begin(), c.end(), gen_func);

#else
    constexpr uint32_t NUM_CHUNKS = NUM_CPU_THREADS_INIT;
    rvs::lp::Log(std::string("Using ") + std::to_string(NUM_CHUNKS) +
                  " Threads on CPU to Initialize NORMAL distributed data",
                  rvs::loginfo);

    std::vector<std::thread> workers;
    workers.reserve(NUM_CHUNKS);

    const uint32_t CHUNK_SIZE = array_size / NUM_CHUNKS;

    for (uint32_t work_id = 0; work_id < NUM_CHUNKS; ++work_id) {
      const uint32_t start = work_id * CHUNK_SIZE;
      const uint32_t end =
          (work_id == NUM_CHUNKS - 1) ? array_size : start + CHUNK_SIZE;

      workers.emplace_back([&, start, end]() {
        std::random_device rd{};
        std::mt19937_64 gen{rd()};
        std::normal_distribution<T> dist{mean, stddev};
        auto gen_func = [&]() { return dist(gen); };

        std::generate(a.begin() + start, a.begin() + end, gen_func);
        std::generate(b.begin() + start, b.begin() + end, gen_func);
        std::generate(c.begin() + start, c.begin() + end, gen_func);
      });
    }

    for (auto& w : workers) {
      w.join();
    }
#endif

    check_error(hipMemcpy(d_a, a.data(), sizeof(T) * array_size, hipMemcpyHostToDevice));
    check_error(hipMemcpy(d_b, b.data(), sizeof(T) * array_size, hipMemcpyHostToDevice));
    check_error(hipMemcpy(d_c, c.data(), sizeof(T) * array_size, hipMemcpyHostToDevice));

  } else {

    rvs::lp::Log("WARNING: Using GPU based NORMAL Distribution Initialization\n"
                  "CPU based NORMAL Distribution Initialization is RECOMMENDED when validating",
                  rvs::loginfo);

    hiprand_cpp::mt19937_engine<HIPRAND_MT19937_DEFAULT_SEED> engine;
    hiprand_cpp::normal_distribution<T> dist{mean, stddev};

    try { dist(engine, d_a, array_size); }
    catch (const std::exception& e) {
      std::string err = std::string("hipRAND ERROR: ") + e.what();
      rvs::lp::Log(err, rvs::logerror);
      std::exit(EXIT_FAILURE);
    }

    try { dist(engine, d_b, array_size); }
    catch (const std::exception& e) {
      std::string err = std::string("hipRAND ERROR: ") + e.what();
      rvs::lp::Log(err, rvs::logerror);
      std::exit(EXIT_FAILURE);
    }

    try { dist(engine, d_c, array_size); }
    catch (const std::exception& e) {
      std::string err = std::string("hipRAND ERROR: ") + e.what();
      rvs::lp::Log(err, rvs::logerror);
      std::exit(EXIT_FAILURE);
    }

    check_error(hipDeviceSynchronize());
  }
}

template <class T>
void HIPStream<T>::read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c)
{
  check_error(hipDeviceSynchronize());
  // Copy device memory to host
  check_error(hipMemcpy(a.data(), d_a, a.size()*sizeof(T), hipMemcpyDeviceToHost));
  check_error(hipMemcpy(b.data(), d_b, b.size()*sizeof(T), hipMemcpyDeviceToHost));
  check_error(hipMemcpy(c.data(), d_c, c.size()*sizeof(T), hipMemcpyDeviceToHost));
}

template<int nt_mode, typename T>
__device__ __forceinline__ T load(const T& ref) {
  if constexpr (nt_mode == NT_ALL || nt_mode == NT_READ)
    return __builtin_nontemporal_load(&ref);
  else
    return ref;
}

template<int nt_mode, typename T>
__device__ __forceinline__ void store(const T& value, T& ref) {
  if constexpr (nt_mode == NT_ALL || nt_mode == NT_WRITE)
    __builtin_nontemporal_store(value, &ref);
  else
    ref = value;
}

template <int nt_mode, unsigned int elements_per_lane, unsigned int chunks_per_block, typename T>
__launch_bounds__(TBSIZE)
__global__
void read_kernel(const T * __restrict a, T * __restrict c)
{
  const auto dx = grid_size() * elements_per_lane;
  const auto gidx = (localid()) * elements_per_lane;

  T tmp{0};
  for (auto i = 0u; i != chunks_per_block; ++i)
  {
    for (auto j = 0u; j != elements_per_lane; ++j)
    {
      tmp += load<nt_mode>(a[gidx + i * dx + j]);
    }
  }

  // Prevent side-effect free loop from being optimised away.
  if (tmp == FLT_MIN)
  {
    c[gidx] = tmp;
  }
}

template <class T>
float HIPStream<T>::read()
{
  float kernel_time = 0.;
  if (evt_timing)
  {
    if(elements_per_lane == 4 && chunks_per_block == 1)
      NT_KERNEL_LAUNCH_EVENTS(read_kernel, 4, 1, T, dim3(block_cnt), dim3(tb_size), nullptr, start_ev, stop_ev, d_a, d_c)
    else if(elements_per_lane == 2 && chunks_per_block == 1)
      NT_KERNEL_LAUNCH_EVENTS(read_kernel, 2, 1, T, dim3(block_cnt), dim3(tb_size), nullptr, start_ev, stop_ev, d_a, d_c)
    else if(elements_per_lane == 4 && chunks_per_block == 2)
      NT_KERNEL_LAUNCH_EVENTS(read_kernel, 4, 2, T, dim3(block_cnt), dim3(tb_size), nullptr, start_ev, stop_ev, d_a, d_c)
    else if(elements_per_lane == 2 && chunks_per_block == 2)
      NT_KERNEL_LAUNCH_EVENTS(read_kernel, 2, 2, T, dim3(block_cnt), dim3(tb_size), nullptr, start_ev, stop_ev, d_a, d_c)
    else if(elements_per_lane == 4 && chunks_per_block == 4)
      NT_KERNEL_LAUNCH_EVENTS(read_kernel, 4, 4, T, dim3(block_cnt), dim3(tb_size), nullptr, start_ev, stop_ev, d_a, d_c)
    else if(elements_per_lane == 2 && chunks_per_block == 4)
      NT_KERNEL_LAUNCH_EVENTS(read_kernel, 2, 4, T, dim3(block_cnt), dim3(tb_size), nullptr, start_ev, stop_ev, d_a, d_c)
    else
      NT_KERNEL_LAUNCH_EVENTS(read_kernel, 4, 2, T, dim3(block_cnt), dim3(tb_size), nullptr, start_ev, stop_ev, d_a, d_c)

    check_error(hipEventSynchronize(stop_ev));
    check_error(hipEventElapsedTime(&kernel_time, start_ev, stop_ev));
  }
  else
  {
    if(elements_per_lane == 4 && chunks_per_block == 1)
      NT_KERNEL_LAUNCH_SYNC(read_kernel, 4, 1, T, dim3(block_cnt), dim3(tb_size), nullptr, stop_ev, d_a, d_c)
    else if(elements_per_lane == 2 && chunks_per_block == 1)
      NT_KERNEL_LAUNCH_SYNC(read_kernel, 2, 1, T, dim3(block_cnt), dim3(tb_size), nullptr, stop_ev, d_a, d_c)
    else if(elements_per_lane == 4 && chunks_per_block == 2)
      NT_KERNEL_LAUNCH_SYNC(read_kernel, 4, 2, T, dim3(block_cnt), dim3(tb_size), nullptr, stop_ev, d_a, d_c)
    else if(elements_per_lane == 2 && chunks_per_block == 2)
      NT_KERNEL_LAUNCH_SYNC(read_kernel, 2, 2, T, dim3(block_cnt), dim3(tb_size), nullptr, stop_ev, d_a, d_c)
    else if(elements_per_lane == 4 && chunks_per_block == 4)
      NT_KERNEL_LAUNCH_SYNC(read_kernel, 4, 4, T, dim3(block_cnt), dim3(tb_size), nullptr, stop_ev, d_a, d_c)
    else if(elements_per_lane == 2 && chunks_per_block == 4)
      NT_KERNEL_LAUNCH_SYNC(read_kernel, 2, 4, T, dim3(block_cnt), dim3(tb_size), nullptr, stop_ev, d_a, d_c)
    else
      NT_KERNEL_LAUNCH_SYNC(read_kernel, 4, 2, T, dim3(block_cnt), dim3(tb_size), nullptr, stop_ev, d_a, d_c)
  }
  return kernel_time;
}

template <int nt_mode, unsigned int elements_per_lane, unsigned int chunks_per_block, typename T>
__launch_bounds__(TBSIZE)
__global__
void write_kernel(T * __restrict c)
{
  const auto dx = grid_size() * elements_per_lane;
  const auto gidx = (localid()) * elements_per_lane;

  for (auto i = 0u; i != chunks_per_block; ++i)
  {
    for (auto j = 0u; j != elements_per_lane; ++j)
    {
      store<nt_mode>(scalar<T>(startC), c[gidx + i * dx + j]);
    }
  }
}

template <class T>
float HIPStream<T>::write()
{
  float kernel_time = 0.;
  if (evt_timing)
  {
    if(elements_per_lane == 4 && chunks_per_block == 1)
      NT_KERNEL_LAUNCH_EVENTS(write_kernel, 4, 1, T, dim3(block_cnt), dim3(tb_size), nullptr, start_ev, stop_ev, d_c)
    else if(elements_per_lane == 2 && chunks_per_block == 1)
      NT_KERNEL_LAUNCH_EVENTS(write_kernel, 2, 1, T, dim3(block_cnt), dim3(tb_size), nullptr, start_ev, stop_ev, d_c)
    else if(elements_per_lane == 4 && chunks_per_block == 2)
      NT_KERNEL_LAUNCH_EVENTS(write_kernel, 4, 2, T, dim3(block_cnt), dim3(tb_size), nullptr, start_ev, stop_ev, d_c)
    else if(elements_per_lane == 2 && chunks_per_block == 2)
      NT_KERNEL_LAUNCH_EVENTS(write_kernel, 2, 2, T, dim3(block_cnt), dim3(tb_size), nullptr, start_ev, stop_ev, d_c)
    else if(elements_per_lane == 4 && chunks_per_block == 4)
      NT_KERNEL_LAUNCH_EVENTS(write_kernel, 4, 4, T, dim3(block_cnt), dim3(tb_size), nullptr, start_ev, stop_ev, d_c)
    else if(elements_per_lane == 2 && chunks_per_block == 4)
      NT_KERNEL_LAUNCH_EVENTS(write_kernel, 2, 4, T, dim3(block_cnt), dim3(tb_size), nullptr, start_ev, stop_ev, d_c)
    else
      NT_KERNEL_LAUNCH_EVENTS(write_kernel, 4, 2, T, dim3(block_cnt), dim3(tb_size), nullptr, start_ev, stop_ev, d_c)

    check_error(hipEventSynchronize(stop_ev));
    check_error(hipEventElapsedTime(&kernel_time, start_ev, stop_ev));
  }
  else
  {
    if(elements_per_lane == 4 && chunks_per_block == 1)
      NT_KERNEL_LAUNCH_SYNC(write_kernel, 4, 1, T, dim3(block_cnt), dim3(tb_size), nullptr, stop_ev, d_c)
    else if(elements_per_lane == 2 && chunks_per_block == 1)
      NT_KERNEL_LAUNCH_SYNC(write_kernel, 2, 1, T, dim3(block_cnt), dim3(tb_size), nullptr, stop_ev, d_c)
    else if(elements_per_lane == 4 && chunks_per_block == 2)
      NT_KERNEL_LAUNCH_SYNC(write_kernel, 4, 2, T, dim3(block_cnt), dim3(tb_size), nullptr, stop_ev, d_c)
    else if(elements_per_lane == 2 && chunks_per_block == 2)
      NT_KERNEL_LAUNCH_SYNC(write_kernel, 2, 2, T, dim3(block_cnt), dim3(tb_size), nullptr, stop_ev, d_c)
    else if(elements_per_lane == 4 && chunks_per_block == 4)
      NT_KERNEL_LAUNCH_SYNC(write_kernel, 4, 4, T, dim3(block_cnt), dim3(tb_size), nullptr, stop_ev, d_c)
    else if(elements_per_lane == 2 && chunks_per_block == 4)
      NT_KERNEL_LAUNCH_SYNC(write_kernel, 2, 4, T, dim3(block_cnt), dim3(tb_size), nullptr, stop_ev, d_c)
    else
      NT_KERNEL_LAUNCH_SYNC(write_kernel, 4, 2, T, dim3(block_cnt), dim3(tb_size), nullptr, stop_ev, d_c)
  }
  return kernel_time;
}

template <int nt_mode, unsigned int elements_per_lane, unsigned int chunks_per_block, typename T>
__launch_bounds__(TBSIZE)
__global__
void copy_kernel(const T * __restrict a, T * __restrict c)
{
  const auto dx = grid_size() * elements_per_lane;
  const auto gidx = (localid()) * elements_per_lane;

  for (auto i = 0u; i != chunks_per_block; ++i)
  {
    for (auto j = 0u; j != elements_per_lane; ++j)
    {
      store<nt_mode>(load<nt_mode>(a[gidx + i * dx + j]), c[gidx + i * dx + j]);
    }
  }
}

template <class T>
float HIPStream<T>::copy()
{
  float kernel_time = 0.;
  if (evt_timing)
  {
    if(elements_per_lane == 4 && chunks_per_block == 1)
      NT_KERNEL_LAUNCH_EVENTS(copy_kernel, 4, 1, T, dim3(block_cnt), dim3(tb_size), nullptr, start_ev, stop_ev, d_a, d_c)
    else if(elements_per_lane == 2 && chunks_per_block == 1)
      NT_KERNEL_LAUNCH_EVENTS(copy_kernel, 2, 1, T, dim3(block_cnt), dim3(tb_size), nullptr, start_ev, stop_ev, d_a, d_c)
    else if(elements_per_lane == 4 && chunks_per_block == 2)
      NT_KERNEL_LAUNCH_EVENTS(copy_kernel, 4, 2, T, dim3(block_cnt), dim3(tb_size), nullptr, start_ev, stop_ev, d_a, d_c)
    else if(elements_per_lane == 2 && chunks_per_block == 2)
      NT_KERNEL_LAUNCH_EVENTS(copy_kernel, 2, 2, T, dim3(block_cnt), dim3(tb_size), nullptr, start_ev, stop_ev, d_a, d_c)
    else if(elements_per_lane == 4 && chunks_per_block == 4)
      NT_KERNEL_LAUNCH_EVENTS(copy_kernel, 4, 4, T, dim3(block_cnt), dim3(tb_size), nullptr, start_ev, stop_ev, d_a, d_c)
    else if(elements_per_lane == 2 && chunks_per_block == 4)
      NT_KERNEL_LAUNCH_EVENTS(copy_kernel, 2, 4, T, dim3(block_cnt), dim3(tb_size), nullptr, start_ev, stop_ev, d_a, d_c)
    else
      NT_KERNEL_LAUNCH_EVENTS(copy_kernel, 4, 2, T, dim3(block_cnt), dim3(tb_size), nullptr, start_ev, stop_ev, d_a, d_c)

    check_error(hipEventSynchronize(stop_ev));
    check_error(hipEventElapsedTime(&kernel_time, start_ev, stop_ev));
  }
  else
  {
    if(elements_per_lane == 4 && chunks_per_block == 1)
      NT_KERNEL_LAUNCH_SYNC(copy_kernel, 4, 1, T, dim3(block_cnt), dim3(tb_size), nullptr, stop_ev, d_a, d_c)
    else if(elements_per_lane == 2 && chunks_per_block == 1)
      NT_KERNEL_LAUNCH_SYNC(copy_kernel, 2, 1, T, dim3(block_cnt), dim3(tb_size), nullptr, stop_ev, d_a, d_c)
    else if(elements_per_lane == 4 && chunks_per_block == 2)
      NT_KERNEL_LAUNCH_SYNC(copy_kernel, 4, 2, T, dim3(block_cnt), dim3(tb_size), nullptr, stop_ev, d_a, d_c)
    else if(elements_per_lane == 2 && chunks_per_block == 2)
      NT_KERNEL_LAUNCH_SYNC(copy_kernel, 2, 2, T, dim3(block_cnt), dim3(tb_size), nullptr, stop_ev, d_a, d_c)
    else if(elements_per_lane == 4 && chunks_per_block == 4)
      NT_KERNEL_LAUNCH_SYNC(copy_kernel, 4, 4, T, dim3(block_cnt), dim3(tb_size), nullptr, stop_ev, d_a, d_c)
    else if(elements_per_lane == 2 && chunks_per_block == 4)
      NT_KERNEL_LAUNCH_SYNC(copy_kernel, 2, 4, T, dim3(block_cnt), dim3(tb_size), nullptr, stop_ev, d_a, d_c)
    else
      NT_KERNEL_LAUNCH_SYNC(copy_kernel, 4, 2, T, dim3(block_cnt), dim3(tb_size), nullptr, stop_ev, d_a, d_c)
  }
  return kernel_time;
}

template <int nt_mode, unsigned int elements_per_lane, unsigned int chunks_per_block, typename T>
__launch_bounds__(TBSIZE)
__global__
void mul_kernel(T * __restrict b, const T * __restrict c)
{
  const auto dx = grid_size() * elements_per_lane;
  const auto gidx = (localid()) * elements_per_lane;

  for (auto i = 0u; i != chunks_per_block; ++i)
  {
    for (auto j = 0u; j != elements_per_lane; ++j)
    {
      store<nt_mode>(scalar<T>(startScalar) * load<nt_mode>(c[gidx + i * dx + j]), b[gidx + i * dx + j]);
    }
  }
}

template <class T>
float HIPStream<T>::mul()
{
  float kernel_time = 0.;
  if (evt_timing)
  {
    if(elements_per_lane == 4 && chunks_per_block == 1)
      NT_KERNEL_LAUNCH_EVENTS(mul_kernel, 4, 1, T, dim3(block_cnt), dim3(tb_size), nullptr, start_ev, stop_ev, d_b, d_c)
    else if(elements_per_lane == 2 && chunks_per_block == 1)
      NT_KERNEL_LAUNCH_EVENTS(mul_kernel, 2, 1, T, dim3(block_cnt), dim3(tb_size), nullptr, start_ev, stop_ev, d_b, d_c)
    else if(elements_per_lane == 4 && chunks_per_block == 2)
      NT_KERNEL_LAUNCH_EVENTS(mul_kernel, 4, 2, T, dim3(block_cnt), dim3(tb_size), nullptr, start_ev, stop_ev, d_b, d_c)
    else if(elements_per_lane == 2 && chunks_per_block == 2)
      NT_KERNEL_LAUNCH_EVENTS(mul_kernel, 2, 2, T, dim3(block_cnt), dim3(tb_size), nullptr, start_ev, stop_ev, d_b, d_c)
    else if(elements_per_lane == 4 && chunks_per_block == 4)
      NT_KERNEL_LAUNCH_EVENTS(mul_kernel, 4, 4, T, dim3(block_cnt), dim3(tb_size), nullptr, start_ev, stop_ev, d_b, d_c)
    else if(elements_per_lane == 2 && chunks_per_block == 4)
      NT_KERNEL_LAUNCH_EVENTS(mul_kernel, 2, 4, T, dim3(block_cnt), dim3(tb_size), nullptr, start_ev, stop_ev, d_b, d_c)
    else
      NT_KERNEL_LAUNCH_EVENTS(mul_kernel, 4, 2, T, dim3(block_cnt), dim3(tb_size), nullptr, start_ev, stop_ev, d_b, d_c)

    check_error(hipEventSynchronize(stop_ev));
    check_error(hipEventElapsedTime(&kernel_time, start_ev, stop_ev));
  }
  else
  {
    if(elements_per_lane == 4 && chunks_per_block == 1)
      NT_KERNEL_LAUNCH_SYNC(mul_kernel, 4, 1, T, dim3(block_cnt), dim3(tb_size), nullptr, stop_ev, d_b, d_c)
    else if(elements_per_lane == 2 && chunks_per_block == 1)
      NT_KERNEL_LAUNCH_SYNC(mul_kernel, 2, 1, T, dim3(block_cnt), dim3(tb_size), nullptr, stop_ev, d_b, d_c)
    else if(elements_per_lane == 4 && chunks_per_block == 2)
      NT_KERNEL_LAUNCH_SYNC(mul_kernel, 4, 2, T, dim3(block_cnt), dim3(tb_size), nullptr, stop_ev, d_b, d_c)
    else if(elements_per_lane == 2 && chunks_per_block == 2)
      NT_KERNEL_LAUNCH_SYNC(mul_kernel, 2, 2, T, dim3(block_cnt), dim3(tb_size), nullptr, stop_ev, d_b, d_c)
    else if(elements_per_lane == 4 && chunks_per_block == 4)
      NT_KERNEL_LAUNCH_SYNC(mul_kernel, 4, 4, T, dim3(block_cnt), dim3(tb_size), nullptr, stop_ev, d_b, d_c)
    else if(elements_per_lane == 2 && chunks_per_block == 4)
      NT_KERNEL_LAUNCH_SYNC(mul_kernel, 2, 4, T, dim3(block_cnt), dim3(tb_size), nullptr, stop_ev, d_b, d_c)
    else
      NT_KERNEL_LAUNCH_SYNC(mul_kernel, 4, 2, T, dim3(block_cnt), dim3(tb_size), nullptr, stop_ev, d_b, d_c)
  }
  return kernel_time;
}

template <int nt_mode, unsigned int elements_per_lane, unsigned int chunks_per_block, typename T>
__launch_bounds__(TBSIZE)
__global__
void add_kernel(const T * __restrict a, const T * __restrict b,
                T * __restrict c)
{
  const auto dx = grid_size() * elements_per_lane;
  const auto gidx = (localid()) * elements_per_lane;

  for (auto i = 0u; i != chunks_per_block; ++i)
  {
    for (auto j = 0u; j != elements_per_lane; ++j)
    {
      store<nt_mode>(load<nt_mode>(a[gidx + i * dx + j]) + load<nt_mode>(b[gidx + i * dx + j]), c[gidx + i * dx + j]);
    }
  }
}

template <class T>
float HIPStream<T>::add()
{
  float kernel_time = 0.;
  if (evt_timing)
  {
    if(elements_per_lane == 4 && chunks_per_block == 1)
      NT_KERNEL_LAUNCH_EVENTS(add_kernel, 4, 1, T, dim3(block_cnt), dim3(tb_size), nullptr, start_ev, stop_ev, d_a, d_b, d_c)
    else if(elements_per_lane == 2 && chunks_per_block == 1)
      NT_KERNEL_LAUNCH_EVENTS(add_kernel, 2, 1, T, dim3(block_cnt), dim3(tb_size), nullptr, start_ev, stop_ev, d_a, d_b, d_c)
    else if(elements_per_lane == 4 && chunks_per_block == 2)
      NT_KERNEL_LAUNCH_EVENTS(add_kernel, 4, 2, T, dim3(block_cnt), dim3(tb_size), nullptr, start_ev, stop_ev, d_a, d_b, d_c)
    else if(elements_per_lane == 2 && chunks_per_block == 2)
      NT_KERNEL_LAUNCH_EVENTS(add_kernel, 2, 2, T, dim3(block_cnt), dim3(tb_size), nullptr, start_ev, stop_ev, d_a, d_b, d_c)
    else if(elements_per_lane == 4 && chunks_per_block == 4)
      NT_KERNEL_LAUNCH_EVENTS(add_kernel, 4, 4, T, dim3(block_cnt), dim3(tb_size), nullptr, start_ev, stop_ev, d_a, d_b, d_c)
    else if(elements_per_lane == 2 && chunks_per_block == 4)
      NT_KERNEL_LAUNCH_EVENTS(add_kernel, 2, 4, T, dim3(block_cnt), dim3(tb_size), nullptr, start_ev, stop_ev, d_a, d_b, d_c)
    else
      NT_KERNEL_LAUNCH_EVENTS(add_kernel, 4, 2, T, dim3(block_cnt), dim3(tb_size), nullptr, start_ev, stop_ev, d_a, d_b, d_c)

    check_error(hipEventSynchronize(stop_ev));
    check_error(hipEventElapsedTime(&kernel_time, start_ev, stop_ev));
  }
  else
  {
    if(elements_per_lane == 4 && chunks_per_block == 1)
      NT_KERNEL_LAUNCH_SYNC(add_kernel, 4, 1, T, dim3(block_cnt), dim3(tb_size), nullptr, stop_ev, d_a, d_b, d_c)
    else if(elements_per_lane == 2 && chunks_per_block == 1)
      NT_KERNEL_LAUNCH_SYNC(add_kernel, 2, 1, T, dim3(block_cnt), dim3(tb_size), nullptr, stop_ev, d_a, d_b, d_c)
    else if(elements_per_lane == 4 && chunks_per_block == 2)
      NT_KERNEL_LAUNCH_SYNC(add_kernel, 4, 2, T, dim3(block_cnt), dim3(tb_size), nullptr, stop_ev, d_a, d_b, d_c)
    else if(elements_per_lane == 2 && chunks_per_block == 2)
      NT_KERNEL_LAUNCH_SYNC(add_kernel, 2, 2, T, dim3(block_cnt), dim3(tb_size), nullptr, stop_ev, d_a, d_b, d_c)
    else if(elements_per_lane == 4 && chunks_per_block == 4)
      NT_KERNEL_LAUNCH_SYNC(add_kernel, 4, 4, T, dim3(block_cnt), dim3(tb_size), nullptr, stop_ev, d_a, d_b, d_c)
    else if(elements_per_lane == 2 && chunks_per_block == 4)
      NT_KERNEL_LAUNCH_SYNC(add_kernel, 2, 4, T, dim3(block_cnt), dim3(tb_size), nullptr, stop_ev, d_a, d_b, d_c)
    else
      NT_KERNEL_LAUNCH_SYNC(add_kernel, 4, 2, T, dim3(block_cnt), dim3(tb_size), nullptr, stop_ev, d_a, d_b, d_c)
  }
  return kernel_time;
}

template <int nt_mode, unsigned int elements_per_lane, unsigned int chunks_per_block, typename T>
__launch_bounds__(TBSIZE)
__global__
void triad_kernel(T * __restrict a, const T * __restrict b,
                  const T * __restrict c)
{
  const auto dx = grid_size() * elements_per_lane;
  const auto gidx = (localid()) * elements_per_lane;

  for (auto i = 0u; i != chunks_per_block; ++i)
  {
    for (auto j = 0u; j != elements_per_lane; ++j)
    {
      store<nt_mode>(load<nt_mode>(b[gidx + i * dx + j]) + scalar<T>(startScalar) * load<nt_mode>(c[gidx + i * dx + j]),
            a[gidx + i * dx + j]);
    }
  }
}

template <class T>
float HIPStream<T>::triad()
{
  float kernel_time = 0.;
  if (evt_timing)
  {
    if(elements_per_lane == 4 && chunks_per_block == 1)
      NT_KERNEL_LAUNCH_EVENTS(triad_kernel, 4, 1, T, dim3(block_cnt), dim3(tb_size), nullptr, start_ev, stop_ev, d_a, d_b, d_c)
    else if(elements_per_lane == 2 && chunks_per_block == 1)
      NT_KERNEL_LAUNCH_EVENTS(triad_kernel, 2, 1, T, dim3(block_cnt), dim3(tb_size), nullptr, start_ev, stop_ev, d_a, d_b, d_c)
    else if(elements_per_lane == 4 && chunks_per_block == 2)
      NT_KERNEL_LAUNCH_EVENTS(triad_kernel, 4, 2, T, dim3(block_cnt), dim3(tb_size), nullptr, start_ev, stop_ev, d_a, d_b, d_c)
    else if(elements_per_lane == 2 && chunks_per_block == 2)
      NT_KERNEL_LAUNCH_EVENTS(triad_kernel, 2, 2, T, dim3(block_cnt), dim3(tb_size), nullptr, start_ev, stop_ev, d_a, d_b, d_c)
    else if(elements_per_lane == 4 && chunks_per_block == 4)
      NT_KERNEL_LAUNCH_EVENTS(triad_kernel, 4, 4, T, dim3(block_cnt), dim3(tb_size), nullptr, start_ev, stop_ev, d_a, d_b, d_c)
    else if(elements_per_lane == 2 && chunks_per_block == 4)
      NT_KERNEL_LAUNCH_EVENTS(triad_kernel, 2, 4, T, dim3(block_cnt), dim3(tb_size), nullptr, start_ev, stop_ev, d_a, d_b, d_c)
    else
      NT_KERNEL_LAUNCH_EVENTS(triad_kernel, 4, 2, T, dim3(block_cnt), dim3(tb_size), nullptr, start_ev, stop_ev, d_a, d_b, d_c)

    check_error(hipEventSynchronize(stop_ev));
    check_error(hipEventElapsedTime(&kernel_time, start_ev, stop_ev));
  }
  else
  {
    if(elements_per_lane == 4 && chunks_per_block == 1)
      NT_KERNEL_LAUNCH_SYNC(triad_kernel, 4, 1, T, dim3(block_cnt), dim3(tb_size), nullptr, stop_ev, d_a, d_b, d_c)
    else if(elements_per_lane == 2 && chunks_per_block == 1)
      NT_KERNEL_LAUNCH_SYNC(triad_kernel, 2, 1, T, dim3(block_cnt), dim3(tb_size), nullptr, stop_ev, d_a, d_b, d_c)
    else if(elements_per_lane == 4 && chunks_per_block == 2)
      NT_KERNEL_LAUNCH_SYNC(triad_kernel, 4, 2, T, dim3(block_cnt), dim3(tb_size), nullptr, stop_ev, d_a, d_b, d_c)
    else if(elements_per_lane == 2 && chunks_per_block == 2)
      NT_KERNEL_LAUNCH_SYNC(triad_kernel, 2, 2, T, dim3(block_cnt), dim3(tb_size), nullptr, stop_ev, d_a, d_b, d_c)
    else if(elements_per_lane == 4 && chunks_per_block == 4)
      NT_KERNEL_LAUNCH_SYNC(triad_kernel, 4, 4, T, dim3(block_cnt), dim3(tb_size), nullptr, stop_ev, d_a, d_b, d_c)
    else if(elements_per_lane == 2 && chunks_per_block == 4)
      NT_KERNEL_LAUNCH_SYNC(triad_kernel, 2, 4, T, dim3(block_cnt), dim3(tb_size), nullptr, stop_ev, d_a, d_b, d_c)
    else
      NT_KERNEL_LAUNCH_SYNC(triad_kernel, 4, 2, T, dim3(block_cnt), dim3(tb_size), nullptr, stop_ev, d_a, d_b, d_c)
  }
  return kernel_time;
}

template<unsigned int n>
struct Reducer {
  template<typename I>
  __device__
  static
  void reduce(I it) noexcept
  {
    if (n == 1) return;

#if defined(__HIP_PLATFORM_NVCC__)
    constexpr unsigned int warpSize = 32;
#endif
    const bool is_same_warp{n <= warpSize * 2};
    if (static_cast<int>(threadIdx.x) < n / 2)
    {
      it[threadIdx.x] += it[threadIdx.x + n / 2];
    }
    is_same_warp ? __threadfence_block() : __syncthreads();

    Reducer<n / 2>::reduce(it);
  }
};

template<>
struct Reducer<1u> {
  template<typename I>
  __device__
  static
  void reduce(I) noexcept
  {}
};

template <int nt_mode, unsigned int elements_per_lane, unsigned int chunks_per_block, typename T, unsigned int tb_size>
__launch_bounds__(TBSIZE)
__global__
void dot_kernel(const T * __restrict a, const T * __restrict b,
                T * __restrict sum)
{
  const auto dx = grid_size() * elements_per_lane;
  const auto gidx = (localid()) * elements_per_lane;

  T tmp{0};
  for (auto i = 0u; i != chunks_per_block; ++i)
  {
    for (auto j = 0u; j != elements_per_lane; ++j)
    {
      tmp += load<nt_mode>(a[gidx + i * dx + j]) * load<nt_mode>(b[gidx + i * dx + j]);
    }
  }

  __shared__ T tb_sum[tb_size];
  tb_sum[threadIdx.x] = tmp;

  __syncthreads();

  Reducer<tb_size>::reduce(tb_sum);

  if (threadIdx.x)
  {
    return;
  }
  store<nt_mode>(tb_sum[0], sum[blockIdx.x]);
}

template <class T>
T HIPStream<T>::dot()
{
  if(elements_per_lane == 4 && chunks_per_block == 1) {
    if(tb_size == 1024) {
      NT_KERNEL_LAUNCH_DOT_SYNC(dot_kernel, 4, 1, T, 1024, dim3(block_cnt), dim3(tb_size), nullptr, coherent_ev, d_a, d_b, sums)
    }
    else if(tb_size == 512) {
      NT_KERNEL_LAUNCH_DOT_SYNC(dot_kernel, 4, 1, T, 512, dim3(block_cnt), dim3(tb_size), nullptr, coherent_ev, d_a, d_b, sums)
    }
  }
  else if(elements_per_lane == 2 && chunks_per_block == 1) {
    if(tb_size == 1024) {
      NT_KERNEL_LAUNCH_DOT_SYNC(dot_kernel, 2, 1, T, 1024, dim3(block_cnt), dim3(tb_size), nullptr, coherent_ev, d_a, d_b, sums)
    }
    else if(tb_size == 512) {
      NT_KERNEL_LAUNCH_DOT_SYNC(dot_kernel, 2, 1, T, 512, dim3(block_cnt), dim3(tb_size), nullptr, coherent_ev, d_a, d_b, sums)
    }
  }
  else if(elements_per_lane == 4 && chunks_per_block == 2) {
    if(tb_size == 1024) {
      NT_KERNEL_LAUNCH_DOT_SYNC(dot_kernel, 4, 2, T, 1024, dim3(block_cnt), dim3(tb_size), nullptr, coherent_ev, d_a, d_b, sums)
    }
    else if(tb_size == 512) {
      NT_KERNEL_LAUNCH_DOT_SYNC(dot_kernel, 4, 2, T, 512, dim3(block_cnt), dim3(tb_size), nullptr, coherent_ev, d_a, d_b, sums)
    }
  }
  else if(elements_per_lane == 2 && chunks_per_block == 2) {
    if(tb_size == 1024) {
      NT_KERNEL_LAUNCH_DOT_SYNC(dot_kernel, 2, 2, T, 1024, dim3(block_cnt), dim3(tb_size), nullptr, coherent_ev, d_a, d_b, sums)
    }
    else if(tb_size == 512) {
      NT_KERNEL_LAUNCH_DOT_SYNC(dot_kernel, 2, 2, T, 512, dim3(block_cnt), dim3(tb_size), nullptr, coherent_ev, d_a, d_b, sums)
    }
  }
  else if(elements_per_lane == 4 && chunks_per_block == 4) {
    if(tb_size == 1024) {
      NT_KERNEL_LAUNCH_DOT_SYNC(dot_kernel, 4, 4, T, 1024, dim3(block_cnt), dim3(tb_size), nullptr, coherent_ev, d_a, d_b, sums)
    }
    else if(tb_size == 512) {
      NT_KERNEL_LAUNCH_DOT_SYNC(dot_kernel, 4, 4, T, 512, dim3(block_cnt), dim3(tb_size), nullptr, coherent_ev, d_a, d_b, sums)
    }
  }
  else if(elements_per_lane == 2 && chunks_per_block == 4) {
    if(tb_size == 1024) {
      NT_KERNEL_LAUNCH_DOT_SYNC(dot_kernel, 2, 4, T, 1024, dim3(block_cnt), dim3(tb_size), nullptr, coherent_ev, d_a, d_b, sums)
    }
    else if(tb_size == 512) {
      NT_KERNEL_LAUNCH_DOT_SYNC(dot_kernel, 2, 4, T, 512, dim3(block_cnt), dim3(tb_size), nullptr, coherent_ev, d_a, d_b, sums)
    }
  }
  else
    NT_KERNEL_LAUNCH_DOT_SYNC(dot_kernel, 4, 2, T, 1024, dim3(block_cnt), dim3(tb_size), nullptr, coherent_ev, d_a, d_b, sums)

  T sum{0};
  for (auto i = 0u; i != block_cnt; ++i)
  {
    sum += sums[i];
  }

  return sum;
}

void listDevices(void)
{
  // Get number of devices
  int count;
  check_error(hipGetDeviceCount(&count));
  std::string msg;
  // Print device names
  if (count == 0)
  {
    rvs::lp::Log("No devices found", rvs::logerror);
  }
  else
  {
    // std::cout << std::endl;
    msg = "Devices:\n" ;
    for (int i = 0; i < count; i++)
    {
      msg += std::to_string(i) + ": " + getDeviceName(i) + "\n";
    }
    rvs::lp::Log(msg, rvs::logresults);
  }
}


std::string getDeviceName(const int device)
{
  hipDeviceProp_t props;
  check_error(hipGetDeviceProperties(&props, device));
  return std::string(props.name);
}


std::string getDeviceDriver(const int device)
{
  check_error(hipSetDevice(device));
  int driver;
  check_error(hipDriverGetVersion(&driver));
  return std::to_string(driver);
}

template class HIPStream<float>;
template class HIPStream<double>;
