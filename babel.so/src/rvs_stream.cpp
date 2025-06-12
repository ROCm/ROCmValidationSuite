// Copyright (c) 2014-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code


#include "include/HIPStream.h"
#include "hip/hip_runtime.h"
#include "include/rvsloglp.h"

#include <cfloat>

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
    const unsigned int _threads_per_block)
  : array_size{ARRAY_SIZE}, evt_timing(event_timing),
  dwords_per_lane(_dwords_per_lane), chunks_per_block(_chunks_per_block), tb_size(_threads_per_block)
{
  std::string msg;

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
void HIPStream<T>::read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c)
{
  check_error(hipDeviceSynchronize());
  // Copy device memory to host
  check_error(hipMemcpy(a.data(), d_a, a.size()*sizeof(T), hipMemcpyDeviceToHost));
  check_error(hipMemcpy(b.data(), d_b, b.size()*sizeof(T), hipMemcpyDeviceToHost));
  check_error(hipMemcpy(c.data(), d_c, c.size()*sizeof(T), hipMemcpyDeviceToHost));
}


// turn on non-temporal by default
#ifndef NONTEMPORAL
#define NONTEMPORAL 1
#endif

#if NONTEMPORAL == 0
template<typename T>
__device__ __forceinline__ T load(const T& ref) {
  return ref;
}

template<typename T>
__device__ __forceinline__ void store(const T& value, T& ref) {
  ref = value;
}
#else
template<typename T>
__device__ __forceinline__ T load(const T& ref) {
  return __builtin_nontemporal_load(&ref);
}

template<typename T>
__device__ __forceinline__ void store(const T& value, T& ref) {
  __builtin_nontemporal_store(value, &ref);
}
#endif

template <unsigned int elements_per_lane, unsigned int chunks_per_block, typename T>
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
      tmp += load(a[gidx + i * dx + j]);
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
    // Support for dwords_per_lane = 4 & chunks_per_block = 1/2/4
    if(elements_per_lane == 4 && chunks_per_block == 1)
      hipLaunchKernelWithEvents(read_kernel<4, 1, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, start_ev,
          stop_ev, d_a, d_c);
    else if(elements_per_lane == 2 && chunks_per_block == 1)
      hipLaunchKernelWithEvents(read_kernel<2, 1, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, start_ev,
          stop_ev, d_a, d_c);
    else if(elements_per_lane == 4 && chunks_per_block == 2)
      hipLaunchKernelWithEvents(read_kernel<4, 2, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, start_ev,
          stop_ev, d_a, d_c);
    else if(elements_per_lane == 2 && chunks_per_block == 2)
      hipLaunchKernelWithEvents(read_kernel<2, 2, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, start_ev,
          stop_ev, d_a, d_c);
    else if(elements_per_lane == 4 && chunks_per_block == 4)
      hipLaunchKernelWithEvents(read_kernel<4, 4, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, start_ev,
          stop_ev, d_a, d_c);
    else if(elements_per_lane == 2 && chunks_per_block == 4)
      hipLaunchKernelWithEvents(read_kernel<2, 4, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, start_ev,
          stop_ev, d_a, d_c);
    else
      hipLaunchKernelWithEvents(read_kernel<4, 2, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, start_ev,
          stop_ev, d_a, d_c);

    check_error(hipEventSynchronize(stop_ev));
    check_error(hipEventElapsedTime(&kernel_time, start_ev, stop_ev));
  }
  else
  {
    if(elements_per_lane == 4 && chunks_per_block == 1)
      hipLaunchKernelSynchronous(read_kernel<4, 1, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, stop_ev,
          d_a, d_c);
    else if(elements_per_lane == 2 && chunks_per_block == 1)
      hipLaunchKernelSynchronous(read_kernel<2, 1, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, stop_ev,
          d_a, d_c);
    else if(elements_per_lane == 4 && chunks_per_block == 2)
      hipLaunchKernelSynchronous(read_kernel<4, 2, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, stop_ev,
          d_a, d_c);
    else if(elements_per_lane == 2 && chunks_per_block == 2)
      hipLaunchKernelSynchronous(read_kernel<2, 2, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, stop_ev,
          d_a, d_c);
    else if(elements_per_lane == 4 && chunks_per_block == 4)
      hipLaunchKernelSynchronous(read_kernel<4, 4, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, stop_ev,
          d_a, d_c);
    else if(elements_per_lane == 2 && chunks_per_block == 4)
      hipLaunchKernelSynchronous(read_kernel<2, 4, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, stop_ev,
          d_a, d_c);
    else
      hipLaunchKernelSynchronous(read_kernel<4, 2, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, stop_ev,
          d_a, d_c);
  }
  return kernel_time;
}

template <unsigned int elements_per_lane, unsigned int chunks_per_block, typename T>
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
      store(scalar<T>(startC), c[gidx + i * dx + j]);
    }
  }
}

template <class T>
float HIPStream<T>::write()
{
  float kernel_time = 0.;
  if (evt_timing)
  {
    // Support for dwords_per_lane = 4 & chunks_per_block = 1/2/4
    if(elements_per_lane == 4 && chunks_per_block == 1)
      hipLaunchKernelWithEvents(write_kernel<4, 1, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, start_ev,
          stop_ev, d_c);
    else if(elements_per_lane == 2 && chunks_per_block == 1)
      hipLaunchKernelWithEvents(write_kernel<2, 1, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, start_ev,
          stop_ev, d_c);
    else if(elements_per_lane == 4 && chunks_per_block == 2)
      hipLaunchKernelWithEvents(write_kernel<4, 2, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, start_ev,
          stop_ev, d_c);
    else if(elements_per_lane == 2 && chunks_per_block == 2)
      hipLaunchKernelWithEvents(write_kernel<2, 2, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, start_ev,
          stop_ev, d_c);
    else if(elements_per_lane == 4 && chunks_per_block == 4)
      hipLaunchKernelWithEvents(write_kernel<4, 4, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, start_ev,
          stop_ev, d_c);
    else if(elements_per_lane == 2 && chunks_per_block == 4)
      hipLaunchKernelWithEvents(write_kernel<2, 4, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, start_ev,
          stop_ev, d_c);
    else
      hipLaunchKernelWithEvents(write_kernel<4, 2, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, start_ev,
          stop_ev, d_c);

    check_error(hipEventSynchronize(stop_ev));
    check_error(hipEventElapsedTime(&kernel_time, start_ev, stop_ev));
  }
  else
  {
    if(elements_per_lane == 4 && chunks_per_block == 1)
      hipLaunchKernelSynchronous(write_kernel<4, 1, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, stop_ev,
          d_c);
    else if(elements_per_lane == 2 && chunks_per_block == 1)
      hipLaunchKernelSynchronous(write_kernel<2, 1, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, stop_ev,
          d_c);
    else if(elements_per_lane == 4 && chunks_per_block == 2)
      hipLaunchKernelSynchronous(write_kernel<4, 2, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, stop_ev,
          d_c);
    else if(elements_per_lane == 2 && chunks_per_block == 2)
      hipLaunchKernelSynchronous(write_kernel<2, 2, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, stop_ev,
          d_c);
    else if(elements_per_lane == 4 && chunks_per_block == 4)
      hipLaunchKernelSynchronous(write_kernel<4, 4, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, stop_ev,
          d_c);
    else if(elements_per_lane == 2 && chunks_per_block == 4)
      hipLaunchKernelSynchronous(write_kernel<2, 4, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, stop_ev,
          d_c);
    else
      hipLaunchKernelSynchronous(write_kernel<4, 2, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, stop_ev,
          d_c);
  }
  return kernel_time;
}

template <unsigned int elements_per_lane, unsigned int chunks_per_block, typename T>
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
      store(load(a[gidx + i * dx + j]), c[gidx + i * dx + j]);
    }
  }
}

template <class T>
float HIPStream<T>::copy()
{
  float kernel_time = 0.;
  if (evt_timing)
  {
    // Support for dwords_per_lane = 4 & chunks_per_block = 1/2/4
    if(elements_per_lane == 4 && chunks_per_block == 1)
      hipLaunchKernelWithEvents(copy_kernel<4, 1, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, start_ev,
          stop_ev, d_a, d_c);
    else if(elements_per_lane == 2 && chunks_per_block == 1)
      hipLaunchKernelWithEvents(copy_kernel<2, 1, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, start_ev,
          stop_ev, d_a, d_c);
    else if(elements_per_lane == 4 && chunks_per_block == 2)
      hipLaunchKernelWithEvents(copy_kernel<4, 2, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, start_ev,
          stop_ev, d_a, d_c);
    else if(elements_per_lane == 2 && chunks_per_block == 2)
      hipLaunchKernelWithEvents(copy_kernel<2, 2, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, start_ev,
          stop_ev, d_a, d_c);
    else if(elements_per_lane == 4 && chunks_per_block == 4)
      hipLaunchKernelWithEvents(copy_kernel<4, 4, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, start_ev,
          stop_ev, d_a, d_c);
    else if(elements_per_lane == 2 && chunks_per_block == 4)
      hipLaunchKernelWithEvents(copy_kernel<2, 4, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, start_ev,
          stop_ev, d_a, d_c);
    else
      hipLaunchKernelWithEvents(copy_kernel<4, 2, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, start_ev,
          stop_ev, d_a, d_c);

    check_error(hipEventSynchronize(stop_ev));
    check_error(hipEventElapsedTime(&kernel_time, start_ev, stop_ev));
  }
  else
  {
    if(elements_per_lane == 4 && chunks_per_block == 1)
      hipLaunchKernelSynchronous(copy_kernel<4, 1, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, stop_ev,
          d_a, d_c);
    else if(elements_per_lane == 2 && chunks_per_block == 1)
      hipLaunchKernelSynchronous(copy_kernel<2, 1, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, stop_ev,
          d_a, d_c);
    else if(elements_per_lane == 4 && chunks_per_block == 2)
      hipLaunchKernelSynchronous(copy_kernel<4, 2, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, stop_ev,
          d_a, d_c);
    else if(elements_per_lane == 2 && chunks_per_block == 2)
      hipLaunchKernelSynchronous(copy_kernel<2, 2, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, stop_ev,
          d_a, d_c);
    else if(elements_per_lane == 4 && chunks_per_block == 4)
      hipLaunchKernelSynchronous(copy_kernel<4, 4, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, stop_ev,
          d_a, d_c);
    else if(elements_per_lane == 2 && chunks_per_block == 4)
      hipLaunchKernelSynchronous(copy_kernel<2, 4, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, stop_ev,
          d_a, d_c);
    else
      hipLaunchKernelSynchronous(copy_kernel<4, 2, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, stop_ev,
          d_a, d_c);
  }
  return kernel_time;
}

template <unsigned int elements_per_lane, unsigned int chunks_per_block, typename T>
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
      store(scalar<T>(startScalar) * load(c[gidx + i * dx + j]), b[gidx + i * dx + j]);
    }
  }
}

template <class T>
float HIPStream<T>::mul()
{
  float kernel_time = 0.;
  if (evt_timing)
  {
    // Support for dwords_per_lane = 4 & chunks_per_block = 1/2/4
    if(elements_per_lane == 4 && chunks_per_block == 1)
      hipLaunchKernelWithEvents(mul_kernel<4, 1, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, start_ev,
          stop_ev, d_b, d_c);
    else if(elements_per_lane == 2 && chunks_per_block == 1)
      hipLaunchKernelWithEvents(mul_kernel<2, 1, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, start_ev,
          stop_ev, d_b, d_c);
    else if(elements_per_lane == 4 && chunks_per_block == 2)
      hipLaunchKernelWithEvents(mul_kernel<4, 2, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, start_ev,
          stop_ev, d_b, d_c);
    else if(elements_per_lane == 2 && chunks_per_block == 2)
      hipLaunchKernelWithEvents(mul_kernel<2, 2, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, start_ev,
          stop_ev, d_b, d_c);
    else if(elements_per_lane == 4 && chunks_per_block == 4)
      hipLaunchKernelWithEvents(mul_kernel<4, 4, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, start_ev,
          stop_ev, d_b, d_c);
    else if(elements_per_lane == 2 && chunks_per_block == 4)
      hipLaunchKernelWithEvents(mul_kernel<2, 4, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, start_ev,
          stop_ev, d_b, d_c);
    else
      hipLaunchKernelWithEvents(mul_kernel<4, 2, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, start_ev,
          stop_ev, d_b, d_c);

    check_error(hipEventSynchronize(stop_ev));
    check_error(hipEventElapsedTime(&kernel_time, start_ev, stop_ev));
  }
  else
  {
    if(elements_per_lane == 4 && chunks_per_block == 1)
      hipLaunchKernelSynchronous(mul_kernel<4, 1, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, stop_ev,
          d_b, d_c);
    else if(elements_per_lane == 2 && chunks_per_block == 1)
      hipLaunchKernelSynchronous(mul_kernel<2, 1, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, stop_ev,
          d_b, d_c);
    else if(elements_per_lane == 4 && chunks_per_block == 2)
      hipLaunchKernelSynchronous(mul_kernel<4, 2, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, stop_ev,
          d_b, d_c);
    else if(elements_per_lane == 2 && chunks_per_block == 2)
      hipLaunchKernelSynchronous(mul_kernel<2, 2, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, stop_ev,
          d_b, d_c);
    else if(elements_per_lane == 4 && chunks_per_block == 4)
      hipLaunchKernelSynchronous(mul_kernel<4, 4, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, stop_ev,
          d_b, d_c);
    else if(elements_per_lane == 2 && chunks_per_block == 4)
      hipLaunchKernelSynchronous(mul_kernel<2, 4, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, stop_ev,
          d_b, d_c);
    else
      hipLaunchKernelSynchronous(mul_kernel<4, 2, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, stop_ev,
          d_b, d_c);
  }
  return kernel_time;
}

template <unsigned int elements_per_lane, unsigned int chunks_per_block, typename T>
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
      store(load(a[gidx + i * dx + j]) + load(b[gidx + i * dx + j]), c[gidx + i * dx + j]);
    }
  }
}

template <class T>
float HIPStream<T>::add()
{
  float kernel_time = 0.;
  if (evt_timing)
  {
    // Support for dwords_per_lane = 4 & chunks_per_block = 1/2/4
    if(elements_per_lane == 4 && chunks_per_block == 1)
      hipLaunchKernelWithEvents(add_kernel<4, 1, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, start_ev,
          stop_ev, d_a, d_b, d_c);
    else if(elements_per_lane == 2 && chunks_per_block == 1)
      hipLaunchKernelWithEvents(add_kernel<2, 1, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, start_ev,
          stop_ev, d_a, d_b, d_c);
    else if(elements_per_lane == 4 && chunks_per_block == 2)
      hipLaunchKernelWithEvents(add_kernel<4, 2, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, start_ev,
          stop_ev, d_a, d_b, d_c);
    else if(elements_per_lane == 2 && chunks_per_block == 2)
      hipLaunchKernelWithEvents(add_kernel<2, 2, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, start_ev,
          stop_ev, d_a, d_b, d_c);
    else if(elements_per_lane == 4 && chunks_per_block == 4)
      hipLaunchKernelWithEvents(add_kernel<4, 4, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, start_ev,
          stop_ev, d_a, d_b, d_c);
    else if(elements_per_lane == 2 && chunks_per_block == 4)
      hipLaunchKernelWithEvents(add_kernel<2, 4, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, start_ev,
          stop_ev, d_a, d_b, d_c);
    else
      hipLaunchKernelWithEvents(add_kernel<4, 2, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, start_ev,
          stop_ev, d_a, d_b, d_c);

    check_error(hipEventSynchronize(stop_ev));
    check_error(hipEventElapsedTime(&kernel_time, start_ev, stop_ev));
  }
  else
  {
    if(elements_per_lane == 4 && chunks_per_block == 1)
      hipLaunchKernelSynchronous(add_kernel<4, 1, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, stop_ev,
          d_a, d_b, d_c);
    else if(elements_per_lane == 2 && chunks_per_block == 1)
      hipLaunchKernelSynchronous(add_kernel<2, 1, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, stop_ev,
          d_a, d_b, d_c);
    else if(elements_per_lane == 4 && chunks_per_block == 2)
      hipLaunchKernelSynchronous(add_kernel<4, 2, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, stop_ev,
          d_a, d_b, d_c);
    else if(elements_per_lane == 2 && chunks_per_block == 2)
      hipLaunchKernelSynchronous(add_kernel<2, 2, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, stop_ev,
          d_a, d_b, d_c);
    else if(elements_per_lane == 4 && chunks_per_block == 4)
      hipLaunchKernelSynchronous(add_kernel<4, 4, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, stop_ev,
          d_a, d_b, d_c);
    else if(elements_per_lane == 2 && chunks_per_block == 4)
      hipLaunchKernelSynchronous(add_kernel<2, 4, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, stop_ev,
          d_a, d_b, d_c);
    else
      hipLaunchKernelSynchronous(add_kernel<4, 2, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, stop_ev,
          d_a, d_b, d_c);
  }
  return kernel_time;
}

template <unsigned int elements_per_lane, unsigned int chunks_per_block, typename T>
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
      store(load(b[gidx + i * dx + j]) + scalar<T>(startScalar) * load(c[gidx + i * dx + j]),
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
    // Support for dwords_per_lane = 4 & chunks_per_block = 1/2/4
    if(elements_per_lane == 4 && chunks_per_block == 1)
      hipLaunchKernelWithEvents(triad_kernel<4, 1, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, start_ev,
          stop_ev, d_a, d_b, d_c);
    else if(elements_per_lane == 2 && chunks_per_block == 1)
      hipLaunchKernelWithEvents(triad_kernel<2, 1, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, start_ev,
          stop_ev, d_a, d_b, d_c);
    else if(elements_per_lane == 4 && chunks_per_block == 2)
      hipLaunchKernelWithEvents(triad_kernel<4, 2, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, start_ev,
          stop_ev, d_a, d_b, d_c);
    else if(elements_per_lane == 2 && chunks_per_block == 2)
      hipLaunchKernelWithEvents(triad_kernel<2, 2, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, start_ev,
          stop_ev, d_a, d_b, d_c);
    else if(elements_per_lane == 4 && chunks_per_block == 4)
      hipLaunchKernelWithEvents(triad_kernel<4, 4, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, start_ev,
          stop_ev, d_a, d_b, d_c);
    else if(elements_per_lane == 2 && chunks_per_block == 4)
      hipLaunchKernelWithEvents(triad_kernel<2, 4, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, start_ev,
          stop_ev, d_a, d_b, d_c);
    else
      hipLaunchKernelWithEvents(triad_kernel<4, 2, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, start_ev,
          stop_ev, d_a, d_b, d_c);

    check_error(hipEventSynchronize(stop_ev));
    check_error(hipEventElapsedTime(&kernel_time, start_ev, stop_ev));
  }
  else
  {
    if(elements_per_lane == 4 && chunks_per_block == 1)
      hipLaunchKernelSynchronous(triad_kernel<4, 1, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, stop_ev,
          d_a, d_b, d_c);
    else if(elements_per_lane == 2 && chunks_per_block == 1)
      hipLaunchKernelSynchronous(triad_kernel<2, 1, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, stop_ev,
          d_a, d_b, d_c);
    else if(elements_per_lane == 4 && chunks_per_block == 2)
      hipLaunchKernelSynchronous(triad_kernel<4, 2, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, stop_ev,
          d_a, d_b, d_c);
    else if(elements_per_lane == 2 && chunks_per_block == 2)
      hipLaunchKernelSynchronous(triad_kernel<2, 2, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, stop_ev,
          d_a, d_b, d_c);
    else if(elements_per_lane == 4 && chunks_per_block == 4)
      hipLaunchKernelSynchronous(triad_kernel<4, 4, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, stop_ev,
          d_a, d_b, d_c);
    else if(elements_per_lane == 2 && chunks_per_block == 4)
      hipLaunchKernelSynchronous(triad_kernel<2, 4, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, stop_ev,
          d_a, d_b, d_c);
    else
      hipLaunchKernelSynchronous(triad_kernel<4, 2, T>,
          dim3(block_cnt), dim3(tb_size), nullptr, stop_ev,
          d_a, d_b, d_c);
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

template <unsigned int elements_per_lane, unsigned int chunks_per_block, typename T, unsigned int tb_size>
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
      tmp += load(a[gidx + i * dx + j]) * load(b[gidx + i * dx + j]);
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
  store(tb_sum[0], sum[blockIdx.x]);
}

template <class T>
T HIPStream<T>::dot()
{
  // Support for dwords_per_lane = 4 & chunks_per_block = 1/2/4
  if(elements_per_lane == 4 && chunks_per_block == 1) {
    if(tb_size == 1024) {
      hipLaunchKernelSynchronous(dot_kernel<4, 1, T, 1024>,
          dim3(block_cnt), dim3(tb_size), nullptr, coherent_ev,
          d_a, d_b, sums);
    }
    else if(tb_size == 512) {
      hipLaunchKernelSynchronous(dot_kernel<4, 1, T, 512>,
          dim3(block_cnt), dim3(tb_size), nullptr, coherent_ev,
          d_a, d_b, sums);
    }
  }
  else if(elements_per_lane == 2 && chunks_per_block == 1) {
    if(tb_size == 1024) {
      hipLaunchKernelSynchronous(dot_kernel<2, 1, T, 1024>,
          dim3(block_cnt), dim3(tb_size), nullptr, coherent_ev,
          d_a, d_b, sums);
    }
    else if(tb_size == 512) {
      hipLaunchKernelSynchronous(dot_kernel<2, 1, T, 512>,
          dim3(block_cnt), dim3(tb_size), nullptr, coherent_ev,
          d_a, d_b, sums);
    }
  }
  else if(elements_per_lane == 4 && chunks_per_block == 2) {
    if(tb_size == 1024) {
      hipLaunchKernelSynchronous(dot_kernel<4, 2, T, 1024>,
          dim3(block_cnt), dim3(tb_size), nullptr, coherent_ev,
          d_a, d_b, sums);
    }
    else if(tb_size == 512) {
      hipLaunchKernelSynchronous(dot_kernel<4, 2, T, 512>,
          dim3(block_cnt), dim3(tb_size), nullptr, coherent_ev,
          d_a, d_b, sums);
    }
  }
  else if(elements_per_lane == 2 && chunks_per_block == 2) {
    if(tb_size == 1024) {
      hipLaunchKernelSynchronous(dot_kernel<2, 2, T, 1024>,
          dim3(block_cnt), dim3(tb_size), nullptr, coherent_ev,
          d_a, d_b, sums);
    }
    else if(tb_size == 512) {
      hipLaunchKernelSynchronous(dot_kernel<2, 2, T, 512>,
          dim3(block_cnt), dim3(tb_size), nullptr, coherent_ev,
          d_a, d_b, sums);
    }
  }
  else if(elements_per_lane == 4 && chunks_per_block == 4) {
    if(tb_size == 1024) {
      hipLaunchKernelSynchronous(dot_kernel<4, 4, T, 1024>,
          dim3(block_cnt), dim3(tb_size), nullptr, coherent_ev,
          d_a, d_b, sums);
    }
    else if(tb_size == 512) {
      hipLaunchKernelSynchronous(dot_kernel<4, 4, T, 512>,
          dim3(block_cnt), dim3(tb_size), nullptr, coherent_ev,
          d_a, d_b, sums);
    }
  }
  else if(elements_per_lane == 2 && chunks_per_block == 4) {
    if(tb_size == 1024) {
      hipLaunchKernelSynchronous(dot_kernel<2, 4, T, 1024>,
          dim3(block_cnt), dim3(tb_size), nullptr, coherent_ev,
          d_a, d_b, sums);
    }
    else if(tb_size == 512) {
      hipLaunchKernelSynchronous(dot_kernel<2, 4, T, 512>,
          dim3(block_cnt), dim3(tb_size), nullptr, coherent_ev,
          d_a, d_b, sums);
    }
  }
  else
    hipLaunchKernelSynchronous(dot_kernel<4, 2, T, 1024>,
        dim3(block_cnt), dim3(tb_size), nullptr, coherent_ev,
        d_a, d_b, sums);

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
