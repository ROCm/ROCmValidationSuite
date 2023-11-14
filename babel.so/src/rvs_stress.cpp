// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <limits>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <cstring>
#include <mutex>

#define VERSION_STRING "3.4"

#include "include/rvs_memworker.h"
#include "include/Stream.h"
#include "include/HIPStream.h"
#include "include/rvsloglp.h"

// Default size of 2^25
std::string csv_separator = ",";
static bool triad_only = false;

template <typename T>
void check_solution(const unsigned int ntimes, std::vector<T>& a, std::vector<T>& b, std::vector<T>& c, T& sum, uint64_t);

template <typename T>
void run_stress(std::pair<int, uint16_t> device, int num_times, int ARRAY_SIZE, bool output_as_csv, bool mibibytes, int subtest);

template <typename T>
void run_triad(std::pair<int, uint16_t> device, int num_times, int ARRAY_SIZE, bool output_as_csv, bool mibibytes, int subtest);

void parseArguments(int argc, char *argv[]);

void run_babel(std::pair<int, uint16_t> device, int num_times, int array_size, bool output_csv, bool mibibytes, int test_type, int subtest) {

    switch(test_type) {
      case FLOAT_TEST:
        run_stress<float>(device, num_times, array_size, output_csv, mibibytes, subtest);
        break;

      case DOUBLE_TEST:
        run_stress<double>(device, num_times, array_size, output_csv, mibibytes, subtest);
        break;

      case TRAID_FLOAT:
        run_triad<float>(device, num_times, array_size, output_csv, mibibytes, subtest);
        break;

      case TRIAD_DOUBLE:
        run_triad<double>(device, num_times, array_size, output_csv, mibibytes, subtest);
        break;

      default:
        std::cout << "\n specify a valid testnumber";
        break;
  }
}

template <typename T>
void run_stress(std::pair<int, uint16_t> device, int num_times, int ARRAY_SIZE, bool output_as_csv, bool mibibytes, int subtest)
{
  std::string   msg;
  std::streamsize ss = std::cout.precision();

  if (!output_as_csv)
  {
    std::cout << "Running kernels " << num_times << " times" << std::endl;


    if (sizeof(T) == sizeof(float)) 
      std::cout << "Precision: float" << std::endl;
    else
      std::cout << "Precision: double" << std::endl;

    if (mibibytes)
    {
      // MiB = 2^20
      std::cout << std::setprecision(1) << std::fixed
                << "Array size: " << ARRAY_SIZE*sizeof(T)*pow(2.0, -20.0) << " MiB"
                << " (=" << ARRAY_SIZE*sizeof(T)*pow(2.0, -30.0) << " GiB)" << std::endl;
      std::cout << "Total size: " << 3.0*ARRAY_SIZE*sizeof(T)*pow(2.0, -20.0) << " MiB"
                << " (=" << 3.0*ARRAY_SIZE*sizeof(T)*pow(2.0, -30.0) << " GiB)" << std::endl;
    }
    else
    {
      // MB = 10^6
      std::cout << std::setprecision(1) << std::fixed
                << "Array size: " << ARRAY_SIZE*sizeof(T)*1.0E-6 << " MB"
                << " (=" << ARRAY_SIZE*sizeof(T)*1.0E-9 << " GB)" << std::endl;
      std::cout << "Total size: " << 3.0*ARRAY_SIZE*sizeof(T)*1.0E-6 << " MB"
                << " (=" << 3.0*ARRAY_SIZE*sizeof(T)*1.0E-9 << " GB)" << std::endl;
    }
    std::cout.precision(ss);

  }

  // Create host vectors
  std::vector<T> a(ARRAY_SIZE);
  std::vector<T> b(ARRAY_SIZE);
  std::vector<T> c(ARRAY_SIZE);

  // Result of the Dot kernel
  T sum;

  Stream<T> *stream;

  // Use the HIP implementation
  stream = new HIPStream<T>(ARRAY_SIZE, device.first);

  stream->init_arrays(startA, startB, startC);

  // List of times
  std::vector<std::vector<double>> timings(5);

  // Declare timers
  std::chrono::high_resolution_clock::time_point t1, t2;

  // Main loop
  for (unsigned int k = 0; k < num_times; k++)
  {
    // Execute Copy
    t1 = std::chrono::high_resolution_clock::now();
    stream->copy();
    t2 = std::chrono::high_resolution_clock::now();
    timings[0].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());

    // Execute Mul
    t1 = std::chrono::high_resolution_clock::now();
    stream->mul();
    t2 = std::chrono::high_resolution_clock::now();
    timings[1].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());

    // Execute Add
    t1 = std::chrono::high_resolution_clock::now();
    stream->add();
    t2 = std::chrono::high_resolution_clock::now();
    timings[2].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());

    // Execute Triad
    t1 = std::chrono::high_resolution_clock::now();
    stream->triad();
    t2 = std::chrono::high_resolution_clock::now();
    timings[3].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());

    // Execute Dot
    t1 = std::chrono::high_resolution_clock::now();
    sum = stream->dot();
    t2 = std::chrono::high_resolution_clock::now();
    timings[4].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());

  }

  // Check solutions
  stream->read_arrays(a, b, c);
  check_solution<T>(num_times, a, b, c, sum, ARRAY_SIZE);

  if (output_as_csv)
  {
    std::cout
      << "gpu_id" << csv_separator
      << "function" << csv_separator
      << "num_times" << csv_separator
      << "n_elements" << csv_separator
      << "sizeof" << csv_separator
      << ((mibibytes) ? "max_mibytes_per_sec" : "max_mbytes_per_sec") << csv_separator
      << "min_runtime" << csv_separator
      << "max_runtime" << csv_separator
      << "avg_runtime" << std::endl;
  }
  else
  {
    std::cout
      << "------------------------------------------------------------------------" << std::endl
      << std::left << std::setw(12) << "GPU Id"
      << std::left << std::setw(12) << "Function"
      << std::left << std::setw(12) << ((mibibytes) ? "MiBytes/sec" : "MBytes/sec")
      << std::left << std::setw(12) << "Min (sec)"
      << std::left << std::setw(12) << "Max"
      << std::left << std::setw(12) << "Average"
      << std::endl
      << std::fixed;
  }


  std::string labels[5] = {"Copy", "Mul", "Add", "Triad", "Dot"};
  size_t sizes[5] = {
    2 * sizeof(T) * ARRAY_SIZE,
    2 * sizeof(T) * ARRAY_SIZE,
    3 * sizeof(T) * ARRAY_SIZE,
    3 * sizeof(T) * ARRAY_SIZE,
    2 * sizeof(T) * ARRAY_SIZE
  };

  for (int i = 0; i < subtest; i++)
  {
    // Get min/max; ignore the first result
    auto minmax = std::minmax_element(timings[i].begin()+1, timings[i].end());

    // Calculate average; ignore the first result
    double average = std::accumulate(timings[i].begin()+1, timings[i].end(), 0.0) / (double)(num_times - 1);
    // Display results
    if (output_as_csv)
    {
      std::cout
        << device.second << csv_separator
        << labels[i] << csv_separator
        << num_times << csv_separator
        << ARRAY_SIZE << csv_separator
        << sizeof(T) << csv_separator
        << ((mibibytes) ? pow(2.0, -20.0) : 1.0E-6) * sizes[i] / (*minmax.first) << csv_separator
        << *minmax.first << csv_separator
        << *minmax.second << csv_separator
        << average
        << std::endl;
    }
    else
    {
      std::cout
        << std::left << std::setw(12) << device.second
        << std::left << std::setw(12) << labels[i]
        << std::left << std::setw(12) << std::setprecision(3) << 
          ((mibibytes) ? pow(2.0, -20.0) : 1.0E-6) * sizes[i] / (*minmax.first)
        << std::left << std::setw(12) << std::setprecision(5) << *minmax.first
        << std::left << std::setw(12) << std::setprecision(5) << *minmax.second
        << std::left << std::setw(12) << std::setprecision(5) << average
        << std::endl;
    }
  }

  delete stream;

}

template <typename T>
void run_triad(std::pair<int, uint16_t> device, int num_times, int ARRAY_SIZE, bool output_as_csv, bool mibibytes, int subtest)
{
  std::string msg;

  triad_only = true;

  if (!output_as_csv)
  {
    std::cout << "Running triad " << num_times << " times" << std::endl;
    std::cout << "Number of elements: " << ARRAY_SIZE << std::endl;

    if (sizeof(T) == sizeof(float))
      std::cout << "Precision: float" << std::endl;
    else
      std::cout << "Precision: double" << std::endl;

    std::streamsize ss = std::cout.precision();
    if (mibibytes)
    {
      std::cout << std::setprecision(1) << std::fixed
        << "Array size: " << ARRAY_SIZE*sizeof(T)*pow(2.0, -10.0) << " KiB"
        << " (=" << ARRAY_SIZE*sizeof(T)*pow(2.0, -20.0) << " MiB)" << std::endl;
      std::cout << "Total size: " << 3.0*ARRAY_SIZE*sizeof(T)*pow(2.0, -10.0) << " KiB"
        << " (=" << 3.0*ARRAY_SIZE*sizeof(T)*pow(2.0, -20.0) << " MiB)" << std::endl;
    }
    else
    {
      std::cout << std::setprecision(1) << std::fixed
        << "Array size: " << ARRAY_SIZE*sizeof(T)*1.0E-3 << " KB"
        << " (=" << ARRAY_SIZE*sizeof(T)*1.0E-6 << " MB)" << std::endl;
      std::cout << "Total size: " << 3.0*ARRAY_SIZE*sizeof(T)*1.0E-3 << " KB"
        << " (=" << 3.0*ARRAY_SIZE*sizeof(T)*1.0E-6 << " MB)" << std::endl;
    }
    std::cout.precision(ss);
  }

  // Create host vectors
  std::vector<T> a(ARRAY_SIZE);
  std::vector<T> b(ARRAY_SIZE);
  std::vector<T> c(ARRAY_SIZE);

  Stream<T> *stream;

  // Use the HIP implementation
  stream = new HIPStream<T>(ARRAY_SIZE, device.first);

  stream->init_arrays(startA, startB, startC);

  // Declare timers
  std::chrono::high_resolution_clock::time_point t1, t2;

  // Run triad in loop
  t1 = std::chrono::high_resolution_clock::now();
  for (unsigned int k = 0; k < num_times; k++)
  {
    stream->triad();
  }
  t2 = std::chrono::high_resolution_clock::now();

  double runtime = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

  // Check solutions
  T sum = 0.0;
  stream->read_arrays(a, b, c);
  check_solution<T>(num_times, a, b, c, sum, ARRAY_SIZE);

  // Display timing results
  double total_bytes = 3 * sizeof(T) * ARRAY_SIZE * num_times;
  double bandwidth = ((mibibytes) ? pow(2.0, -30.0) : 1.0E-9) * (total_bytes / runtime);

  if (output_as_csv)
  {
    std::cout
      << "gpu_id" << csv_separator
      << "function" << csv_separator
      << "num_times" << csv_separator
      << "n_elements" << csv_separator
      << "sizeof" << csv_separator
      << ((mibibytes) ? "gibytes_per_sec" : "gbytes_per_sec") << csv_separator
      << "runtime"
      << std::endl;
    std::cout
      << device.second << csv_separator
      << "Triad" << csv_separator
      << num_times << csv_separator
      << ARRAY_SIZE << csv_separator
      << sizeof(T) << csv_separator
      << bandwidth << csv_separator
      << runtime
      << std::endl;
  }
  else
  {
    std::cout
      << "--------------------------------"
      << std::endl << std::fixed
      << "GPU Id: " << std::left << device.second << std::endl
      << "Runtime (seconds): " << std::left << std::setprecision(5)
      << runtime << std::endl
      << "Bandwidth (" << ((mibibytes) ? "GiB/s" : "GB/s") << "):  "
      << std::left << std::setprecision(3)
      << bandwidth << std::endl;
  }


  delete stream;
}

template <typename T>
void check_solution(const unsigned int ntimes, std::vector<T>& a, std::vector<T>& b, std::vector<T>& c, T& sum, uint64_t ARRAY_SIZE)
{
  // Generate correct solution
  T goldA = startA;
  T goldB = startB;
  T goldC = startC;
  T goldSum = 0.0;
  std::string  msg;

  const T scalar = startScalar;

  for (unsigned int i = 0; i < ntimes; i++)
  {
    // Do STREAM!
    if (!triad_only)
    {
      goldC = goldA;
      goldB = scalar * goldC;
      goldC = goldA + goldB;
    }
    goldA = goldB + scalar * goldC;
  }

  // Do the reduction
  goldSum = goldA * goldB * ARRAY_SIZE;

  // Calculate the average error
  double errA = std::accumulate(a.begin(), a.end(), 0.0, [&](double sum, const T val){ return sum + fabs(val - goldA); });
  errA /= a.size();
  double errB = std::accumulate(b.begin(), b.end(), 0.0, [&](double sum, const T val){ return sum + fabs(val - goldB); });
  errB /= b.size();
  double errC = std::accumulate(c.begin(), c.end(), 0.0, [&](double sum, const T val){ return sum + fabs(val - goldC); });
  errC /= c.size();
  double errSum = fabs(sum - goldSum);

  double epsi = std::numeric_limits<T>::epsilon() * 100.0;

  if (errA > epsi)
    std::cerr
      << "Validation failed on a[]. Average error " << errA
      << std::endl;
  if (errB > epsi)
    std::cerr
      << "Validation failed on b[]. Average error " << errB
      << std::endl;
  if (errC > epsi)
    std::cerr
      << "Validation failed on c[]. Average error " << errC
      << std::endl;
  if (!triad_only && errSum > 1.0E-8)
    std::cerr
      << "Validation failed on sum. Error " << errSum
      << std::endl << std::setprecision(15)
      << "Sum was " << sum << " but should be " << goldSum
      << std::endl;
}
