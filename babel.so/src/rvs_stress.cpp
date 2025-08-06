// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include <iostream>
#include <vector>
#include <vector>
#include <string>
#include <numeric>
#include <cmath>
#include <limits>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <cstring>
#include <mutex>
#include <sstream>

#define VERSION_STRING "3.4"
#include "include/rvs_util.h"
#include "include/rvs_memworker.h"
#include "include/Stream.h"
#include "include/HIPStream.h"
#include "include/rvsloglp.h"

// Default size of 2^25
std::string csv_separator = ",";
static bool triad_only = false;

 bool event_timing = false;
 std::string module_name{"babel"};


template <typename T>
void check_solution(const unsigned int ntimes, std::vector<T>& a, std::vector<T>& b, std::vector<T>& c, T& sum, uint64_t);

template <typename T>
void run_stress(std::pair<int, uint16_t> device, int num_times, int ARRAY_SIZE, bool output_as_csv, bool mibibytes, int subtest,
    uint16_t dwords_per_lane, uint16_t chunks_per_block, uint16_t tb_size, bool json, std::string action, int rwtest);

template <typename T>
void run_triad(std::pair<int, uint16_t> device, int num_times, int ARRAY_SIZE, bool output_as_csv, bool mibibytes, int subtest,
    uint16_t dwords_per_lane, uint16_t chunks_per_block, uint16_t tb_size, bool json, std::string action, int rwtest);

void parseArguments(int argc, char *argv[]);

bool run_babel(std::pair<int, uint16_t> device, int num_times, int array_size, bool output_csv, bool mibibytes, int test_type, int subtest,
    uint16_t dwords_per_lane, uint16_t chunks_per_block, uint16_t tb_size, bool json, std::string action, int rwtest) {

  switch(test_type) {
    case FLOAT_TEST:
      run_stress<float>(device, num_times, array_size, output_csv, mibibytes, subtest, dwords_per_lane, chunks_per_block, tb_size,
          json, action, rwtest);
      break;

    case DOUBLE_TEST:
      run_stress<double>(device, num_times, array_size, output_csv, mibibytes, subtest, dwords_per_lane, chunks_per_block, tb_size,
          json, action, rwtest);
      break;

    case TRAID_FLOAT:
      run_triad<float>(device, num_times, array_size, output_csv, mibibytes, subtest, dwords_per_lane, chunks_per_block, tb_size,
          json, action, rwtest);
      break;

    case TRIAD_DOUBLE:
      run_triad<double>(device, num_times, array_size, output_csv, mibibytes, subtest, dwords_per_lane, chunks_per_block, tb_size,
          json, action, rwtest);
      break;

    default:
      std::cout << "\n specify a valid testnumber";
      break;
  }
}

template <typename T>
void run_stress(std::pair<int, uint16_t> device, int num_times, int ARRAY_SIZE, bool output_as_csv, bool mibibytes, int subtest,
    uint16_t dwords_per_lane, uint16_t chunks_per_block, uint16_t tb_size, bool json, std::string action, int rwtest)
{
  std::string   msg;
  std::streamsize ss = std::cout.precision();
  std::stringstream sstr;
  auto desc = action_descriptor{action, module_name, device.second};
  if (!output_as_csv)
  {
    msg = "Running kernels " + std::to_string(num_times) + " times, " ;


    if (sizeof(T) == sizeof(float)) 
      msg += "Precision: float";
    else
      msg += "Precision: double";

    rvs::lp::Log(msg, rvs::logresults);
    if (mibibytes)
    {
      // MiB = 2^20
      sstr << std::setprecision(1) << std::fixed
                << "Array size: " << ARRAY_SIZE*sizeof(T)*pow(2.0, -20.0) << " MiB"
                << " (=" << ARRAY_SIZE*sizeof(T)*pow(2.0, -30.0) << " GiB), ";
      sstr << "Total size: " << 3.0*ARRAY_SIZE*sizeof(T)*pow(2.0, -20.0) << " MiB"
                << " (=" << 3.0*ARRAY_SIZE*sizeof(T)*pow(2.0, -30.0) << " GiB)" << std::endl;
    }
    else
    {
      // MB = 10^6
      sstr << std::setprecision(1) << std::fixed
                << "Array size: " << ARRAY_SIZE*sizeof(T)*1.0E-6 << " MB"
                << " (=" << ARRAY_SIZE*sizeof(T)*1.0E-9 << " GB), ";
      sstr << "Total size: " << 3.0*ARRAY_SIZE*sizeof(T)*1.0E-6 << " MB"
                << " (=" << 3.0*ARRAY_SIZE*sizeof(T)*1.0E-9 << " GB)" << std::endl;
    }
    rvs::lp::Log(sstr.str(), rvs::logresults);
    std::cout.precision(ss);

  }

  //json
  if (json){
    std::string scale = mibibytes ? "MiB" : "MB";
    auto arr_size = mibibytes ? ARRAY_SIZE*sizeof(T)*pow(2.0, -20.0) :
	    ARRAY_SIZE*sizeof(T)*1.0E-6;
    auto total_size = mibibytes ? 3.0*ARRAY_SIZE*sizeof(T)*pow(2.0, -20.0) :
	    3.0*ARRAY_SIZE*sizeof(T)*1.0E-6;
    log_to_json(desc, rvs::logresults,"Array size", std::to_string(arr_size),
	      "Total size", std::to_string(total_size),
	      "Iterations", std::to_string(num_times) );
  }

  // Create host vectors
  std::vector<T> a(ARRAY_SIZE);
  std::vector<T> b(ARRAY_SIZE);
  std::vector<T> c(ARRAY_SIZE);

  // Result of the Dot kernel
  T sum;

  Stream<T> *stream;

  // Use the HIP implementation
  stream = new HIPStream<T>(ARRAY_SIZE, event_timing, device.first, dwords_per_lane, chunks_per_block, tb_size);

  stream->init_arrays(startA, startB, startC);

  // List of times
  std::vector<std::vector<double>> rwtimings(2);
  std::vector<std::vector<double>> timings(5);

  // Declare timers
  std::chrono::high_resolution_clock::time_point t1, t2;

  for (unsigned int k = 0; k < num_times; k++)
  {
    // Execute Read
    t1 = std::chrono::high_resolution_clock::now();
    stream->read();
    t2 = std::chrono::high_resolution_clock::now();
    rwtimings[0].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());

    // Execute Write
    t1 = std::chrono::high_resolution_clock::now();
    stream->write();
    t2 = std::chrono::high_resolution_clock::now();
    rwtimings[1].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());
  }

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
  sstr.str( std::string() );
  sstr.clear();
  if (output_as_csv)
  {
     sstr  << "gpu_id" << csv_separator
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
      sstr << "\n------------------------------------------------------------------------" << std::endl
      << std::left << std::setw(12) << "GPU Id"
      << std::left << std::setw(12) << "Function"
      << std::left << std::setw(12) << ((mibibytes) ? "MiBytes/sec" : "MBytes/sec")
      << std::left << std::setw(12) << "Min (sec)"
      << std::left << std::setw(12) << "Max"
      << std::left << std::setw(12) << "Average"
      << std::endl
      << "------------------------------------------------------------------------" << std::endl
      << std::fixed;
  }

  std::string rwlabels[2] = {"Read", "Write"};
  size_t rwsizes[2] = {
    sizeof(T) * ARRAY_SIZE,
    sizeof(T) * ARRAY_SIZE
  };

  for (int i = 0; i < rwtest; i++)
  {
    // Get min/max; ignore the first result
    auto minmax = std::minmax_element(rwtimings[i].begin()+1, rwtimings[i].end());

    // Calculate average; ignore the first result
    double average = std::accumulate(rwtimings[i].begin()+1, rwtimings[i].end(), 0.0) / (double)(num_times - 1);
    // Display results
    if (output_as_csv)
    {
      sstr
        << device.second << csv_separator
        << rwlabels[i] << csv_separator
        << num_times << csv_separator
        << ARRAY_SIZE << csv_separator
        << sizeof(T) << csv_separator
        << ((mibibytes) ? pow(2.0, -20.0) : 1.0E-6) * rwsizes[i] / (*minmax.first) << csv_separator
        << *minmax.first << csv_separator
        << *minmax.second << csv_separator
        << average
        << std::endl;
    }
    else
    {
      sstr
        << std::left << std::setw(12) << device.second
        << std::left << std::setw(12) << rwlabels[i]
        << std::left << std::setw(12) << std::setprecision(3) <<
          ((mibibytes) ? pow(2.0, -20.0) : 1.0E-6) * rwsizes[i] / (*minmax.first)
        << std::left << std::setw(12) << std::setprecision(5) << *minmax.first
        << std::left << std::setw(12) << std::setprecision(5) << *minmax.second
        << std::left << std::setw(12) << std::setprecision(5) << average
        << std::endl;
    }
    if (json){
      log_to_json(desc, rvs::logresults, "Function",std::string(rwlabels[i]),
		      "MBytes/sec", (mibibytes) ?
		      std::to_string(pow(2.0, -20.0)) : std::to_string((1.0E-6) * rwsizes[i] / (*minmax.first)),
		      "Min(s)",std::to_string( *minmax.first),
		      "Max(s)", std::to_string(*minmax.second),
		      "Average(s)", std::to_string(average),
		      "pass", "true");
    }
  }

  //rvs::lp::Log(sstr.str(), rvs::logresults); 
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
    //sstr.str( std::string() );
    //sstr.clear();
    // Calculate average; ignore the first result
    double average = std::accumulate(timings[i].begin()+1, timings[i].end(), 0.0) / (double)(num_times - 1);
    // Display results
    if (output_as_csv)
    {
      sstr
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
      sstr
        << std::left << std::setw(12) << device.second
        << std::left << std::setw(12) << labels[i]
        << std::left << std::setw(12) << std::setprecision(3) << 
          ((mibibytes) ? pow(2.0, -20.0) : 1.0E-6) * sizes[i] / (*minmax.first)
        << std::left << std::setw(12) << std::setprecision(5) << *minmax.first
        << std::left << std::setw(12) << std::setprecision(5) << *minmax.second
        << std::left << std::setw(12) << std::setprecision(5) << average
        << std::endl;
    }
    if (json){
      log_to_json(desc, rvs::logresults, "Function",std::string(labels[i]),
		      "MBytes/sec", (mibibytes) ? 
		      std::to_string(pow(2.0, -20.0)) : std::to_string((1.0E-6) * sizes[i] / (*minmax.first)),
		      "Min(s)",std::to_string( *minmax.first), 
		      "Max(s)", std::to_string(*minmax.second),
		      "Average(s)", std::to_string(average),
		      "pass", "true");
    } 
  }
  sstr
    << "------------------------------------------------------------------------" << std::endl;
  rvs::lp::Log(sstr.str(), rvs::logresults);
  delete stream;

}

template <typename T>
void run_triad(std::pair<int, uint16_t> device, int num_times, int ARRAY_SIZE, bool output_as_csv, bool mibibytes, int subtest,
    uint16_t dwords_per_lane, uint16_t chunks_per_block, uint16_t tb_size, bool json, std::string action, int rwtest)
{
  std::string msg;
  auto desc = action_descriptor{action, module_name, device.second};
  triad_only = true;
  std::stringstream sstr;
  if (!output_as_csv)
  {
    msg = "Running triad " + std::to_string (num_times) + " times,";
    msg += "Number of elements: " + std::to_string(ARRAY_SIZE) + ", ";

    if (sizeof(T) == sizeof(float))
      msg += "Precision: float\n";
    else
      msg += "Precision: double\n" ;
    
    rvs::lp::Log(msg, rvs::loginfo);
    std::streamsize ss = std::cout.precision();
    if (mibibytes)
    {
      sstr << std::setprecision(1) << std::fixed
        << "Array size: " << ARRAY_SIZE*sizeof(T)*pow(2.0, -10.0) << " KiB"
        << " (=" << ARRAY_SIZE*sizeof(T)*pow(2.0, -20.0) << " MiB)" << std::endl;
      std::cout << "Total size: " << 3.0*ARRAY_SIZE*sizeof(T)*pow(2.0, -10.0) << " KiB"
        << " (=" << 3.0*ARRAY_SIZE*sizeof(T)*pow(2.0, -20.0) << " MiB)" << std::endl;
    }
    else
    {
      sstr << std::setprecision(1) << std::fixed
        << "Array size: " << ARRAY_SIZE*sizeof(T)*1.0E-3 << " KB"
        << " (=" << ARRAY_SIZE*sizeof(T)*1.0E-6 << " MB)" << std::endl;
      std::cout << "Total size: " << 3.0*ARRAY_SIZE*sizeof(T)*1.0E-3 << " KB"
        << " (=" << 3.0*ARRAY_SIZE*sizeof(T)*1.0E-6 << " MB)" << std::endl;
    }
    rvs::lp::Log(sstr.str(), rvs::logresults);
    if (json){
      std::string scale = mibibytes ? "MiB" : "MB";
      auto arr_size = mibibytes ? ARRAY_SIZE*sizeof(T)*pow(2.0, -20.0) :
	      ARRAY_SIZE*sizeof(T)*1.0E-6;
     auto total_size = mibibytes  ? 3.0*ARRAY_SIZE*sizeof(T)*pow(2.0, -20.0) :
	     3.0*ARRAY_SIZE*sizeof(T)*1.0E-6;
     log_to_json(desc, rvs::logresults,"Array size", std::to_string(arr_size),
              "Total size", std::to_string(total_size),
              "Iterations", std::to_string(num_times) );
    }
    std::cout.precision(ss);
  }
  sstr.str( std::string() );
  sstr.clear();
  // Create host vectors
  std::vector<T> a(ARRAY_SIZE);
  std::vector<T> b(ARRAY_SIZE);
  std::vector<T> c(ARRAY_SIZE);

  Stream<T> *stream;

  // Use the HIP implementation
  stream = new HIPStream<T>(ARRAY_SIZE, event_timing, device.first, dwords_per_lane, chunks_per_block, tb_size);

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
    sstr
      << "gpu_id" << csv_separator
      << "function" << csv_separator
      << "num_times" << csv_separator
      << "n_elements" << csv_separator
      << "sizeof" << csv_separator
      << ((mibibytes) ? "gibytes_per_sec" : "gbytes_per_sec") << csv_separator
      << "runtime"
      << std::endl
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
    sstr
      << "--------------------------------"
      << std::endl << std::fixed
      << "GPU Id: " << std::left << device.second << std::endl
      << "Runtime (seconds): " << std::left << std::setprecision(5)
      << runtime << std::endl
      << "Bandwidth (" << ((mibibytes) ? "GiB/s" : "GB/s") << "):  "
      << std::left << std::setprecision(3)
      << bandwidth << std::endl;
  }
   rvs::lp::Log(sstr.str(), rvs::logresults);
   if (json){
     std::string bw_field{"Bandwidth ("};
     bw_field +=(mibibytes) ? "GiB/s" : "GB/s";
     bw_field += ")";
     log_to_json(desc, rvs::logresults,
		     "GPU Id", std::to_string(device.second),
		     "Runtime (seconds)", std::to_string(runtime),
		     bw_field, std::to_string(bandwidth),
		     "pass", "true");
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
      rvs::lp::Log("Validation failed on a[]. Average error " + std::to_string(errA), rvs::logerror);
  if (errB > epsi)
      rvs::lp::Log("Validation failed on b[]. Average error " + std::to_string(errB),rvs::logerror);
  if (errC > epsi)
      rvs::lp::Log("Validation failed on c[]. Average error " + std::to_string(errC),rvs::logerror);
  if (!triad_only && errSum > 1.0E-8){
    std::stringstream sstr;
     sstr  << "Validation failed on sum. Error " << errSum
      << std::endl << std::setprecision(15)
      << "Sum was " << sum << " but should be " << goldSum
      << std::endl;
     rvs::lp::Log(sstr.str() ,rvs::logerror);
  }
}

