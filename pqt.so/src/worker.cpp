/********************************************************************************
 *
 * Copyright (c) 2018 ROCm Developer Tools
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
#include "worker.h"

#ifdef __cplusplus
extern "C" {
#endif
#include <pci/pci.h>
#include <linux/pci.h>
#ifdef __cplusplus
}
#endif

#include <chrono>
#include <map>
#include <string>
#include <algorithm>
#include <iostream>
#include <mutex>

#include "rvs_module.h"
#include "pci_caps.h"
#include "gpu_util.h"
#include "rvsloglp.h"
#include "rvshsa.h"


pqtworker::pqtworker() {}
pqtworker::~pqtworker() {}

/**
 * @brief Thread function
 *
 * Loops while brun == TRUE and performs polled monitoring avery 1msec.
 *
 * */
void pqtworker::run() {
  while(brun) {
    do_transfer();
    std::this_thread::yield();
  }
  log("pqt worker thread has finished", rvs::logdebug);
}

/**
 * @brief Stop processing
 *
 * Sets brun member to FALSE thus signaling end of processing.
 * Then it waits for std::thread to exit before returning.
 *
 * */
void pqtworker::stop() {
  log("pqt in pqtworker::stop()", rvs::logdebug);

  brun = false;

  // wait a bit to make sure thread has exited
  try {
    if (t.joinable())
      t.join();
  }
  catch(...) {
  }
}

/**
 * @brief Init worker object and set transfer parameters
 *
 * @param Src source NUMA node
 * @param Dst destination NUMA node
 * @param Bidirect 'true' for bidirectional transfer
 * @return 0 - if successfull, non-zero otherwise
 *
 * */
int pqtworker::initialize(int Src, int Dst, bool Bidirect) {
  src_node = Src;
  dst_node = Dst;
  bidirect = Bidirect;
  pHsa = rvs::hsa::Get();

  running_size = 0;
  running_duration = 0;

  total_size = 0;
  total_duration = 0;

  return 0;
}

/**
 * @brief Executes data transfer
 *
 * Based on transfer parameters, initiates and performs one way or
 * bidirectional data transfer. Resulting measurements are compounded in running
 * totals for periodical printout during the test.
 * @return 0 - if successfull, non-zero otherwise
 *
 * */
int pqtworker::do_transfer() {
  double duration;
  int sts;

  for (size_t i = 0; i < pHsa->size_list.size(); i++) {
    current_size = pHsa->size_list[i];
    sts = pHsa->SendTraffic(src_node, dst_node, current_size,
                            bidirect, &duration);

    if (sts) {
      std::cerr << "RVS-PQT: internal error, src: " << src_node
                << "   dst: " << dst_node
                << "   current size: " << current_size << std::endl;
      return sts;
    }

    {
      std::lock_guard<std::mutex> lk(cntmutex);
      running_size += current_size;
      running_duration += duration;
    }

    /*
    std::string msg = "pqt packet size: " + std::to_string(phsa->size_list[i])
      + "   throughput: "
      + std::to_string(phsa->size_list[i]/duration/(1024*1024*1024)*2);
    rvs::lp::Log(msg, rvs::logresults);
    */
  }

  return 0;
}

/**
 * @brief Get running cumulatives for data trnasferred and time ellapsed
 *
 * @param Src [out] source NUMA node
 * @param Dst [out] destination NUMA node
 * @param Bidirect [out] 'true' for bidirectional transfer
 * @param Size [out] cumulative size of transferred data in this sampling
 * interval (in bytes)
 * @param Duration [out] cumulative duration of transfers in this sampling
 * interval (in seconds)
 *
 * */
void pqtworker::get_running_data(int*    Src,  int*    Dst,     bool* Bidirect,
                             size_t* Size, double* Duration) {
  // lock data until totalling has finished
  std::lock_guard<std::mutex> lk(cntmutex);

  // update total
  total_size += running_size;
  total_duration += running_duration;

  *Src = src_node;
  *Dst = dst_node;
  *Bidirect = bidirect;
  *Size = running_size;
  *Duration = running_duration;

  // reset running totas
  running_size = 0;
  running_duration = 0;
}

/**
 * @brief Get final cumulatives for data trnasferred and time ellapsed
 *
 * @param Src [out] source NUMA node
 * @param Dst [out] destination NUMA node
 * @param Bidirect [out] 'true' for bidirectional transfer
 * @param Size [out] cumulative size of transferred data in
 * this test (in bytes)
 * @param Duration [out] cumulative duration of transfers in
 * this test (in seconds)
 *
 * */
void pqtworker::get_final_data(int*    Src,  int*    Dst,     bool* Bidirect,
                           size_t* Size, double* Duration) {
  // lock data until totalling has finished
  std::lock_guard<std::mutex> lk(cntmutex);

  // update total
  total_size += running_size;
  total_duration += running_duration;

  *Src = src_node;
  *Dst = dst_node;
  *Bidirect = bidirect;
  *Size = total_size;
  *Duration = total_duration;

  // reset running totas
  running_size = 0;
  running_duration = 0;

  // reset final toral
  total_size = 0;
  total_duration = 0;
}
