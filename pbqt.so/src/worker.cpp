/********************************************************************************
 *
 * Copyright (c) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "include/worker.h"

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

#include "include/rvs_module.h"
#include "include/pci_caps.h"
#include "include/gpu_util.h"
#include "include/rvsloglp.h"
#include "include/rvshsa.h"
#include "TransferBench.hpp"
#define MODULE_NAME "PBQT"


pbqtworker::pbqtworker() {
  // set to 'true' so that do_transfer() will also work
  // when parallel: false
  brun = true;
  a2a_mode = 0;
  a2a_direct = 1;
  a2a_local = 0;
  a2a_num_gpus = 0;
  use_remote_read = 0;
}
pbqtworker::~pbqtworker() {}

extern uint64_t time_diff(
                std::chrono::time_point<std::chrono::system_clock> t_end,
                std::chrono::time_point<std::chrono::system_clock> t_start);
extern uint64_t test_duration;
 
/**
 * @brief Thread function
 *
 * Loops while brun == TRUE and performs polled monitoring avery 1msec.
 *
 * */
void pbqtworker::run() {
  std::string msg;
  std::chrono::time_point<std::chrono::system_clock> pbqt_start_time;
  std::chrono::time_point<std::chrono::system_clock> pbqt_end_time;

  msg = "[" + action_name + "] pbqt thread " + std::to_string(src_node) + " "
  + std::to_string(dst_node) + " has started";
  rvs::lp::Log(msg, rvs::logdebug);

  brun = true;

  pbqt_start_time = std::chrono::system_clock::now();
  do {
      do_transfer();

      pbqt_end_time = std::chrono::system_clock::now();

      uint64_t test_time = time_diff(pbqt_end_time, pbqt_start_time) ;

      if(test_time >= test_duration) {
          break;
      }
   } while (brun);

  msg = "[" + action_name + "] pbqt thread " + std::to_string(src_node) + " "
  + std::to_string(dst_node) + " has finished";
  rvs::lp::Log(msg, rvs::logdebug);
}

/**
 * @brief Stop processing
 *
 * Sets brun member to FALSE thus signaling end of processing.
 * Then it waits for std::thread to exit before returning.
 *
 * */
void pbqtworker::stop() {
  std::string msg;

  msg = "[" + stop_action_name + "] pbqt transfer " + std::to_string(src_node)
      + " " + std::to_string(dst_node) + " in pbqtworker::stop()";
  rvs::lp::Log(msg, rvs::logtrace);

  brun = false;
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
int pbqtworker::initialize(uint16_t Src, uint16_t Dst, bool Bidirect) {
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
int pbqtworker::do_transfer() {
  double duration;
  int sts;
  unsigned int startsec;
  unsigned int startusec;
  unsigned int endsec;
  unsigned int endusec;
  std::string msg;

  msg = "[" + action_name + "] pbqt transfer " + std::to_string(src_node) + " "
    + std::to_string(dst_node) + " ";

  rvs::lp::get_ticks(&startsec, &startusec);

  if (transfer_method == "transferbench") {

    if (transferbench_test == "p2p") {

      // Configure TransferBench parameters
      TransferBench::ConfigOptions cfg;

      cfg.general.numIterations = hot_calls;
      cfg.general.numWarmups = warm_calls;

      TransferBench::MemType src_mem = TransferBench::MEM_GPU;
      TransferBench::MemType dst_mem = TransferBench::MEM_GPU;

      std::vector<TransferBench::Transfer> transfers(bidirect? 2 : 1);

      transfers[0].numBytes = block_size[0];

      transfers[0].srcs.push_back({src_mem,
          src_mem == TransferBench::MEM_GPU ? src_node - TransferBench::GetNumExecutors(TransferBench::EXE_CPU) : src_node});
      transfers[0].dsts.push_back({dst_mem,
          dst_mem == TransferBench::MEM_GPU ? dst_node - TransferBench::GetNumExecutors(TransferBench::EXE_CPU) : dst_node});

      transfers[0].exeDevice = {executor == "gpu" ? TransferBench::EXE_GPU_GFX : TransferBench::EXE_GPU_DMA,
        src_mem == TransferBench::MEM_GPU ? src_node - TransferBench::GetNumExecutors(TransferBench::EXE_CPU) : src_node};

      transfers[0].exeSubIndex = -1;

      transfers[0].numSubExecs = subexecutor;

      if (bidirect) {
        transfers[1].numBytes = block_size[0];

        transfers[1].srcs.push_back({dst_mem,
            dst_mem == TransferBench::MEM_GPU ? dst_node - TransferBench::GetNumExecutors(TransferBench::EXE_CPU) : dst_node});
        transfers[1].dsts.push_back({src_mem,
            src_mem == TransferBench::MEM_GPU ? src_node - TransferBench::GetNumExecutors(TransferBench::EXE_CPU) : src_node});

        transfers[1].exeDevice = {executor == "gpu" ? TransferBench::EXE_GPU_GFX : TransferBench::EXE_GPU_DMA,
          dst_mem == TransferBench::MEM_GPU ? dst_node - TransferBench::GetNumExecutors(TransferBench::EXE_CPU) : dst_node};

        transfers[1].exeSubIndex = -1;

        transfers[1].numSubExecs = subexecutor;
      }

      TransferBench::TestResults results;

      if (!TransferBench::RunTransfers(cfg, transfers, results)) {
        for (auto const& err : results.errResults)
          printf("%s\n", err.errMsg.c_str());
        exit(1);
      }

      {
        std::lock_guard<std::mutex> lk(cntmutex);

        for (size_t t = 0; t < results.tfrResults.size(); t++) {
          const auto& res = results.tfrResults[t];
          running_size += results.numTimedIterations * res.numBytes;
          running_duration += (res.avgDurationMsec/1000) * results.numTimedIterations;
        }
      }

    } else if (transferbench_test == "alltoall") {

      int numDetectedGpus = TransferBench::GetNumExecutors(TransferBench::EXE_GPU_GFX);
      int numGpus = (a2a_num_gpus > 0 && static_cast<int>(a2a_num_gpus) <= numDetectedGpus)
                    ? static_cast<int>(a2a_num_gpus) : numDetectedGpus;

      if (numGpus < 2) {
        msg += "alltoall requires at least 2 GPUs, detected: " + std::to_string(numDetectedGpus);
        rvs::lp::Err(msg, MODULE_NAME, action_name);
        return -1;
      }

      int numSrcs, numDsts;
      switch (a2a_mode) {
        case 1:  numSrcs = 1; numDsts = 0; break;
        case 2:  numSrcs = 0; numDsts = 1; break;
        default: numSrcs = 1; numDsts = 1; break;
      }

      TransferBench::ExeType exeType = (executor == "dma")
          ? TransferBench::EXE_GPU_DMA : TransferBench::EXE_GPU_GFX;
      TransferBench::MemType memType = TransferBench::MEM_GPU;

      std::vector<TransferBench::Transfer> transfers;

      for (int i = 0; i < numGpus; i++) {
        for (int j = 0; j < numGpus; j++) {

          if (i == j) {
            if (!a2a_local) continue;
          } else if (a2a_direct) {
            uint32_t linkType, hopCount;
            hipError_t hip_err = hipExtGetLinkTypeAndHopCount(i, j, &linkType, &hopCount);
            if (hip_err != hipSuccess || hopCount != 1) continue;
          }

          TransferBench::Transfer transfer;
          transfer.numBytes = block_size.size() > 0 ? block_size[0] : 0;
          for (int x = 0; x < numSrcs; x++)
            transfer.srcs.push_back({memType, i});
          if (numDsts)
            transfer.dsts.push_back({memType, j});
          for (int x = 1; x < numDsts; x++)
            transfer.dsts.push_back({memType, i});

          transfer.exeDevice = {exeType, (use_remote_read ? j : i)};
          transfer.exeSubIndex = -1;
          transfer.numSubExecs = subexecutor;

          transfers.push_back(transfer);
        }
      }

      if (transfers.empty()) {
        msg += "alltoall: no valid GPU pairs found";
        rvs::lp::Err(msg, MODULE_NAME, action_name);
        return -1;
      }

      TransferBench::ConfigOptions cfg;
      cfg.general.numIterations = hot_calls;
      cfg.general.numWarmups = warm_calls;

      TransferBench::TestResults results;

      if (!TransferBench::RunTransfers(cfg, transfers, results)) {
        for (auto const& err : results.errResults) {
          std::string errmsg = "[" + action_name + "] alltoall error: " + err.errMsg;
          rvs::lp::Err(errmsg, MODULE_NAME, action_name);
        }
        return -1;
      }

      {
        std::lock_guard<std::mutex> lk(cntmutex);

        double totalBandwidthGbps = 0.0;
        size_t totalBytes = 0;
        double maxDurationMsec = 0.0;

        for (size_t t = 0; t < results.tfrResults.size(); t++) {
          const auto& res = results.tfrResults[t];
          totalBandwidthGbps += res.avgBandwidthGbPerSec;
          totalBytes += results.numTimedIterations * res.numBytes;
          if (res.avgDurationMsec > maxDurationMsec)
            maxDurationMsec = res.avgDurationMsec;
        }

        running_size += totalBytes;
        running_duration += (maxDurationMsec / 1000.0) * results.numTimedIterations;
      }

      std::string a2a_msg = "[" + action_name + "] alltoall "
          + std::to_string(numGpus) + " GPUs, "
          + std::to_string(transfers.size()) + " transfers, "
          + "aggregate bandwidth: "
          + std::to_string(results.avgTotalBandwidthGbPerSec) + " GB/s";
      rvs::lp::Log(a2a_msg, rvs::loginfo);

    } else {
      msg += "unknown transferbench_test: " + transferbench_test;
      rvs::lp::Err(msg, MODULE_NAME, action_name);
      return -1;
    }

  }
  else {

    if (block_size.size() == 0) {
      block_size = pHsa->size_list;
    }

    for (size_t i = 0; brun && i < block_size.size(); i++) {
      current_size = block_size[i];
      sts = pHsa->SendTraffic(src_node, dst_node, current_size,
          bidirect, b2b, warm_calls, hot_calls, &duration);

      if (sts) {
        msg = "internal error, src: " + std::to_string(src_node)
          + "   dst: " + std::to_string(dst_node)
          + "   current size: " + std::to_string(current_size);
        rvs::lp::Err(msg, MODULE_NAME, action_name);
        return sts;
      }

      {
        std::lock_guard<std::mutex> lk(cntmutex);
        running_size += current_size;
        running_duration += duration;
      }
    }
  }
  rvs::lp::get_ticks(&endsec, &endusec);
  rvs::lp::Log(msg + "start", rvs::logdebug, startsec, startusec);
  rvs::lp::Log(msg + "finish", rvs::logdebug, endsec, endusec);

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
void pbqtworker::get_running_data(uint16_t* Src,  uint16_t* Dst, bool* Bidirect,
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
 * @param bReset [in] if 'true' set final totals to zero
 *
 * */
void pbqtworker::get_final_data(uint16_t* Src,  uint16_t* Dst, bool* Bidirect,
                           size_t* Size, double* Duration, bool bReset) {
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

  // reset final totals
  if (bReset) {
    total_size = 0;
    total_duration = 0;
  }
}
