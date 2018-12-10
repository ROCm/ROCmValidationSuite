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
#include "include/worker_b2b.h"

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

using std::string;
using std::vector;
using std::map;

pebbworker_b2b::pebbworker_b2b()
: pebbworker() {
}
pebbworker_b2b::~pebbworker_b2b() {}

/**
 * @brief Init worker object and set transfer parameters
 *
 * @param Src source NUMA node
 * @param Dst destination NUMA node
 * @param h2d 'true' for host to device transfer
 * @param d2h 'true' for device to host transfer
 * @param Size size of block used for transfer
 * @return 0 - if successfull, non-zero otherwise
 *
 * */
int pebbworker_b2b::initialize(uint16_t Src, uint16_t Dst,
                               bool h2d, bool d2h, size_t Size) {
  pebbworker::initialize(Src, Dst, h2d, d2h);

  b2b_block_size = Size;

  ctx_fwd.SrcAgentIx = pHsa->FindAgent(Src);
  ctx_fwd.SrcAgent = pHsa->agent_list[ctx_fwd.SrcAgentIx].agent;

  ctx_fwd.DstAgentIx = pHsa->FindAgent(Dst);
  ctx_fwd.DstAgent = pHsa->agent_list[ctx_fwd.DstAgentIx].agent;

  ctx_fwd.Sig.handle = 0;
  ctx_fwd.pSrcBuff = nullptr;
  ctx_fwd.pDstBuff = nullptr;

  ctx_rev.SrcAgentIx = ctx_fwd.DstAgentIx;
  ctx_rev.SrcAgent = ctx_fwd.DstAgent;

  ctx_rev.DstAgentIx = ctx_fwd.SrcAgentIx;
  ctx_rev.DstAgent = ctx_fwd.SrcAgent;
  ctx_rev.Sig.handle = 0;

  ctx_rev.pSrcBuff = nullptr;
  ctx_rev.pDstBuff = nullptr;

  return 0;
}

/**
 * @brief release all resources used in transfers
 */
void pebbworker_b2b::deinit() {
  RVSTRACE_
  // release fwd buffers if any
  if (ctx_fwd.pSrcBuff) {
    hsa_amd_memory_pool_free(ctx_fwd.pSrcBuff);
    ctx_fwd.pSrcBuff = nullptr;
  }

  RVSTRACE_
  if (ctx_fwd.pDstBuff) {
    hsa_amd_memory_pool_free(ctx_fwd.pDstBuff);
    ctx_fwd.pDstBuff = nullptr;
  }

  RVSTRACE_
  if (ctx_fwd.Sig.handle) {
    hsa_signal_destroy(ctx_fwd.Sig);
    ctx_fwd.Sig.handle = 0;
  }

  RVSTRACE_
  if (ctx_rev.pSrcBuff) {
    hsa_amd_memory_pool_free(ctx_rev.pSrcBuff);
    ctx_rev.pSrcBuff = nullptr;
  }

  RVSTRACE_
  if (ctx_rev.pDstBuff) {
    hsa_amd_memory_pool_free(ctx_rev.pDstBuff);
    ctx_rev.pDstBuff = nullptr;
  }

  RVSTRACE_
  if (ctx_rev.Sig.handle) {
    hsa_signal_destroy(ctx_rev.Sig);
    ctx_rev.Sig.handle = 0;
  }
  RVSTRACE_
}

/**
 * @brief Thread function
 *
 * Loops while brun == TRUE and performs polled monitoring avery 1msec.
 *
 * */
void pebbworker_b2b::run() {
  int sts;
  hsa_status_t status;

  RVSTRACE_

  // enable test
  brun = true;

  // allocate buffers and grant permissions for forward transfer
  if (prop_h2d) {
    sts = pHsa->Allocate(ctx_fwd.SrcAgentIx, ctx_fwd.DstAgentIx, b2b_block_size,
            &ctx_fwd.SrcPool, &ctx_fwd.pSrcBuff,
            &ctx_fwd.DstPool, &ctx_fwd.pDstBuff);
    if (sts) {
      RVSTRACE_
      deinit();
      return;
    }

    // Create a signal to wait on forward copy operation
    if (HSA_STATUS_SUCCESS !=
      (status = hsa_signal_create(1, 0, NULL, &ctx_fwd.Sig))) {
      rvs::hsa::print_hsa_status(__FILE__, __LINE__, __func__,
                "hsa_signal_create()", status);
      RVSTRACE_
      deinit();
      return;
    }
  }

  // allocate buffers and grant permissions for reverse transfer
  if (prop_d2h) {
    sts = pHsa->Allocate(ctx_rev.SrcAgentIx, ctx_rev.DstAgentIx, b2b_block_size,
            &ctx_rev.SrcPool, &ctx_rev.pSrcBuff,
            &ctx_rev.DstPool, &ctx_rev.pDstBuff);

    if (sts) {
      RVSTRACE_
      deinit();
      return;
    }

    // Create a signal to wait on reverse copy operation
    if (HSA_STATUS_SUCCESS !=
      (status = hsa_signal_create(1, 0, NULL, &ctx_rev.Sig))) {
      rvs::hsa::print_hsa_status(__FILE__, __LINE__, __func__,
                "hsa_signal_create()", status);
      RVSTRACE_
      deinit();
      return;
    }
  }


  while (brun) {
    // initiate forward transfer
    if (prop_h2d) {
      RVSTRACE_
      hsa_signal_store_relaxed(ctx_fwd.Sig, 1);
      if (HSA_STATUS_SUCCESS !=
        (status = hsa_amd_memory_async_copy(
                    ctx_fwd.pDstBuff, ctx_fwd.DstAgent,
                    ctx_fwd.pSrcBuff, ctx_fwd.SrcAgent,
                    b2b_block_size,
                    0, NULL, ctx_fwd.Sig))) {
        rvs::hsa::print_hsa_status(__FILE__, __LINE__, __func__,
                  "hsa_amd_memory_async_copy()",
                  status);
        break;
      }
    }

    if (prop_d2h) {
      RVSTRACE_
      // initiate reverse transfer
      hsa_signal_store_relaxed(ctx_rev.Sig, 1);
      if (HSA_STATUS_SUCCESS != (status = hsa_amd_memory_async_copy(
                    ctx_rev.pDstBuff, ctx_rev.DstAgent,
                    ctx_rev.pSrcBuff, ctx_rev.SrcAgent,
                    b2b_block_size,
                    0, NULL, ctx_rev.Sig))) {
        rvs::hsa::print_hsa_status(__FILE__, __LINE__, __func__,
                "hsa_amd_memory_async_copy()",
                status);
        break;
      }
    }

    // wait for transfer to complete
    if (prop_h2d) {
      RVSTRACE_
      while (hsa_signal_wait_acquire(ctx_fwd.Sig, HSA_SIGNAL_CONDITION_LT,
      1, uint64_t(-1), HSA_WAIT_STATE_ACTIVE)) {}
    }

    // if bidirectional, also wait for reverse transfer to complete
    if (prop_d2h) {
      RVSTRACE_
      while (hsa_signal_wait_acquire(ctx_rev.Sig, HSA_SIGNAL_CONDITION_LT,
      1, uint64_t(-1), HSA_WAIT_STATE_ACTIVE)) {}
    }

    RVSTRACE_
    // get transfer duration
    double duration = 0.0;
    if (!prop_h2d && prop_d2h) {
      duration = pHsa->GetCopyTime(bidirect,
                                  ctx_rev.Sig, ctx_fwd.Sig)/1000000000;
    } else {
      duration = pHsa->GetCopyTime(bidirect,
                                  ctx_fwd.Sig, ctx_rev.Sig)/1000000000;
    }

    {
      RVSTRACE_
      std::lock_guard<std::mutex> lk(cntmutex);
      running_size += b2b_block_size;
      running_duration += duration;
    }
  }  // while(brun)

  RVSTRACE_
  // deallocate buffers and signals
  deinit();
}

