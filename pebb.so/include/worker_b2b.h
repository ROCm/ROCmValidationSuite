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
#ifndef PEBB_SO_INCLUDE_WORKER_B2B_H_
#define PEBB_SO_INCLUDE_WORKER_B2B_H_

#include <string>
#include <vector>
#include <mutex>

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

#include "include/worker.h"



/**
 * @class pebbworker_b2b
 * @ingroup PEBB
 *
 * @brief Bandwidth test back-to-back transfer implementation class
 *
 * Derives from pebbworker and implements actual test functionality
 * in its run() and do_transfer() methods.
 *
 */

namespace rvs {
class hsa;
}
// hsa_signal_exchange_scacq_screl
class pebbworker_b2b : public pebbworker {
 public:
/**
 * @class transfer_context_t
 * @ingroup PEBB
 *
 * @brief Utility class used to store transfer context for back-to-back
 * transfers
 *
 */
  typedef struct {
    //! source agent indes in @p rvs::hsa::agentagent_list
    int SrcAgentIx;
    //! source HSA agent
    hsa_agent_t SrcAgent;
    //! destination agent indes in @p rvs::hsa::agentagent_list
    int DstAgentIx;
    //! destination HSA agent
    hsa_agent_t DstAgent;
    //! source HSA memory pool
    hsa_amd_memory_pool_t SrcPool;
    //! source buffer
    void* pSrcBuff;
    //! destination HSA memory pool
    hsa_amd_memory_pool_t DstPool;
    //! destination buffer
    void* pDstBuff;
    //! signal used for async transfer timing
    hsa_signal_t Sig;
  } transfer_context_t;

 public:
  //! default constructor
  pebbworker_b2b();
  //! default destructor
  virtual ~pebbworker_b2b();

  int initialize(uint16_t iSrc, uint16_t iDst, bool h2d, bool d2h, size_t Size);
  //! Set back-to-back block size
  void set_b2b_block_sizes(const size_t val) { b2b_block_size = val; }

 protected:
  virtual void run(void);
  void deinit();

 protected:
  //! size of data block used in back-to-back transfer
  size_t b2b_block_size;
  //! context of forward (host-to-device) transfer
  transfer_context_t ctx_fwd;
  //! context of revers (device-to-host) transfer
  transfer_context_t ctx_rev;
};

#endif  // PEBB_SO_INCLUDE_WORKER_B2B_H_
