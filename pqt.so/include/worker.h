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
#ifndef PQT_SO_INCLUDE_WORKER_H_
#define PQT_SO_INCLUDE_WORKER_H_

#include <string>
#include <vector>
#include <mutex>

#include "include/rvsthreadbase.h"


/**
 * @class pqtworker
 * @ingroup PQT
 *
 * @brief Bandwidth test implementation class
 *
 * Derives from rvs::ThreadBase and implements actual test functionality
 * in its run() method.
 *
 */

namespace rvs {
class hsa;
}

class pqtworker : public rvs::ThreadBase {
 public:
  //! default constructor
  pqtworker();
  //! default destructor
  virtual ~pqtworker();

  //! stop thread loop and exit thread
  void stop();
  //! Sets initiating action name
  void set_name(const std::string& name) { action_name = name; }
  //! sets stopping action name
  void set_stop_name(const std::string& name) { stop_action_name = name; }
  //! Sets JSON flag
  void json(const bool flag) { bjson = flag; }
  //! Returns initiating action name
  const std::string& get_name(void) { return action_name; }

  int initialize(uint16_t Src, uint16_t Dst, bool Bidirect);
  int do_transfer();
  void get_running_data(uint16_t* Src, uint16_t* Dst, bool* Bidirect,
                        size_t* Size, double* Duration);
  void get_final_data(uint16_t* Src, uint16_t* Dst, bool* Bidirect,
                      size_t* Size, double* Duration, bool bReset = true);
  //! Set transfer index
  void set_transfer_ix(uint16_t val) { transfer_ix = val; }
  //! Get transfer index
  uint16_t get_transfer_ix() { return transfer_ix; }
  //! Set total number of transfers
  void set_transfer_num(uint16_t val) { transfer_num = val; }
  //! Get total number of transfers
  uint16_t get_transfer_num() { return transfer_num; }
  //! Set list of test sizes
  void set_block_sizes(const std::vector<uint32_t>& val) { block_size = val; }

 protected:
  virtual void run(void);

 protected:
  //! TRUE if JSON output is required
  bool    bjson;
  //! Loops while TRUE
  bool    brun;
  //! Name of the action which initiated thread
  std::string  action_name;
  //! Name of the action which stops thread
  std::string  stop_action_name;

  //! ptr to RVS HSA singleton wrapper
  rvs::hsa* pHsa;
  //! source NUMA node
  uint16_t src_node;
  //! destination NUMA node
  uint16_t dst_node;
  //! 'true' for bidirectional transfer
  bool bidirect;

  //! Current size of transfer data
  size_t current_size;

  //! running total for size (bytes)
  size_t running_size;
  //! running total for duration (sec)
  double running_duration;

  //! final total size (bytes)
  size_t total_size;
  //! final total duration (sec)
  double total_duration;

  //! transfer index
  uint16_t transfer_ix;
  //! total number of transfers
  uint16_t transfer_num;

  //! list of test block sizes
  std::vector<uint32_t> block_size;

  //! synchronization mutex
  std::mutex cntmutex;
};

#endif  // PQT_SO_INCLUDE_WORKER_H_
