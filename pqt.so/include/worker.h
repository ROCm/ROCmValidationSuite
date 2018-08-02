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
#include <mutex>

#include "rvsthreadbase.h"


/**
 * @class Worker
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

class Worker : public rvs::ThreadBase {
 public:
  //! default constructor
  Worker();
  //! default destructor
  virtual ~Worker();

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

  int initialize(int iSrc, int iDst, bool Bidirect);
  int do_transfer();
  void get_running_data(int* Src, int* Dst, bool* Bidirect,
                        size_t* Size, double* Duration);
  void get_final_data(int* Src, int* Dst, bool* Bidirect,
                      size_t* Size, double* Duration);

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
  int src_node;
  //! destination NUMA node
  int dst_node;
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

  //! synchronization mutex
  std::mutex cntmutex;
};

#endif  // PQT_SO_INCLUDE_WORKER_H_
