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
#ifndef PESM_SO_INCLUDE_WORKER_H_
#define PESM_SO_INCLUDE_WORKER_H_

#include <string>
#include <vector>

#include "include/rvsthreadbase.h"


/**
 * @class Worker
 * @ingroup PESM
 *
 * @brief Monitoring implementation class
 *
 * Derives from rvs::ThreadBase and implements actual monitoring functionality
 * in its run() method.
 *
 */

class Worker : public rvs::ThreadBase {
 public:
  Worker();
  virtual ~Worker();

  //! Stops monitoring
  void stop(void);
  //! Sets initiating action name
  void set_name(const std::string& name) { action_name = name; }
  //! sets stopping action name
  void set_stop_name(const std::string& name) { stop_action_name = name; }
  //! Sets device id for filtering
  void set_deviceid(const int id) { device_id = id; }
  //! Sets GPU IDs for filtering
  void set_gpuids(const std::vector<uint16_t>& GpuIds);
  //! Sets GPU IDs for filtering (string used in messages)
  //! @param Devices List of devices to monitor
  void set_strgpuids(const std::string& Devices) { strgpuids = Devices; }
  //! Sets JSON flag
  void json(const bool flag) { bjson = flag; }
  //! Returns initiating action name
  const std::string& get_name(void) { return action_name; }

 protected:
  virtual void run(void);

 protected:
  //! TRUE if JSON output is required
  bool    bjson;
  //! Loops while TRUE
  bool     brun;
  //! device id to filter for. 0 if no filtering.
  int device_id;
  //! GPU id filtering flag
  bool bfiltergpu;
  //! list of GPU devices to monitor
  std::vector<uint16_t> gpuids;
  //! list of GPU devices to monitor (string used in messages)
  std::string strgpuids;
  //! Name of the action which initiated monitoring
  std::string  action_name;
  //! Name of the action which stops monitoring
  std::string  stop_action_name;
};



#endif  // PESM_SO_INCLUDE_WORKER_H_
