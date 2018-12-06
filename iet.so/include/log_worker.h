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
#ifndef IET_SO_INCLUDE_LOG_WORKER_H_
#define IET_SO_INCLUDE_LOG_WORKER_H_

#include <string>
#include <mutex>
#include "include/rvsthreadbase.h"

/**
 * @class log_worker
 * @ingroup IET
 *
 * @brief log_worker action implementation class
 *
 * Derives from rvs::ThreadBase and implements actual action functionality
 * in its run() method.
 *
 */
class log_worker : public rvs::ThreadBase {
 public:
    explicit log_worker(bool _bjson);
    virtual ~log_worker();

    //! sets action name
    void set_name(const std::string& name) { action_name = name; }
    //! returns action name
    const std::string& get_name(void) { return action_name; }

    //! sets the GPU power-index
    void set_pwr_device_id(int _pwr_device_id) {
        pwr_device_id = _pwr_device_id;
    }
    //! returns the GPU power-index
    int get_pwr_device_id(void) { return pwr_device_id; }

    //! sets GPU ID
    void set_gpu_id(uint16_t _gpu_id) { gpu_id = _gpu_id; }
    //! returns GPU ID
    uint16_t get_gpu_id(void) { return gpu_id; }

    //! sets the time interval at which the module reports the GPU power
    void set_log_interval(uint64_t _log_interval) {
        log_interval = _log_interval;
    }
    //! returns the time interval at which the module reports the GPU power
    uint64_t get_log_interval(void) { return log_interval; }

    void pause(void);
    void resume(void);
    void stop(void);

 protected:
    virtual void run(void);
    void log_to_json(const std::string &key, const std::string &value,
                     int log_level);

 protected:
    //! name of the action
    std::string action_name;
    //! GPU's power-index
    uint32_t pwr_device_id;
    //! ID of the GPU that will run the EDPp test
    uint16_t gpu_id;
    //! time interval at which the GPU power is computed and logged out
    uint64_t log_interval;
    //! TRUE if JSON output is required
    bool bjson;
    //! Loops while TRUE
    bool brun;
    //! TRUE is the worker is paused
    bool bpaused;

    //! brun synchronization mutex
    std::mutex mtx_brun;
    //! bpaused synchronization mutex
    std::mutex mtx_bpaused;
};
#endif  // IET_SO_INCLUDE_LOG_WORKER_H_
