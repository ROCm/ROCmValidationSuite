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
#ifndef IET_SO_INCLUDE_BLAS_WORKER_H_
#define IET_SO_INCLUDE_BLAS_WORKER_H_

#include <string>
#include <memory>
#include <mutex>
#include "include/rvsthreadbase.h"
#include "include/rvs_blas.h"

/**
 * @class blas_worker
 * @ingroup IET
 *
 * @brief blas_worker action implementation class
 *
 * Derives from rvs::ThreadBase and implements actual action functionality
 * in its run() method.
 *
 */
class blas_worker : public rvs::ThreadBase {
 public:
    blas_worker(int _gpu_device_index, uint64_t _matrix_size);
    virtual ~blas_worker();

    void set_sgemm_delay(uint64_t _sgemm_delay);
    uint64_t get_sgemm_delay(void);

    void set_bcount_sgemm(bool _bcount_sgemm);
    bool get_bcount_sgemm(void);
    uint64_t get_num_sgemm_ops(void);

    bool is_setup_complete(void);
    bool is_sgemm_complete(void);

    void pause(void);
    void resume(void);
    void stop(void);

    //! returns the GPU index
    int get_gpu_device_index(void) { return gpu_device_index; }
    //! returns the SGEMM matrix size
    uint64_t get_matrix_size(void) { return matrix_size; }
    //! returns the BLAS error code
    int get_blas_error(void) { return blas_error; }

 protected:
    virtual void run(void);
    void set_setup_complete(void);
    void setup_blas(void);
    void usleep_ex(uint64_t microseconds);

 protected:
    //! index of the GPU that will run the SGEMM
    int gpu_device_index;
    //! SGEMM matrix size
    uint64_t matrix_size;
    //! total number of SGEMM that the thread managed to run
    uint64_t num_sgemm_ops;
    //! SGEMM delay (which gives the actual SGEMM frequency)
    uint64_t sgemm_delay;
    //! TRUE when needed to count the number of SGEMM
    bool bcount_sgemm;
    //! Loops while TRUE
    bool brun;
    //! TRUE is BLAS worker is paused
    bool bpaused;
    //! TRUE when BLAS setup finished
    bool setup_finished;
    //! TRUE if last SGEMM finished
    bool sgemm_done;
    //! brun synchronization mutex
    std::mutex mtx_brun;
    //! bpaused synchronization mutex
    std::mutex mtx_bpaused;
    //! SGEMM counter synchronization mutex
    std::mutex mtx_num_sgemm;
    //! BLAS setup synchronization mutex
    std::mutex mtx_blas_setup;
    //! SGEMM delay synchronization mutex
    std::mutex mtx_sgemm_delay;
    //! SGEMM counter flag synchronization mutex
    std::mutex mtx_bcount_sgemm;
    //! SGEMM done synchronization mutex
    std::mutex mtx_bsgemm_done;
    //! rvs_blas pointer
    std::unique_ptr<rvs_blas> gpu_blas;
    //! BLAS related error code
    int blas_error;
};
#endif  // IET_SO_INCLUDE_BLAS_WORKER_H_
