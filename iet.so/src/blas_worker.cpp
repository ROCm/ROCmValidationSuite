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
#include "include/blas_worker.h"

#include <unistd.h>
#include <string>
#include <memory>
#include <mutex>

#include "include/rvs_blas.h"
#include "include/rvsloglp.h"

#define IET_MEM_ALLOC_ERROR                     1
#define IET_BLAS_ERROR                          2
#define IET_BLAS_MEMCPY_ERROR                   3
#define MODULE_NAME "IET"

#define USLEEP_MAX_VAL                          (1000000 - 1)

using std::string;

/**
 * @brief default class constructor
 * @param _gpu_device_index index of the gpu that will run the GEMM
 * @param _matrix_size size of SGEMM m atrices
 */
blas_worker::blas_worker(int _gpu_device_index, uint64_t _matrix_size) :
                            gpu_device_index(_gpu_device_index),
                            matrix_size(_matrix_size) {
    bcount_sgemm = false;
    sgemm_delay = 0;
    blas_error = 0;
    setup_finished = false;
    bpaused = false;
}

blas_worker::~blas_worker() {}

/**
 * @brief performs the rvsBlas setup
 */
void blas_worker::setup_blas(void) {
    blas_error = 0;
    // setup rvsBlas
    gpu_blas = std::unique_ptr<rvs_blas>(
        new rvs_blas(gpu_device_index, matrix_size, matrix_size, matrix_size));

    // no lock guard for blas_error atm because there are no sync issues
    if (gpu_blas == nullptr) {
        blas_error = IET_MEM_ALLOC_ERROR;
        set_setup_complete();
        return;
    }

    if (gpu_blas->error()) {
        blas_error = IET_BLAS_ERROR;
        set_setup_complete();
        return;
    }

    // generate random matrix & copy it to the GPU
    gpu_blas->generate_random_matrix_data();
    if (!gpu_blas->copy_data_to_gpu()) {
        blas_error = IET_BLAS_MEMCPY_ERROR;
        set_setup_complete();
        return;
    }

    set_setup_complete();
}

/**
 * @brief marks the BLAS setup as completed
 */
void blas_worker::set_setup_complete(void) {
    std::lock_guard<std::mutex> lck(mtx_blas_setup);
    setup_finished = true;
}

/**
 * @brief checks for BLAS setup completeness
 * @return true if BLAS setup finished, false otherwise
 */
bool blas_worker::is_setup_complete(void) {
    std::lock_guard<std::mutex> lck(mtx_blas_setup);
    return setup_finished;
}

/**
 * @brief checks for SGEMM completeness
 * @return true if last SGEMM finished, false otherwise
 */
bool blas_worker::is_sgemm_complete(void) {
    std::lock_guard<std::mutex> lck(mtx_bsgemm_done);
    return sgemm_done;
}

/**
 * @brief sets the brun flag to false (signal the thread to stop)
 */
void blas_worker::stop(void) {
    std::lock_guard<std::mutex> lck(mtx_brun);
    brun = false;
}

/**
 * @brief returns the total number of SGEMM that the thread managed to run
 * @return SGEMMs number
 */
uint64_t blas_worker::get_num_sgemm_ops(void) {
    std::lock_guard<std::mutex> lck(mtx_num_sgemm);
    return num_sgemm_ops;
}

/**
 * @brief sets whether SGEMM counting is needed or not
 * @param _bcount_sgemm true if SGEMM ops counting is needed, false otherwise
 */
void blas_worker::set_bcount_sgemm(bool _bcount_sgemm) {
    std::lock_guard<std::mutex> lck(mtx_bcount_sgemm);
    bcount_sgemm = _bcount_sgemm;
}

/**
 * @brief checks for the SGEMM-counter active-flag status 
 * @return true if BLAS was setup to count the SGEMM ops, false otherwise
 */
bool blas_worker::get_bcount_sgemm(void) {
    std::lock_guard<std::mutex> lck(mtx_bcount_sgemm);
    return bcount_sgemm;
}

/**
 * @brief sets the SGEMM delay(frequency)
 * @param _sgemm_delay SGEMM delay
 */
void blas_worker::set_sgemm_delay(uint64_t _sgemm_delay) {
    std::lock_guard<std::mutex> lck(mtx_sgemm_delay);
    sgemm_delay = _sgemm_delay;
}

/**
 * @brief pauses the BLAS worker
 */
void blas_worker::pause(void) {
    std::lock_guard<std::mutex> lck(mtx_bpaused);
    bpaused = true;
}

/**
 * @brief resumes the BLAS worker
 */
void blas_worker::resume(void) {
    std::lock_guard<std::mutex> lck(mtx_bpaused);
    bpaused = false;
}

/**
 * @brief returns the current SGEMM delay
 * @return SGEMM delay
 */
uint64_t blas_worker::get_sgemm_delay(void) {
    std::lock_guard<std::mutex> lck(mtx_sgemm_delay);
    return sgemm_delay;
}

/**
 * @brief performs SGEMMs on the selected GPU with a given frequency
 */
void blas_worker::run() {
    setup_blas();
    if (blas_error)
        return;

    {
        std::lock_guard<std::mutex> lck(mtx_brun);
        brun = true;
    }

    {
        std::lock_guard<std::mutex> lck(mtx_num_sgemm);
        num_sgemm_ops = 0;
    }

    for (;;) {
        {
            std::lock_guard<std::mutex> lck(mtx_brun);
            if (!brun)
                break;
        }

        {
            std::lock_guard<std::mutex> lck(mtx_bpaused);
            if (bpaused)
                continue;
        }

        {
            std::lock_guard<std::mutex> lck(mtx_bsgemm_done);
            sgemm_done = false;
        }

        bool sgemm_success = true;
        // run SGEMM & wait for completion
        if (gpu_blas->run_blass_gemm()) {
            while (!gpu_blas->is_gemm_op_complete()) {}
        } else {
            sgemm_success = false;
        }

        {
            std::lock_guard<std::mutex> lck(mtx_bsgemm_done);
            sgemm_done = true;
        }

        // increase number of SGEMM ops
        if (sgemm_success) {
            {
                std::lock_guard<std::mutex> lck(mtx_bcount_sgemm);
                if (bcount_sgemm) {
                    // lock_guard [num_sgemm_ops]
                    std::lock_guard<std::mutex> lck(mtx_num_sgemm);
                    num_sgemm_ops++;
                }
            }

            // lock_guard [sgemm_delay]
            {
                std::lock_guard<std::mutex> lck(mtx_sgemm_delay);
                usleep_ex(sgemm_delay);
            }
        }

        // check if stop signal was received
        if (rvs::lp::Stopping())
            break;
    }
}

/**
 * @brief extends the usleep for more than 1000000us
 * @param microseconds us to sleep
 */
void blas_worker::usleep_ex(uint64_t microseconds) {
    uint64_t total_microseconds = microseconds;
    for (;;) {
         if (total_microseconds > USLEEP_MAX_VAL) {
            usleep(USLEEP_MAX_VAL);
            total_microseconds -= USLEEP_MAX_VAL;
        } else {
            usleep(total_microseconds);
            return;
        }
    }
}
