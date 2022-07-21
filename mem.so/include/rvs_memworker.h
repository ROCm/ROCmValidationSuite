/********************************************************************************
 *
 * Copyright (c) 2018-2022 ROCm Developer Tools
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
#ifndef MEM_SO_INCLUDE_MEM_WORKER_H_
#define MEM_SO_INCLUDE_MEM_WORKER_H_

#include <vector>
#include "include/rvsthreadbase.h"


#define TDIFF(tb, ta) (tb.tv_sec - ta.tv_sec + 0.000001*(tb.tv_usec - ta.tv_usec))
#define MEM_RESULT_PASS_MESSAGE         "true"
#define MEM_RESULT_FAIL_MESSAGE         "false"
#define ERR_GENERAL             -999

#define MODULE_NAME                     "mem"
#define MODULE_NAME_CAPS                "MEM"

#if 0
#define HIP_CHECK(status)                                                                          \
     if (status != hipSuccess) {                                                                    \
         std::cout << "Got Status: " << status << " at Line: " << __LINE__ << std::endl;            \
         exit(0);                                                                                   \
     }
#endif

#define HIP_CHECK(error)                                                                            \
    {                                                                                              \
        hipError_t localError = error;                                                             \
        if ((localError != hipSuccess)&& (localError != hipErrorPeerAccessAlreadyEnabled)&&        \
                     (localError != hipErrorPeerAccessNotEnabled )) {                              \
            printf("%serror: '%s'(%d) from %s at %s:%d%s\n", KRED, hipGetErrorString(localError),  \
                   localError, #error, __FILE__, __LINE__, KNRM);                                  \
            failed("API returned error code.");                                                    \
        }                                                                                          \
    }



#if 1
#define MEM_MEM_ALLOC_ERROR                     "memory allocation error!"
#define MEM_BLAS_ERROR                          "memory/blas error!"
#define MEM_BLAS_MEMCPY_ERROR                   "HostToDevice mem copy error!"
#define MAX_ERR_RECORD_COUNT                    10
#define MEM_NUM_SAVE_BLOCKS                     16

#define MEM_START_MSG                           "start"
#define MEM_PASS_KEY                            "pass"
#endif


/**
 * @class MEMWorker
 * @ingroup MEM
 *
 * @brief MEMWorker action implementation class
 *
 * Derives from rvs::ThreadBase and implements actual action functionality
 * in its run() method.
 *
 */
class MemWorker : public rvs::ThreadBase {
 public:
    MemWorker();
    virtual ~MemWorker();

    void list_tests_info(void);

    void usage(char** argv);

    void run_tests(char* ptr, unsigned int tot_num_blocks);

    void test0(char* ptr, unsigned int tot_num_blocks);

    //! sets action name
    void set_name(const std::string& name) { action_name = name; }
    //! returns action name
    const std::string& get_name(void) { return action_name; }

    //! sets GPU ID
    void set_gpu_id(uint16_t _gpu_id) { gpu_id = _gpu_id; }
    //! returns GPU ID
    uint16_t get_gpu_id(void) { return gpu_id; }

    //! sets the GPU index
    void set_gpu_device_index(int _gpu_device_index) {
        gpu_device_index = _gpu_device_index;
    }
    //! returns the GPU index
    int get_gpu_device_index(void) { return gpu_device_index; }

    //! sets the run delay
    void set_run_wait_ms(uint64_t _run_wait_ms) { run_wait_ms = _run_wait_ms; }
    //! returns the run delay
    uint64_t get_run_wait_ms(void) { return run_wait_ms; }

    //! sets the total stress test run duration
    void set_run_duration_ms(uint64_t _run_duration_ms) {
        run_duration_ms = _run_duration_ms;
    }
    //! returns the total stress test run duration
    uint64_t get_run_duration_ms(void) { return run_duration_ms; }

    //! sets the mapped memory property
    void set_mapped_mem(bool _mapped_mem) {
        useMappedMemory = _mapped_mem;
    }
    //! Gets the mapped memory property
    uint64_t get_mapped_mem(void) { 
      return useMappedMemory; }

    //! sets the max num of blocks
    void set_num_mem_blocks(uint64_t _num_blocks) {
        max_num_blocks = _num_blocks;
    }
    //! returns the max num of blocks
    uint64_t get_num_mem_blocks(void) { 
      return max_num_blocks; 
    }

    //! sets the memory pattern
    void set_pattern(uint64_t _pattern) { pattern = _pattern; }

    //! returns the memory pattern
    bool get_pattern(void) { return pattern; }

    //! sets the number of iterations
    void set_num_iterations(uint64_t _num_iterations) {
        num_iterations = _num_iterations;
    }
    //! returns the number of iterations
    uint64_t get_num_iterations(void) { return num_iterations; }

    //! set num passes
    void set_num_passes(uint64_t _num_pases) {
        num_passes = _num_pases;
    }
 
    //!get num passes
    uint64_t get_num_passes(void) {
        return num_passes;
    }

    //! set num passes
    void set_threads_per_block(uint64_t _threads_per_blk) {
        threadsPerBlock = _threads_per_blk;
    }
 
    //!get num passes
    uint64_t get_threads_per_block(void) {
        return threadsPerBlock;
    }

    //! sets the SGEMM matrix size
    void set_stress(uint64_t _stress) {
        stress = _stress;
    }

    //! sets the SGEMM matrix size
    bool get_stress() {
        return stress;
    }

    //! sets the JSON flag
    static void set_use_json(bool _bjson) { bjson = _bjson; }
    //! returns the JSON flag
    static bool get_use_json(void) { return bjson; }
    static void init_tests(const std::vector<uint32_t>& exclude_list);

 protected:
    void setup_blas(int *error, std::string *err_description);
    void hit_max_gflops(int *error, std::string *err_description);
    bool do_mem_ramp(int *error, std::string *err_description);
    bool do_mem_stress_test(int *error, std::string *err_description);
    void log_mem_test_result(bool mem_test_passed);
    virtual void run(void);
    void log_to_json(const std::string &key, const std::string &value,
                     int log_level);
    void log_interval_gflops(double gflops_interval);
    bool check_gflops_violation(double gflops_interval);
    void check_target_stress(double gflops_interval);
    void usleep_ex(uint64_t microseconds);
    void Initialization(void);

 protected:
    //! name of the action
    std::string action_name;
    //! index of the GPU that will run the stress test
    int gpu_device_index;
    //! ID of the GPU that will run the stress test
    uint16_t gpu_id;
    //! stress test run delay
    uint64_t run_wait_ms;
    //! stress test run duration
    uint64_t run_duration_ms;
    //! Memory mapped
    uint64_t mem_mapped;
    //! Max number of blocks
    uint64_t max_num_blocks;
    //! Mapped mem
    bool useMappedMemory;
    //! Num of passes
    uint64_t num_passes;
    //! Pattern
    uint64_t pattern;
    //! Number of iterations
    uint64_t num_iterations;
    //! stress
    bool stress;
    //! TRUE if JSON output is required
    static bool bjson;
    //! synchronization mutex
    std::mutex wrkrmutex;
    //threads per block
    uint64_t  threadsPerBlock;
    //Mapped memory pointer
    void*   mappedHostPtr;
};

#endif  // MEM_SO_INCLUDE_MEM_WORKER_H_
