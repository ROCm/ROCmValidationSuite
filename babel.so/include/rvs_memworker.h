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
#ifndef MEM_SO_INCLUDE_MEM_WORKER_H_
#define MEM_SO_INCLUDE_MEM_WORKER_H_

#include "include/rvsthreadbase.h"


#define TDIFF(tb, ta) (tb.tv_sec - ta.tv_sec + 0.000001*(tb.tv_usec - ta.tv_usec))
#define MEM_RESULT_PASS_MESSAGE         "true"
#define MEM_RESULT_FAIL_MESSAGE         "false"
#define ERR_GENERAL             -999

#define MODULE_NAME                     "babel"
#define MODULE_NAME_CAPS                "BABEL"


#define KNRM "\x1B[0m"
#define KRED "\x1B[31m"
#define KGRN "\x1B[32m"
#define KYEL "\x1B[33m"
#define KBLU "\x1B[34m"
#define KMAG "\x1B[35m"
#define KCYN "\x1B[36m"
#define KWHT "\x1B[37m"

#define DEBUG_PRINTF(fmt,...) do {          \
      PRINTF(fmt, ##__VA_ARGS__);         \
}while(0)


#define PRINTF(fmt,...) do{           \
  printf("[%s][%s][%d]:" fmt, time_string(), hostname, gpu_idx, ##__VA_ARGS__); \
  fflush(stdout);             \
} while(0)

#define FPRINTF(fmt,...) do{            \
  fprintf(stderr, "[%s][%s][%d]:" fmt, time_string(), hostname, gpu_idx, ##__VA_ARGS__); \
  fflush(stderr);             \
} while(0)

#define HIP_ASSERT(x) (assert((x)==hipSuccess))

#define RVS_DEVICE_SERIAL_BUFFER_SIZE 0
#define MAX_ERR_RECORD_COUNT          10
#define MAX_NUM_GPUS                  128
#define ERR_MSG_LENGTH                4096
#define RANDOM_CT                     320000
#define RANDOM_DIV_CT                 0.1234

#define passed()                                                                                   \
    printf("%sPASSED!%s\n", KGRN, KNRM);                                                           \
    exit(0);

#define failed(...)                                                                                \
    printf("%serror: ", KRED);                                                                     \
    printf(__VA_ARGS__);                                                                           \
    printf("\n");                                                                                  \
    printf("error: TEST FAILED\n%s", KNRM);                                                        \
    abort();

#define warn(...)                                                                                  \
    printf("%swarn: ", KYEL);                                                                      \
    printf(__VA_ARGS__);                                                                           \
    printf("\n");                                                                                  \
    printf("warn: TEST WARNING\n%s", KNRM);

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

#define FLOAT_TEST    1
#define DOUBLE_TEST   2
#define TRAID_FLOAT   3
#define TRIAD_DOUBLE  4

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

    //! sets the number of iterations
    void set_num_iterations(uint64_t _num_iterations) {
        num_iterations = _num_iterations;
    }
    //! returns the number of iterations
    uint64_t get_num_iterations(void) { return num_iterations; }

    //! sets the array size
    void set_array_size(uint64_t _array_size) {
        array_size = _array_size;
    }
    //! returns the array size
    uint64_t get_array_size(void) { return array_size; }

    //! sets the test type
    void set_test_type(int _test_type) {
        test_type = _test_type;
    }
    //! returns the test type
    int get_test_type(void) { return test_type; }

    //! sets the sub test type
    void set_subtest_type(int _test_type) {
        subtest = _test_type;
    }
    //! returns the sub test type
    int get_subtest_type(void) { return subtest; }

    //! sets the mibi bytes
    void set_mibibytes(bool _mibibytes) {
        mibibytes = _mibibytes;
    }
    //! returns the nibibytes
    bool get_mibibytes(void) { return mibibytes; }

    //! sets the test type
    void set_output_csv(bool _opascsv) {
        output_csv = _opascsv;
    }
    //! returns the test type
    bool get_output_csv(void) { return output_csv; }

    //! sets the numbers of dwords per lane
    void set_dwords_per_lane(uint16_t _dwords_per_lane) {
        dwords_per_lane = _dwords_per_lane;
    }
    //! returns the numbers of dwords per lane
    uint16_t get_dwords_per_lane(void) { return dwords_per_lane; }

    //! sets the numbers of chunks per block
    void set_chunks_per_block(uint16_t _chunks_per_block) {
        chunks_per_block = _chunks_per_block;
    }
    //! returns the numbers of chunks per block
    uint16_t get_chunks_per_block(void) { return chunks_per_block; }

    static void set_use_json(bool _bjson) { bjson = _bjson; }
    //! returns the JSON flag
    static bool get_use_json(void) { return bjson; }

 protected:
    bool do_mem_stress_test(int *error, std::string *err_description);
    void log_mem_test_result(bool mem_test_passed);
    virtual void run(void);
    void log_interval_gflops(double gflops_interval);
    void usleep_ex(uint64_t microseconds);

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
    //! Number of iterations
    uint64_t num_iterations;
    //! output as csv
    bool output_csv;
    //! Mibibytes
    bool mibibytes;
    //! Number of array size
    uint64_t array_size;
    //! Test type
    int test_type;
    //! Sub Test type
    int subtest;
    //! number of dwords per lane
    uint16_t dwords_per_lane;
    //! number of chunks per block
    uint16_t chunks_per_block;

    //! TRUE if JSON output is required
    static bool bjson;
    //! synchronization mutex
    std::mutex wrkrmutex;
};

#endif  // MEM_SO_INCLUDE_MEM_WORKER_H_
