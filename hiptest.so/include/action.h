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
#ifndef HIPTEST_SO_INCLUDE_ACTION_H_
#define HIPTEST_SO_INCLUDE_ACTION_H_

#ifdef __cplusplus
extern "C" {
#endif
#include <pci/pci.h>
#ifdef __cplusplus
}
#endif

#include <vector>
#include <string>
#include <map>

#include "include/rvsactionbase.h"

using std::vector;
using std::string;
using std::map;

/**
 * @class hiptest_action
 * @ingroup GST
 *
 * @brief GST action implementation class
 *
 * Derives from rvs::actionbase and implements actual action functionality
 * in its run() method.
 *
 */
class hiptest_action: public rvs::actionbase {
 public:
    hiptest_action();
    virtual ~hiptest_action();

    virtual int run(void);

    std::string m_test_file_path;
    std::string m_test_file_args;

 protected:
    //! TRUE if JSON output is required
    bool bjson;

    bool get_all_hiptest_config_keys(void);
  /**
  * @brief reads all common configuration keys from
  * the module's properties collection
  * @return true if no fatal error occured, false otherwise
  */
    bool get_all_common_config_keys(void);

    int get_num_amd_gpu_devices(void);
    bool start_hiptest_runners();
  /**
  * @brief gets the number of ROCm compatible AMD GPUs
  * @return run number of GPUs
  */
    int run_hip_tests();
};

#endif  // HIPTEST_SO_INCLUDE_ACTION_H_
