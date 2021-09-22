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
#ifndef HIPTEST_SO_INCLUDE_HIPTEST_WORKER_H_
#define HIPTEST_SO_INCLUDE_HIPTEST_WORKER_H_

#include <string>
#include <memory>
#include "include/rvsthreadbase.h"
#include "include/rvs_blas.h"
#include "include/rvs_util.h"

#define GST_RESULT_PASS_MESSAGE         "true"
#define GST_RESULT_FAIL_MESSAGE         "false"


/**
 * @class hipTestWorker
 * @ingroup GST
 *
 * @brief hipTestWorker action implementation class
 *
 * Derives from rvs::ThreadBase and implements actual action functionality
 * in its run() method.
 *
 */
class hipTestWorker : public rvs::ThreadBase {
 public:
    hipTestWorker();
    virtual ~hipTestWorker();

    //! sets action name
    void set_name(const std::string& name) { action_name = name; }
    //! returns action name
    const std::string& get_name(void) { return action_name; }

    //! sets test path
    void set_path(std::string pathname) { m_test_path = pathname; }

    const std::string& get_path(void) { return m_test_path; }
    bool start_hip_tests(int &error, std::string &errdesc);
 protected:
    virtual void run(void);
 protected:
    //! name of the action
    std::string action_name;
    //! path to execute test
    std::string m_test_path;
};

#endif  // HIPTEST_SO_INCLUDE_HIPTEST_WORKER_H_
