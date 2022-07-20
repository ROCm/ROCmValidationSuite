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
#include "include/hiptest_worker.h"

#include <unistd.h>
#include <string>
#include <memory>
#include <iostream>
#include "include/rvs_blas.h"
#include "include/rvs_module.h"
#include "include/rvsloglp.h"
#include "include/rvs_util.h"

#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/wait.h>

#define MODULE_NAME                             "hiptest"

using std::string;

//bool hipTestWorker::bjson = false;

hipTestWorker::hipTestWorker() {}
hipTestWorker::~hipTestWorker() {}


/**
 * @brief performs the stress test on the given GPU
 */
void hipTestWorker::run() {
    string msg, err_description;
    int error = 0;


    // log GST stress test - start message
    msg = "[" + action_name + "] " + MODULE_NAME + " " +
            " Starting the Hip test "; 
    rvs::lp::Log(msg, rvs::logtrace);

    // let the GPU ramp-up and check the result
    bool hipsuccess = start_hip_tests(error, err_description);

    // GPU was not able to do the processing (HIP/rocBlas error(s) occurred)
    if (error) {
        string msg = "[" + action_name + "] " + MODULE_NAME + " "
                         + err_description;
        rvs::lp::Log(msg, rvs::logerror);
        return;
    }

}

/**
 * @brief forks and execs test result
 * @param return true if test succeeded, false otherwise
 */

bool hipTestWorker::start_hip_tests(int &error, string &errdesc){
    int pid, status;
    auto found = m_test_path.find_last_of('/');
    auto fname = m_test_path.substr(found+1);
    if((pid = fork()) == 0){ // child
	execl(m_test_path.c_str(), fname.c_str(), m_test_args.c_str(), 0);
    }else{
	waitpid(pid, &status, 0);
	if (WIFEXITED(status)){
	    error = 0;
	    return true;
	}
	return false;
    }
}
