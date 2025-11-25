/********************************************************************************
 *
 * Copyright (c) 2024 ROCm Developer Tools
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
#include "include/em_worker.h"

#include <unistd.h>
#include <string>
#include <memory>
#include <iostream>
#include <fstream>
#include "include/rvs_module.h"
#include "include/rvsloglp.h"
#include "include/rvs_util.h"

#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/wait.h>

#define MODULE_NAME                             "runexternal"

using std::string;

//bool emWorker::bjson = false;

emWorker::emWorker() {}
emWorker::~emWorker() {}


/**
 * @brief performs the stress test on the given GPU
 */
void emWorker::run() {
    string msg, err_description;
    int error = 0;
    auto pos = m_test_path.find_last_of('/');
    string tname = (pos != std::string::npos) ? m_test_path.substr(pos+1) : "" ;
    if (pos != std::string::npos)
    msg = "[" + action_name + "] " + MODULE_NAME + " " +
            " Starting :  " + tname; 
    rvs::lp::Log(msg, rvs::loginfo);

    bool status = start_ext_tests(error, err_description);

    if (error) {
        string msg = "[" + action_name + "] " + MODULE_NAME + " "
                         + err_description;
        rvs::lp::Log(msg, rvs::logerror);
        return;
    }
    msg = "[" + action_name + "] " + MODULE_NAME + " " +
            " Completed running :  " + tname;
    rvs::lp::Log(msg, rvs::loginfo);

}

/**
 * @brief forks and execs test result
 * @param return true if test succeeded, false otherwise
 */

bool emWorker::start_ext_tests(int &error, string &errdesc){
    int pid, status;
    auto found = m_test_path.find_last_of('/');
    auto fname = m_test_path.substr(found+1);
    {
      std::ifstream f(m_test_path.c_str());
      if(! f.good()){
	      error = -1;
	      errdesc = "Binary file absent";
	      return false;
      }
    }
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
