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
#include "rcqt_subactions.h"

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <pwd.h>
#include <grp.h>
#include <string.h>
#include <vector>
#include <sys/utsname.h>



#define BUFFER_SIZE 256
#define PACKAGE "package"
#define VERSION "version"
#define INTERNAL_ERROR "Internal Error"

using namespace std;

/**
 * Check if the package is installed in the system (optional: check package version )
 * @param property config file map fields
 * @return 0 - success, non-zero otherwise
 * */

int pkgchk_run(map<string,string> property) {
  //static rvs::actionbase actionbase;
  
  string package_name;
  auto iter = property.find(PACKAGE);
  if(iter != property.end()) {
    bool version_exists = false;
    package_name = iter->second;
    string version_name;
    
    iter = property.find(VERSION);
    // Checking if version field exists
    if(iter != property.end()) {
      version_exists = true;
      version_name = iter->second;
    }
    
    pid_t pid;
    int fd[2];
    pipe(fd);
    
    pid = fork();
    if(pid == 0) {
      // Child process
      
      // Pipe the standard output to the fd[1]
      dup2(fd[1], STDOUT_FILENO);
      dup2(fd[1], STDERR_FILENO);
      char buffer[BUFFER_SIZE];
      snprintf(buffer, BUFFER_SIZE, "dpkg-query -W -f='${Status} ${Version}\n' %s", package_name.c_str());
      
      // We execute the dpkg-querry
      system(buffer);
      
    } else if (pid>0) {
      // Parent
      
      char result[BUFFER_SIZE];
      int count;
      close(fd[1]);
      
      // We read the result from the dpk-querry from the fd[0]
      count=read(fd[0], result, BUFFER_SIZE);
      
      result[count]=0;
      string result1 = result;
      
      // We parse the given result
      string version_value = result1.substr(21, count - 21);
      int ending = version_value.find_first_of('\n');
      if(ending > 0) {
        version_value = version_value.substr(0, ending);
      }
      string passed = "packagecheck " + package_name + " TRUE";
      string failed = "packagecheck " + package_name + " FALSE";
      
      /* 
       * If result start with dpkg-querry: then we haven't found the package
       * If we get something different, then we confirme that the package is found
       * if version is equal to the required then the test pass
       */
      if(strstr(result, "dpkg-query:") == result)
        log(failed.c_str(), rvs::logresults);
      else if(version_exists == false) {
        log(passed.c_str(), rvs::logresults);
      }else if(version_name.compare(version_value) == 0) {
        log(passed.c_str(), rvs::logresults);
      }else {
        log(failed.c_str(), rvs::logresults);
      }
      
    }else {
      // fork process error
      cerr << INTERNAL_ERROR << endl;
      return -1;
    }
    return 0;
  }
  return -1;  
}