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
#include "action_pkgchk.h"

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

#include "rvs_module.h"
#include "rvsliblogger.h"
#include "rvs_util.h"

#define BUFFER_SIZE 3000

using namespace std;

int pkgchk_run(std::map<string,string> property){
  map<string, string>::iterator iter;
  
  iter = property.find("package");
  if(iter != property.end()){
    bool version_exists = false;
    int fd[2];
    string package_name;
    
    package_name = iter->second;
    string version_name;
    iter = property.find("version");
    if(iter != property.end()){
      version_name = iter->second;
      version_exists = true;
    }
    
    pid_t pid;
    pipe(fd);
    
    pid = fork();
    if(pid == 0){
      // Child process
      
      dup2(fd[1], STDOUT_FILENO);
      dup2(fd[1], STDERR_FILENO);
      char buffer[256];
      snprintf(buffer, 255, "dpkg-query -W -f='${Status} ${Version}\n' %s", package_name.c_str());
      
      system(buffer);
      
    } else if (pid>0){
      // Parent
      
      char result[BUFFER_SIZE];
      int count;
      close(fd[1]);
      
      count=read(fd[0], result, BUFFER_SIZE);
      
      result[count]=0;
      string result1 = result;
      
      string version_value = result1.substr(21, count - 21);
      int ending = version_value.find_first_of('\n');
      if(ending > 0){
        version_value = version_value.substr(0, ending);
      }
      string passed = "packagecheck " + package_name + " TRUE";
      string failed = "packagecheck " + package_name + " FALSE";
      
      if(strstr(result, "dpkg-query:") == result)
        log(failed.c_str(), rvs::logresults);
      else if(version_exists == false){
        log(passed.c_str(), rvs::logresults);
      }else if(version_name.compare(version_value) == 0){
        log(passed.c_str(), rvs::logresults);
      }else{
        log(failed.c_str(), rvs::logresults);
      }
      
    }else   {
      // fork process error
      cerr << "Internal Error" << endl;
      return -1;
    }
  }
  
  
  
}