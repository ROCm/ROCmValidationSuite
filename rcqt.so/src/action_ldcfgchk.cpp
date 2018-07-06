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
#include <stdlib.h>
#include <map>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <sys/utsname.h>


#define SONAME  "soname"
#define LDPATH  "ldpath"
#define ARCH    "arch"

#define BUFFER_SIZE 3000

using namespace std;

/**
 * Check if the shared object is in the given location with the correct architecture
 * @param property config file map fields
 * @return 0 - success, non-zero otherwise
 * */

int ldcfgchk_run(std::map<string,string> property) {
  
  string soname_requested;
  auto iter = property.find(SONAME);
  if(iter != property.end()) {
    soname_requested = iter->second;
    iter = property.find(ARCH);
    if(iter == property.end()) {
      cerr << "acrhitecture field missing in conflig" << endl;
      return -1;
    }
    string arch_requested = iter->second;
    iter = property.find("ldpath");
    if(iter == property.end()) {
      cerr << "libraty path field missing in conflig" << endl;
      return -1;
    }
    string ldpath_requested = iter->second;
    // Full path of shared object
    string full_ld_path = ldpath_requested + "/" + soname_requested;
    
    int fd[2];
    pid_t pid;
    pipe(fd);
    pid = fork();
    if(pid == 0) {
      // child process
      dup2(fd[1], STDOUT_FILENO);
      dup2(fd[1], STDERR_FILENO);
      char buffer[256];
      snprintf(buffer,256, "objdump -f %s", full_ld_path.c_str());
      
      system(buffer);
    }
    else if(pid > 0) {
      // Parent process
      char result[BUFFER_SIZE];
      int count;
      close(fd[1]);
      
      count=read(fd[0], result, BUFFER_SIZE);
      string ld_config_result = "[rcqt] ldconfigcheck ";
      
      string result_string = result;
      
      if(strstr(result, "architecture:") != nullptr) {
        vector<string> objdump_lines = str_split(result_string, "\n");
        int begin_of_the_arch_string = 0;
        int end_of_the_arch_string = 0;
        for(int i = 0; i < objdump_lines.size(); i++){
          //cout << objdump_lines[i] << "*" << endl;
          if( objdump_lines[i].find("architecture") != string::npos) {
            begin_of_the_arch_string = objdump_lines[i].find(":");
            end_of_the_arch_string = objdump_lines[i].find(",");
            string arch_found = objdump_lines[i].substr(begin_of_the_arch_string + 2, end_of_the_arch_string - begin_of_the_arch_string - 2);
            if(arch_found.compare(arch_requested) == 0){
              string arch_pass = ld_config_result + soname_requested + " " + full_ld_path + " " + arch_found + " pass";
              log(arch_pass.c_str(), rvs::logresults);
            }else {
              string arch_pass = ld_config_result + soname_requested + " " + full_ld_path + " " + arch_found + " fail";
              log(arch_pass.c_str(), rvs::logresults);
            }
          }
        }
      }else {
        string lib_fail = ld_config_result + soname_requested + " not found " + "na " + "fail" ;
        log(lib_fail.c_str(), rvs::logresults);
      }
      
    }else
      cerr << "Internal Error" << endl;
    return 0;
  }
  return -1;
}