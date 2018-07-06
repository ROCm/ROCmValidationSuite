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
#include <string.h>
#include <vector>
#include <map>
#include <sys/types.h>
#include <sys/utsname.h>

#include "rvs_module.h"
#include "rvsliblogger.h"
#include "rvs_util.h"

#define OS_VERSION "os_version"
#define KERNEL_VERSION "kernel_version"

using namespace std;

/**
 * Check if the os and kernel version in the system match the givem os and kernel version
 * @param property config file map fields
 * @return 0 - success, non-zero otherwise
 * */

int kernelchk_run(std::map<string,string> property) {
  
  string os_version_values;
  string kernel_version_values;
  auto iter = property.find(OS_VERSION);
  
  if(iter != property.end()) {
    os_version_values = iter->second;
    iter = property.find(KERNEL_VERSION);
    if(iter == property.end()) {
      cerr << "Kernel version missing in config" << endl;
      return -1;
    }
    kernel_version_values = iter->second;
    /*
     * Fill the os version vector and kernel version vector with
     */
    vector<string> os_version_vector = str_split(os_version_values, ",");
    vector<string> kernel_version_vector = str_split(kernel_version_values, ",");
    
    /*
     * Parsing /etc/os-release file for pretty name to extract 
     */
    ifstream os_version_read("/etc/os-release");
    string os_actual = "";
    string os_file_line;
    bool os_version_correct = false;
    bool os_version_found_in_system = false;
    while(getline(os_version_read, os_file_line)){
      if(strcasestr(os_file_line.c_str(), "pretty_name") != nullptr) {
        os_version_found_in_system = true;
        os_actual = os_file_line.substr(13, os_file_line.length() - 14);        
        vector<string>::iterator os_iter;
        for(os_iter = os_version_vector.begin(); os_iter != os_version_vector.end(); os_iter++) {
          if(strcmp(os_iter->c_str(), os_actual.c_str()) == 0) {
            os_version_correct = true;
            break;
          }
        }
        if(os_version_correct == true)
          break;
      }
    }
    if(os_version_found_in_system == false) {
      cerr << "Unable to locate actual OS installed" << endl;
    }
    
    // Get data about the kernel version
    struct utsname kernel_version_struct ;
    if(uname(&kernel_version_struct) != 0)
      cerr << "Unable to read kernel version" << endl;
    
    
    string kernel_actual = kernel_version_struct.release ;
    bool kernel_version_correct = false;
    
    // Check if the given kernel version matches one from the list
    vector<string>::iterator kernel_iter;
    for(kernel_iter = kernel_version_vector.begin() ; kernel_iter != kernel_version_vector.end(); kernel_iter++)
      if(kernel_actual.compare(*kernel_iter) == 0 ) {
        kernel_version_correct = true;
        break;
      }
    string result = "[rcqt] kernelcheck " + os_actual + " " + kernel_actual + " " + (os_version_correct && kernel_version_correct ? "pass" : "fail");
    log(result.c_str(), rvs::logresults) ;
    return 0;
  }
  
  return -1;
}
  