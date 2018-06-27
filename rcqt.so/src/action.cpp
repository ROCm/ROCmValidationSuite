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
#include "action.h"

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <map>
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


action::action() {
}

action::~action() {
  property.clear();
}

int action::run()
{
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
    
    
    iter = property.find("user");
    if(iter != property.end()){
      bool group_exists = false;
      bool user_is_in_all_groups = true;
      string user_name = iter->second;
      string group_values_string;
      iter = property.find("group");
      if(iter != property.end()){
        group_values_string = iter->second;
        group_exists= true;
      }
    
      struct passwd *p;
      struct group *g;
      string user_exists = "[rcqt] usercheck " + user_name + " user exists";
      string user_not_exists = "[rcqt] usercheck " + user_name + " user not exists";
      if((p = getpwnam(user_name.c_str())) == nullptr){
        log(user_not_exists.c_str(), rvs::logresults);
      }else{
        log( user_exists.c_str(), rvs::logresults);
      }
      if(group_exists){
        //rvs_util utility;
        string delimiter = ",";
        vector<string> group_vector ;
        group_vector = str_split(group_values_string, delimiter);
        
        for(vector<string>::iterator vector_iter = group_vector.begin(); vector_iter != group_vector.end(); vector_iter++){
          string user_group = "[rcqt] usercheck " + user_name ;
          if((g = getgrnam(vector_iter->c_str())) == nullptr){
            cerr << "group doesn't exist" << endl;
            
          }
      
          //string user_is_in_group = "[rcqt] 

          int i;
          int j=0;
          
          for(i=0; g->gr_mem[i]!=NULL; i++){
            if(strcmp(g->gr_mem[i], user_name.c_str()) == 0){
              //log("[rcqt] usercheck
              //printf("user is in group\n");
              user_group = user_group + " " + vector_iter->c_str() + " is member";
              log(user_group.c_str(), rvs::logresults);
              j=1;
              break;
            }
          }

          
          if(j==0){ 
            //printf("user is not in the group\n");
            user_group = user_group + " " + vector_iter->c_str() + " is not member";
            log(user_group.c_str(), rvs::logresults);
          }
        }
      }
    }
    iter = property.find("os_version");
    if(iter != property.end()){
      string os_version_values = iter->second;
      iter = property.find("kernel_version");
      if(iter == property.end()){
        cerr << "kernel version field missing" << endl;
        return -1;
      }
      string kernel_version_values = iter->second;
      vector<string> os_version_vector = str_split(os_version_values, ",");
      vector<string> kernel_version_vector = str_split(kernel_version_values, ",");
      
      ifstream os_version_read("/etc/os-release");
      string os_actual = "";
      string os_file_line;
      bool os_version_correct = false;
      bool os_version_found_in_system = false;
      while(getline(os_version_read, os_file_line)){
        if(strcasestr(os_file_line.c_str(), "pretty_name") != nullptr){
          os_version_found_in_system = true;
          os_actual = os_file_line.substr(13, os_file_line.length() - 14);        
          vector<string>::iterator os_iter;
          for(os_iter = os_version_vector.begin(); os_iter != os_version_vector.end(); os_iter++){
            if(strcmp(os_iter->c_str(), os_actual.c_str()) == 0){
              os_version_correct = true;
              break;
            }
          }
          if(os_version_correct == true)
            break;
        }
      }
      if(os_version_found_in_system == false){
        cerr << "Unable to locate actual OS installed" << endl;
      }
      
      struct utsname kernel_version_struct ;
      if(uname(&kernel_version_struct) != 0)
        cerr << "Unable to read kernel version" << endl;
    
      //cout << kernel_version_struct.release << endl ;
      string kernel_actual = kernel_version_struct.release ;
      bool kernel_version_correct = false;
      
      vector<string>::iterator kernel_iter;
      for(kernel_iter = kernel_version_vector.begin() ; kernel_iter != kernel_version_vector.end(); kernel_iter++)
        if(kernel_actual.compare(*kernel_iter) == 0 ){
          kernel_version_correct = true;
          break;
        }
      string result = "[rcqt] kernelcheck " + os_actual + " " + kernel_actual + " " + (os_version_correct && kernel_version_correct ? "pass" : "fail");
      log(result.c_str(), rvs::logresults) ;
    }
    
    return 0;
}