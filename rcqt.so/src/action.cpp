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
#include <stdlib.h>
#include <map>
#include <sys/types.h>
#include <unistd.h>
#include <pwd.h>
#include <grp.h>
#include <string.h>

#include <vector>

#include "rvs_module.h"
#include "rvsliblogger.h"


#define BUFFER_SIZE 3000

using namespace std;


action::action() {
}

action::~action() {
  property.clear();
}

int action::property_set(const char* Key, const char* Val) {
  return rvs::lib::actionbase::property_set(Key, Val);
}

void action::split_string(vector <string> &group_array,char delimiter, string string_of_groups)
{
  size_t found = string_of_groups.find_first_of(delimiter);
  while( found != string::npos){
    group_array.push_back(string_of_groups.substr(0, found - 1));
    string_of_groups = string_of_groups.substr(found, string_of_groups.size());
  }
  group_array.push_back(string_of_groups);
  
  for(int i = 0 ; i < group_array.size(); i++)
    cout << group_array[i] << endl;
}

int action::run()
{
    string test = "jedan,dva,tri,cetiri";
    vector<string>vector_test;
    split_string(vector_test, ',', test);
    
    
    
    /*map<string, string>::iterator iter;
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
    if(iter == property.end()){
        
      bool group_exists = false;
      bool user_is_in_all_groups = true;
      string user_name = iter->second;
      string group_name;
      iter = property.find("group");
      if(iter != property.end()){
        group_name = iter->second;
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
      vector<string> group_array;
      
      
      if(group_exists == true){
        if((g = getgrnam(group_name.c_str())) == nullptr){
            cerr << "group doesn't exist" << endl;
            
        }
      }   
      
      //string user_is_in_group = "[rcqt] 
    
      int i;
      int j=0;
      if(group_exists){
        for(i=0; g->gr_mem[i]!=NULL; i++){
            if(strcmp(g->gr_mem[i], group_name.c_str()) == 0){
                //log("[rcqt] usercheck
                printf("user is in group\n");
                j=1;
                break;
            }
        }
      }
        
      if(j==0) printf("user is not in the group\n");
    
    }*/
    
    return 0;
}