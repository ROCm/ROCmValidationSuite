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

#include <map>
#include <iostream>
#include <stdlib.h>
#include <grp.h>
#include <string.h>
#include <vector>
#include <unistd.h>
#include <pwd.h>

#define USER "user"
#define GROUP "group"

using namespace std;

/**
 * Check if the user exists in the system (optional: check for the group membership )
 * @param property config file map fields
 * @return 0 - success, non-zero otherwise
 * */

int usrchk_run(std::map<string,string> property) {
  
  string user_name;
  auto iter = property.find(USER);
  if(iter != property.end()) {
    user_name = iter->second;
    bool group_exists = false;
    bool user_is_in_all_groups = true;
    string group_values_string;
    iter = property.find(GROUP);
    if(iter != property.end()) {
      group_values_string = iter->second;
      group_exists = true;
    }
  
    // Structures for checking group and user
    struct passwd *p;
    struct group *g;
    string user_exists = "[rcqt] usercheck " + user_name + " user exists";
    string user_not_exists = "[rcqt] usercheck " + user_name + " user not exists";
    
    // Check for given user
    if((p = getpwnam(user_name.c_str())) == nullptr) {
      log(user_not_exists.c_str(), rvs::logresults);
    }else {
      log( user_exists.c_str(), rvs::logresults);
    }
    if(group_exists) {      
      // Put the group list into vector
      string delimiter = ",";
      vector<string> group_vector ;
      group_vector = str_split(group_values_string, delimiter);
      
      // Check if the group exists
      for(vector<string>::iterator vector_iter = group_vector.begin(); vector_iter != group_vector.end(); vector_iter++) {
        string user_group = "[rcqt] usercheck " + user_name ;
        if((g = getgrnam(vector_iter->c_str())) == nullptr) {
          cerr << "group doesn't exist" << endl;
          return -1;
        }
        
        int i;
        int j=0;
        
        // Compare if the user group id is equal to the group id
        for(i=0; g->gr_mem[i]!=NULL; i++) {
          if(strcmp(g->gr_mem[i], user_name.c_str()) == 0) {
            user_group = user_group + " " + vector_iter->c_str() + " is member";
            log(user_group.c_str(), rvs::logresults);
            j=1;
            break;
          }
        }
        
        // If the index is 0 then we user id doesn't match the group id
        if(j==0) { 
          //printf("user is not in the group\n");
          user_group = user_group + " " + vector_iter->c_str() + " is not member";
          log(user_group.c_str(), rvs::logresults);
        }
      }
    }
    return 0;
  }
  
  
  return -1;
}