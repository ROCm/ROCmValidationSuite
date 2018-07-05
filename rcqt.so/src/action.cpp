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
#include "action_pkgchk.h"
#include "action_usrchk.h"
#include "action_kernelchk.h"
#include "action_ldcfgchk.h"
#include "action_filechk.h"

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
#include "rvs_util.h"
#include "rvsactionbase.h"


#define BUFFER_SIZE 3000

using namespace std;


action::action() {
}

action::~action() {
  property.clear();
}

int action::property_set(const char* Key, const char* Val) {
  return rvs::actionbase::actionbase::property_set(Key, Val);
}

void action::check_property(string field_name, bool &return_bool){
  map<string, string>::iterator iter;
  iter = property.find(field_name);
  if(iter == property.end())
    return_bool = false;
}

int action::run()
{
    int return_value = 0;
    //map<string, string>::iterator iter;
    
    bool pkgchk_bool = true;
    bool usrchk_bool = true;
    bool kernelchk_bool = true;
    bool ldcfgchk_bool = true;
    bool filechk_bool = true;
    
    // check if package check action is going to trigger   
    check_property("package", pkgchk_bool);    
    
    if(pkgchk_bool == true)
      return_value += pkgchk_run(property);
    
    // check if usrer check action is going to trigger
    check_property("user", usrchk_bool);
    
    if(usrchk_bool == true)
      return_value += usrchk_run(property);
    
    // chck if kernel version action is going to trigger
    check_property("os_version", kernelchk_bool);
    check_property("kernel_version", kernelchk_bool);
    
    if(kernelchk_bool == true)
      return_value += kernelchk_run(property);
    
    // check if ldcfg check action is going to trigger
    check_property("soname", ldcfgchk_bool);
    check_property("arch", ldcfgchk_bool);
    check_property("ldpath", ldcfgchk_bool);
    
    if(ldcfgchk_bool == true)
      return_value += ldcfgchk_run(property);
   
    //check if file check action is going to trigger
    check_property("file", filechk_bool);
    
    if(filechk_bool == true)
      return_value += filechk_run(property);
    
        
    return return_value < 0 ? -1 : 0;
}