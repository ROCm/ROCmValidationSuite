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



int action::run()
{
    int return_value = 0;
    //map<string, string>::iterator iter;
    
    bool pkgchk_bool = false;
    bool usrchk_bool = false;
    bool kernelchk_os_bool = false;
    bool kernelchk_kernel_bool = false;
    bool ldcfgchk_so_bool = false;
    bool ldcfgchk_arch_bool = false;
    bool ldcfgchk_ldpath_bool = false;
    
    string dummy;
    
    // check if package check action is going to trigger
    
    pkgchk_bool =  rvs::actionbase::has_property("package", dummy);    
    
    if(pkgchk_bool == true)
      return pkgchk_run(property);
    
    // check if usrer check action is going to trigger
    usrchk_bool = rvs::actionbase::has_property("user", dummy);
    
    if(usrchk_bool == true)
      return usrchk_run(property);
    
    // chck if kernel version action is going to trigger
    kernelchk_os_bool = rvs::actionbase::has_property("os_version", dummy);
    kernelchk_kernel_bool = rvs::actionbase::has_property("kernel_version", dummy);
    
    if(kernelchk_os_bool && kernelchk_kernel_bool)
      return kernelchk_run(property);
    
    // check if ldcfg check action is going to trigger
    ldcfgchk_so_bool = rvs::actionbase::has_property("soname", dummy);
    ldcfgchk_arch_bool = rvs::actionbase::has_property("arch", dummy);
    ldcfgchk_ldpath_bool = rvs::actionbase::has_property("ldpath", dummy);
    
    if(ldcfgchk_so_bool && ldcfgchk_arch_bool && ldcfgchk_ldpath_bool)
      return ldcfgchk_run(property);    
        
    return -1;
}