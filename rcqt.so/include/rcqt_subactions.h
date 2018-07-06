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
#ifndef INCLUDE_SUBACTIONS_H_
#define INCLUDE_SUBACTIONS_H_

#include <map>
#include <string>

#include "action.h"
#include "rvs_module.h"
#include "rvsliblogger.h"
#include "rvs_util.h"
#include "rvsactionbase.h"

/**
 *  @brief Function used in rcqt action class to check for given user and group membership
 */

extern int usrchk_run(std::map<std::string,std::string> property);

/**
 *  @brief Function used in rcqt action class to check for given package
 */

extern int pkgchk_run(std::map<std::string,std::string> property);

/**
 *  @brief Function used in rcqt action class to check for os and kernel version
 */

extern int kernelchk_run(std::map<std::string,std::string> property);

/**
 *  @brief Function used in rcqt action class to check for o shared library existance and architecture
 */
extern int ldcfgchk_run(std::map<std::string,std::string> property);

#endif  // INCLUDE_SUBACTIONS_H_