/********************************************************************************
 *
 * Copyright (c) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef RCQT_SO_INCLUDE_ACTION_H_
#define RCQT_SO_INCLUDE_ACTION_H_

#include "include/rvsactionbase.h"
#include "include/rvs_module.h"
#include "include/rvs_util.h"
#include "include/rcutils.h"


/**
 * @class rcqt_action
 * @ingroup RCQT
 *
 * @brief RCQT action implementation class
 *
 * Derives from rvs::actionbase and implements actual action functionality
 * in its run() method.
 *
 */
class rcqt_action : public rvs::actionbase {
  public:
    rcqt_action();
    virtual ~rcqt_action();
    virtual int run(void);
    static void cleanup_logs();
  protected:
    //! json_rcqt_node is json node shared through submodules
    void *json_rcqt_node;

    std::string PACKAGELIST; 
    /**
     *  @brief Function used in rcqt action class to check for given package
     */
    virtual int pkgchk_run();

    /**
     *  @brief Function used in rcqt action class to list installed packages and its version.
     */
    virtual int pkglist_run();
};

#endif  // RCQT_SO_INCLUDE_ACTION_H_
