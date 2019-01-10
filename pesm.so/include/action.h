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
#ifndef PESM_SO_INCLUDE_ACTION_H_
#define PESM_SO_INCLUDE_ACTION_H_

#include <string>
#include <vector>

#include "include/rvsactionbase.h"

/**
 * @class pesm_action
 * @ingroup PESM
 *
 * @brief PESM action implementation class
 *
 * Derives from rvs::actionbase and implements actual action functionality
 * in its run() method.
 *
 */
class pesm_action : public rvs::actionbase {
 public:
  pesm_action();
  virtual ~pesm_action();

  virtual int run(void);

 protected:
  int do_gpu_list(void);
  bool get_all_common_config_keys(void);
  bool get_all_pesm_config_keys(void);

 protected:
  //! json logging flag
  bool bjson;
  //! debug wait helper
  int prop_debugwait;
  //! 'true' if monitoring is to be initiated
  bool prop_monitor;
};

#endif  // PESM_SO_INCLUDE_ACTION_H_
