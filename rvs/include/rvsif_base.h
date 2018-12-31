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
#ifndef RVS_INCLUDE_RVSIF_BASE_H_
#define RVS_INCLUDE_RVSIF_BASE_H_

#include "include/rvsmodule_if.h"

namespace rvs {

/**
 * @class ifbase
 * @ingroup Launcher
 *
 * @brief Base class for RVS interfaces.
 *
 */
class ifbase {
 public:
  //! Dfault destructor
  virtual ~ifbase();
  virtual int    has_interface(int);

 protected:
  //! Default constructor
  ifbase();
  //! Copy constructor
  ifbase(const ifbase& rhs);

  virtual ifbase& operator=(const ifbase& rhs);
  //! pure virtual function to enforce support for interface cloning
  virtual ifbase* clone(void) = 0;

 protected:
  //! Pointer to action instance in an RVS module
  void*  plibaction;
  //! Pointer to module function checking interface existance
  t_rvs_module_has_interface   rvs_module_has_interface;

//! Factory class
friend class module;
};

}  // namespace rvs

#endif  // RVS_INCLUDE_RVSIF_BASE_H_
