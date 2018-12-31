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
#ifndef RVS_INCLUDE_RVSIF0_H_
#define RVS_INCLUDE_RVSIF0_H_


#include "include/rvsmodule_if0.h"
#include "include/rvsif_base.h"

namespace rvs {

/**
 * @class if0
 * @ingroup Launcher
 *
 * @brief RVS IF0 interface
 *
 */
class if0 : public ifbase {
 public:
  virtual ~if0();
  virtual const char*  get_description(void);
  virtual const char*  get_config(void);
  virtual const char*  get_output(void);

 protected:
  if0();
  if0(const if0&);

  virtual if0& operator= (const if0& rhs);
  virtual ifbase* clone(void);

 protected:
  //! Pointer to module function returning module description
  t_rvs_module_get_description rvs_module_get_description;
  //! Pointer to module function returning configuration info
  t_rvs_module_get_config      rvs_module_get_config;
  //! Pointer to module function returning output info
  t_rvs_module_get_output      rvs_module_get_output;

friend class module;
};

}  // namespace rvs

#endif  // RVS_INCLUDE_RVSIF0_H_
