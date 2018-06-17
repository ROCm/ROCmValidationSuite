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
#ifndef RVSIF1_H_
#define RVSIF1_H_

#include <string>

#include "rvsmodule_if1.h"
#include "rvsif_base.h"

using namespace std;

namespace rvs 
{

class if1 : public ifbase {

public:
  virtual ~if1();
  virtual int   property_set(const char*, const char*);
  virtual int   property_set(const string&, const string&);
  virtual int   run(void);
  virtual char* get_errstring(int);

protected:
  if1();
  if1(const if1&);

virtual if1& operator= (const if1& rhs);
virtual ifbase* clone(void);

protected:
  t_rvs_module_action_property_set  rvs_module_action_property_set;
  t_rvs_module_action_run           rvs_module_action_run;
  t_rvs_module_get_errstring        rvs_module_get_errstring;

friend class module;
};

} // namespace rvs
#endif /* RVSIF1_H_ */
