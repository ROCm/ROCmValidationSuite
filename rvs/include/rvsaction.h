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
#ifndef RVS_INCLUDE_RVSACTION_H_
#define RVS_INCLUDE_RVSACTION_H_

#include <memory>
#include <map>
#include <string>
#include <utility>

namespace rvs {

class ifbase;
class module;

/**
 * @class action
 * @ingroup Launcher
 *
 * @brief Proxy class for RVS module actions
 *
 * Maintains RVS interaces to various action functionalities.
 *
 */
class action {
typedef std::pair< int, std::shared_ptr<ifbase> > t_impair;

 public:
  virtual rvs::ifbase*     get_interface(int);

 protected:
  action(const char* pName, void* pLibAction);
  virtual ~action();

 protected:
  //! action name as defined in YAML .conf file for this action
  std::string name;
  //! pointer to actual action object created in and RVS module
  void*       plibaction;
  //! list of RVS interfaces supported by this action
  std::map< int, std::shared_ptr<ifbase> > ifmap;

  //! factory class
friend class module;
};

}  // namespace rvs


#endif  // RVS_INCLUDE_RVSACTION_H_
