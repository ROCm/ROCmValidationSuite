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
#ifndef RVS_INCLUDE_RVSEXEC_H_
#define RVS_INCLUDE_RVSEXEC_H_

#include <string>
#include "yaml-cpp/node/node.h"


namespace rvs {

class if1;

/**
 * @class exec
 * @ingroup Launcher
 *
 * @brief Command executor class.
 *
 * Execute functionality based on command line and the contents of .conf file.
 *
 */

class exec {
 public:
  exec();
  ~exec();

  int run();

 protected:
  void  do_help(void);
  void  do_version(void);
  int   do_gpu_list(void);

  int   do_yaml(const std::string& config_file);
  int   do_yaml_properties(const YAML::Node& node,
                           const std::string& module_name, if1* pif1);
  bool  is_yaml_properties_collection(const std::string& module_name,
                                      const std::string& proprty_name);
  int   do_yaml_properties_collection(const YAML::Node& node,
                                      const std::string& parent_name,
                                      if1* pif1);
};

}  // namespace rvs

#endif  // RVS_INCLUDE_RVSEXEC_H_
