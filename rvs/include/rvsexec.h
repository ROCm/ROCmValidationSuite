/********************************************************************************
 *
 * Copyright (c) 2018-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <map>
#include "include/rvs.h"
#include "include/rvsactionbase.h"
#include "yaml-cpp/node/node.h"


namespace rvs {

class if1;

enum class yaml_data_type_t {
  YAML_FILE = 0,
  YAML_STRING = 1
};

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
  int run(std::map<std::string, std::string>& opt);

  /* Set Application callback */
  int set_callback(void (*callback)(const rvs_results_t * results, int user_param), int user_param);

  static void action_callback(const action_result_t * result, void * user_param);

  void callback(const action_result_t * result);
  void callback(const rvs_results_t * result);

 protected:
  void  do_help(void);
  void  do_version(void);
  int   do_gpu_list(void);
  int enumerate_platform();
  int   do_yaml(const std::string& config_file);
  int   do_yaml(yaml_data_type_t data_type, const std::string& data);
  int   do_yaml_properties(const YAML::Node& node,
                           const std::string& module_name, if1* pif1);
  bool  is_yaml_properties_collection(const std::string& module_name,
                                      const std::string& proprty_name);
  int   do_yaml_properties_collection(const YAML::Node& node,
                                      const std::string& parent_name,
                                      if1* pif1);

  /* Application Callback */
  void (*app_callback)(const rvs_results_t * results, int user_param);
  int user_param;
};

}  // namespace rvs

#endif  // RVS_INCLUDE_RVSEXEC_H_
