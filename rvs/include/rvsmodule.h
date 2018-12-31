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
#ifndef RVS_INCLUDE_RVSMODULE_H_
#define RVS_INCLUDE_RVSMODULE_H_

#include <map>
#include <utility>
#include <string>
#include <memory>

#include "yaml-cpp/yaml.h"

#include "include/rvsmodule_if.h"

namespace rvs {

class ifbase;
class action;


/**
 * @class module
 * @ingroup Launcher
 *
 * @brief RVS module manager class
 *
 */
class module {
//! module name - module instance pair
typedef std::pair<std::string, module*> t_mmpair;

  // collection related members
 public:
  static int     initialize(const char* pConfigName);
  static action* action_create(const char* pModuleShortName);
  static int     action_destroy(action*);
  static int     terminate();
  static void    do_list_modules(void);

 protected:
  static module* find_create_module(const char* pShortName);

  //! YAML configuration
  static YAML::Node  config;

  //! short name -> rvsmodule* mapping
  static std::map<std::string, module*> modulemap;

  //! short name -> .so filename mapping
  static std::map<std::string, std::string> filemap;

 protected:
  module(const char* pModuleName, void* pSoLib);
  //! Destructor
  virtual ~module();

 protected:
  int     initialize();
  int     terminate_internal();

  void*   action_create();
  int     action_destroy_internal(action*);

  int     init_interfaces(void);
  int     init_interface_method(void** ppfunc, const char* pMethodName);
  int     init_interface_0(void);
  int     init_interface_1(void);

 protected:
  //! collection of interfaces supported by this module
  std::map<int, std::shared_ptr<ifbase>> ifmap;
  //! Pointer to shared object library obtained through dlopnen() call
  void*       psolib;
  //! Module name
  std::string name;

  //! Pointer to module init function
  t_rvs_module_init           rvs_module_init;
  //! Pointer to module terminate function
  t_rvs_module_terminate      rvs_module_terminate;
  //! Pointer to module action create function
  t_rvs_module_action_create  rvs_module_action_create;
  //! Pointer to module action destroy function
  t_rvs_module_action_destroy rvs_module_action_destroy;
  //! Pointer to module function checking interface existance
  t_rvs_module_has_interface   rvs_module_has_interface;
};

}  // namespace rvs


#endif  // RVS_INCLUDE_RVSMODULE_H_
