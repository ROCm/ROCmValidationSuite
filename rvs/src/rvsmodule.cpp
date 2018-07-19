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
#include "rvsmodule.h"

#include <dlfcn.h>
#include <stdio.h>

#include <utility>
#include <string>
#include <map>
#include <memory>
#include <iostream>
#include <fstream>

#include "rvsliblogger.h"
#include "rvsif0.h"
#include "rvsif1.h"
#include "rvsaction.h"
#include "rvsliblog.h"
#include "rvsoptions.h"


std::map<std::string, rvs::module*> rvs::module::modulemap;
std::map<std::string, std::string>  rvs::module::filemap;
YAML::Node rvs::module::config;

using std::string;

/**
 * @brief Constructor
 *
 * @param pModuleName Module name
 * @param pSoLib .so library which implements module
 *
 */
rvs::module::module(const char* pModuleName, void* pSoLib)
:
psolib(pSoLib),
name(pModuleName) {
}

//! Destructor
rvs::module::~module() {
}

/**
 * @brief Module manager initialization method
 *
 * Reads module name -> .so file mapping from configuration file name
 * specified by pConfig
 *
 * @param pConfig Name of configuration file
 * @return 0 - success, non-zero otherwise
 *
 */
int rvs::module::initialize(const char* pConfig) {
  // Check if pConfig file exists
  std::ifstream file(pConfig);

  if (!file.good()) {
    std::cerr << "ERROR: " << pConfig << " file does not exist.\n";
    return -1;
  } else {
    file.close();
  }

  // load list of supported modules from config file
  YAML::Node config = YAML::LoadFile(pConfig);

  // verify that that the file format is supported
  YAML::const_iterator it = config.begin();
  if (it == config.end()) {
    std::cerr << "ERROR: unsupported file format. Version string not found."
              << std::endl;
    return -1;
  }

    std::string key = it->first.as<std::string>();
    std::string value = it->second.as<std::string>();

  if (key != "version") {
    std::cerr << "ERROR: unsupported file format. Version string not found."
              << std::endl;
    return -1;
  }

  if (value != "1") {
    std::cerr << "ERROR: version is not 1" << std::endl;
    return -1;
  }

  // load nam-file pairs:
  for (it++; it != config.end(); ++it) {
    key = it->first.as<std::string>();
    value = it->second.as<std::string>();
    filemap.insert(std::pair<string, string>(key, value));
  }

  return 0;
}

/**
 * @brief Given module name, return pointer to module instance
 *
 * This method will load and initialize module if needed.
 *
 * @param name Name of configuration file
 * @return Pointer to module instance
 *
 */
rvs::module* rvs::module::find_create_module(const char* name) {
  module* m = nullptr;

  // find module based on short name
  auto it = modulemap.find(std::string(name));

  // not found...
  if (it == modulemap.end()) {
    // ... try opening .so

    // first find proper .so filename
    auto it = filemap.find(std::string(name));

    // not found...
    if (it == filemap.end()) {
      // this should never happen if .config is OK
      std::cerr << "ERROR: module '" << name << "' not found in configuration."
                << std::endl;
      return NULL;
    }

    // open .so
    string libpath;
    if (rvs::options::has_option("-m", libpath)) {
      libpath += "/";
    } else {
      rvs::options::has_option("pwd", libpath);
      libpath += "/";
    }
    string sofullname(libpath + it->second);
    void* psolib = dlopen(sofullname.c_str(), RTLD_NOW);

    // error?
    if (!psolib) {
      std::cerr << "ERROR: could not load .so '" << sofullname << "' reason: "
                << dlerror() << std::endl;
      return NULL;  // fail
    }

    // create module object
    m = new rvs::module(name, psolib);
    if (!m) {
      dlclose(psolib);
      return NULL;
    }

    // initialize API function pointers
    if (m->init_interfaces()) {
      std::cerr << "ERROR: could not init interfaces for '"
                << it->second.c_str() << "'" << std::endl;
      dlclose(psolib);
      delete m;
      return nullptr;
    }

    // initialize newly loaded module
    if (m->initialize()) {
      std::cerr << "ERROR: could not initialize '" << it->second.c_str() << "'"
                << std::endl;
      dlclose(psolib);
      delete m;
      return nullptr;
    }

    // add to map
    modulemap.insert(t_mmpair(name, m));
  } else {
    m = it->second;
  }

  return m;
}

/**
 * @brief Module instance initialization method
 *
 * Fills module initialization structure with pointers to
 * Logger API and passes it to module Initialize() API.
 *
 * @return 0 - success, non-zero otherwise
 *
 */
int rvs::module::initialize() {
  T_MODULE_INIT d;

  d.cbLog             = rvs::logger::Log;
  d.cbLogExt          = rvs::logger::LogExt;
  d.cbLogRecordCreate = rvs::logger::LogRecordCreate;
  d.cbLogRecordFlush   = rvs::logger::LogRecordFlush;
  d.cbCreateNode      = rvs::logger::CreateNode;
  d.cbAddString       = rvs::logger::AddString;
  d.cbAddInt          = rvs::logger::AddInt;
  d.cbAddNode         = rvs::logger::AddNode;

  return (*rvs_module_init)(reinterpret_cast<void*>(&d));
}


/**
 * @brief Given module name, create module action
 *
 * @param name Module name
 * @return Pointer to action instance created in module
 *
 */
rvs::action* rvs::module::action_create(const char* name) {
  // find module
  rvs::module* m = module::find_create_module(name);
  if (!m) {
    std::cerr << "ERROR: module '" << name << "' not available." << std::endl;
    return nullptr;
  }

  // create lib action objct
  void* plibaction = m->action_create();
  if (!plibaction)  {
    std::cerr << "ERROR: module '" << name << "' could not create lib action."
         << std::endl;
    return nullptr;
  }

  // create action proxy object
  rvs::action* pa = new rvs::action(name, plibaction);
  if (!pa) {
    std::cerr << "ERROR: module '" << name << "' could not create action proxy."
              << std::endl;
    return nullptr;
  }

  // create interfaces for the proxy
  // clone from module and assign libaction ptr

  for (auto it = m->ifmap.begin(); it != m->ifmap.end(); it++) {
    std::shared_ptr<rvs::ifbase> sptrif(it->second->clone());
    sptrif->plibaction = plibaction;
    pa->ifmap.insert(rvs::action::t_impair(it->first, sptrif));
  }

  return pa;
}

/**
 * @brief Create module action
 *
 * Note: internal, used by static action_create()
 *
 * @return Pointer to action instance created in module
 *
 */
void* rvs::module::action_create() {
  return (*rvs_module_action_create)();
}


/**
 * @brief Destroy module action
 *
 * Note: after this call, proxy action is also destroyed and its
 * pointer can no
 * @param paction Pointer to action proxy instance
 * @return 0 - success, non-zero otherwise
 *
 */
int rvs::module::action_destroy(rvs::action* paction) {
  // find module
  rvs::module* m = module::find_create_module(paction->name.c_str());
  if (!m)
    return -1;

  return m->action_destroy_internal(paction);
}

/**
 * @brief Destroy module action
 *
 * Note: internal, used by static action_destroy()
 *
 * @param paction Pointer to action proxy instance
 * @return 0 - success, non-zero otherwise
 *
 */
int rvs::module::action_destroy_internal(rvs::action* paction) {
  int sts = (*rvs_module_action_destroy)(paction->plibaction);
  delete paction;

  return sts;
}

/**
 * @brief Cleanup module manager
 *
 * @return 0 - success, non-zero otherwise
 *
 */
int rvs::module::terminate() {
  for (auto it = rvs::module::modulemap.begin();
       it != rvs::module::modulemap.end(); it++) {
    it->second->terminate_internal();
    dlclose(it->second->psolib);
    delete it->second;
  }

  modulemap.clear();

  return 0;
}

/**
 * @brief Cleanup module instance
 *
 * @return 0 - success, non-zero otherwise
 *
 */
int rvs::module::terminate_internal() {
  return (*rvs_module_terminate)();
}


/**
 * @brief Init module interfaces
 *
 * Module interfaces are initialized upon loading of module.
 * Pointer to starndard RVS API functions are obtained by calling ldsym()
 * given API function names. Action proxy interfaces are also created at this
 * point and later cloned into particular action proxy upon creation of action.
 *
 * @return 0 - success, non-zero otherwise
 *
 */
int rvs::module::init_interfaces() {
  // init global helper methods for this library
  if (init_interface_method(
    reinterpret_cast<void**>(&rvs_module_init), "rvs_module_init"))
    return -1;

  if (init_interface_method(
    reinterpret_cast<void**>(&rvs_module_terminate), "rvs_module_terminate"))
    return -1;

  if (init_interface_method(
    reinterpret_cast<void**>(&rvs_module_action_create),
                            "rvs_module_action_create"))
    return -1;

  if (init_interface_method(
    reinterpret_cast<void**>(&rvs_module_action_destroy),
                            "rvs_module_action_destroy"))
    return -1;

  if (init_interface_0())
    return -1;

  if (init_interface_1())
    return -1;

  return 0;
}

/**
 * @brief Initialize function pointer
 *
 * Initialize function pointer by searching function by name in .so library
 *
 * @param ppfunc pointer to function pointer to be initialized
 * @param pMethodName Method name
 *
 * @return 0 - success, non-zero otherwise
 *
 */
int rvs::module::init_interface_method(void** ppfunc, const char* pMethodName) {
  if (!psolib) {
    std::cerr << "ERROR: psolib is null. " << std::endl;
    return -1;
  }
  void* pf = dlsym(psolib, pMethodName);
  if (!pf) {
    std::cerr << "ERROR: could not find .so method '" << pMethodName << "'"
              << std::endl;
  }

  *ppfunc = pf;

  return 0;
}

/**
 * @brief Init RVS IF0 interfaces
 *
 * @return 0 - success, non-zero otherwise
 *
 */
int rvs::module::init_interface_0(void) {
  rvs::if0* pif0 = new rvs::if0();
  if (!pif0)
    return -1;

  int sts = 0;

  if (init_interface_method(
    reinterpret_cast<void**>(&(pif0->rvs_module_get_version)),
                            "rvs_module_get_version"))
    sts--;

  if (init_interface_method(
    reinterpret_cast<void**>(&(pif0->rvs_module_get_name)),
                            "rvs_module_get_name"))
    sts--;

  if (init_interface_method(
    reinterpret_cast<void**>(&(pif0->rvs_module_get_description)),
                            "rvs_module_get_description"))
    sts--;

  if (init_interface_method(
    reinterpret_cast<void**>(&(pif0->rvs_module_has_interface)),
                            "rvs_module_has_interface"))
    sts--;

  if (init_interface_method(
    reinterpret_cast<void**>(&(pif0->rvs_module_get_config)),
                            "rvs_module_get_config"))
    sts--;

  if (init_interface_method(
    reinterpret_cast<void**>(&(pif0->rvs_module_get_output)),
                            "rvs_module_get_output"))
    sts--;

  if (sts) {
    delete pif0;
    return sts;
  }

  std::shared_ptr<rvs::ifbase> sptr((rvs::ifbase*)pif0);
  ifmap.insert(rvs::action::t_impair(0, sptr));

  return 0;
}

/**
 * @brief Init RVS IF1 interfaces
 *
 * @return 0 - success, non-zero otherwise
 *
 */
int rvs::module::init_interface_1(void) {
  rvs::if1* pif1 = new rvs::if1();
  if (!pif1)
    return -1;

  int sts = 0;
  if (init_interface_method(
    reinterpret_cast<void**>(&(pif1->rvs_module_action_property_set)),
                            "rvs_module_action_property_set"))
    sts--;

  if (init_interface_method(
    reinterpret_cast<void**>(&(pif1->rvs_module_action_run)),
                            "rvs_module_action_run"))
    sts--;

  if (init_interface_method(
    reinterpret_cast<void**>(&(pif1->rvs_module_get_errstring)),
                            "rvs_module_get_errstring"))
    sts--;

  if (sts) {
    delete pif1;
    return sts;
  }

  std::shared_ptr<rvs::ifbase> sptr((rvs::ifbase*)pif1);
  ifmap.insert(rvs::action::t_impair(1, sptr));

  return 0;
}

/**
 * @brief Lists available modules
 *
 */
void rvs::module::do_list_modules(void) {
  // for all modules
  for (auto it = filemap.begin(); it != filemap.end(); it++) {
    // create action
    rvs::action* pa = rvs::module::action_create(it->first.c_str());
    if (!pa) {
      std::cerr << "ERROR: could not open module: " << it->first.c_str()
                << std::endl;
      continue;
    }

    // output module name
    std::cout << "\t" << it->first.c_str() << ":" << std::endl;

    // obtain IF0
    rvs::if0* pif0 = dynamic_cast<rvs::if0*>(pa->get_interface(0));
    if (!pif0) {
      // action no longer needed so destroy it
      rvs::module::action_destroy(pa);

      std::cerr << "ERROR: could not obtain data." << std::endl;
      continue;
    }

    // output info
    std::cout << "\t\tDescription: " << pif0->get_description() << std::endl;
    std::cout << "\t\tconfig: " << pif0->get_config() << std::endl;
    std::cout << "\t\toutput: " << pif0->get_output() << std::endl;

    // action no longer needed so destroy it
    rvs::module::action_destroy(pa);
  }
}






