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
#include <iostream>
#include <memory>
#include <string>
#include <algorithm>

#include "include/rvsexec.h"
#include "yaml-cpp/yaml.h"

#include "include/rvsif0.h"
#include "include/rvsif1.h"
#include "include/rvsaction.h"
#include "include/rvsmodule.h"
#include "include/rvsliblogger.h"
#include "include/rvsoptions.h"
#include "include/rvs_util.h"

#define MODULE_NAME_CAPS "CLI"

/*** Example rvs.conf file structure

actions:
- name: action_1
  device: all
  module: gpup
  properties:
    mem_banks_count:
  io_links-properties:
    version_major:
- name: action_2
  module: gpup
  device: all

***/


using std::string;

/**
 * @brief Executes actions listed in .conf file.
 *
 * @return 0 if successful, non-zero otherwise
 *
 */
int rvs::exec::do_yaml(const std::string& config_file) {
  int sts = 0;

  YAML::Node config = YAML::LoadFile(config_file);

  // find "actions" map
  const YAML::Node& actions = config["actions"];

  // for all actions...
  for (YAML::const_iterator it = actions.begin(); it != actions.end(); ++it) {
    const YAML::Node& action = *it;

    // if stop was requested
    if (rvs::logger::Stopping()) {
      return -1;
    }

    // find module name
    std::string rvsmodule;
    try {
      rvsmodule = action["module"].as<std::string>();
    } catch(...) {
    }

    // not found or empty
    if (rvsmodule == "") {
      // report error and go to next action
      char buff[1024];
      snprintf(buff, sizeof(buff), "action '%s' does not specify module.",
               action["name"].as<std::string>().c_str());
      rvs::logger::Err(buff, MODULE_NAME_CAPS);
      return -1;
    }

    // create action excutor in .so
    rvs::action* pa = module::action_create(rvsmodule.c_str());
    if (!pa) {
      char buff[1024];
      snprintf(buff, sizeof(buff),
               "action '%s' could not crate action object in module '%s'",
               action["name"].as<std::string>().c_str(),
               rvsmodule.c_str());
      rvs::logger::Err(buff, MODULE_NAME_CAPS);
      return -1;
    }

    if1* pif1 = dynamic_cast<if1*>(pa->get_interface(1));
    if (!pif1) {
      char buff[1024];
      snprintf(buff, sizeof(buff),
               "action '%s' could not obtain interface if1",
               action["name"].as<std::string>().c_str());
      module::action_destroy(pa);
      return -1;
    }

    // load action properties from yaml file
    sts += do_yaml_properties(action, rvsmodule, pif1);
    if (sts) {
      module::action_destroy(pa);
      return sts;
    }

    // set also command line options:
    for (auto clit = rvs::options::get().begin();
         clit != rvs::options::get().end(); ++clit) {
      std::string p(clit->first);
      p = "cli." + p;
      pif1->property_set(p, clit->second);
    }

    // execute action
    sts = pif1->run();

    // processing finished, release action object
    module::action_destroy(pa);

    // errors?
    if (sts) {
      // cancel actions and return
      return sts;
    }
  }

  return 0;
}

/**
 * @brief Loads action properties.
 *
 * @return 0 if successful, non-zero otherwise
 *
 */
int rvs::exec::do_yaml_properties(const YAML::Node& node,
                                  const std::string& module_name,
                                  rvs::if1* pif1) {
  int sts = 0;

  string indexes;
  bool indexes_provided = false;
  if (rvs::options::has_option("-i", &indexes) && (!indexes.empty()))
    indexes_provided = true;


  // for all child nodes
  for (YAML::const_iterator it = node.begin(); it != node.end(); it++) {
    // if property is collection of module specific properties,
    if (is_yaml_properties_collection(module_name,
        it->first.as<std::string>())) {
      // pass properties collection to .so action object
      sts += do_yaml_properties_collection(it->second,
                                           it->first.as<std::string>(),
                                           pif1);
    } else {
      // just set this one propertiy
      if (indexes_provided && it->first.as<std::string>() == "device") {
        std::replace(indexes.begin(), indexes.end(), ',', ' ');
        sts += pif1->property_set("device", indexes);
      } else {
        sts += pif1->property_set(it->first.as<std::string>(),
                                it->second.as<std::string>());
      }
    }
  }

  return sts;
}

/**
 * @brief Loads property collection for collection type node in .conf file.
 *
 * @return 0 if successful, non-zero otherwise
 *
 */
int rvs::exec::do_yaml_properties_collection(const YAML::Node& node,
                                             const std::string& parent_name,
                                             if1* pif1) {
  int sts = 0;

  // for all child nodes
  for (YAML::const_iterator it = node.begin(); it != node.end(); it++) {
    // prepend dot separated parent name and pass property to module
    sts += pif1->property_set(parent_name + "." + it->first.as<std::string>(),
    it->second.IsNull() ? std::string("") : it->second.as<std::string>());
  }

  return sts;
}

/**
 * @brief Checks if property is collection type property in .conf file.
 *
 * @param module_name module name
 * @param property_name property name
 * @return 'true' if property is collection, 'false' otherwise
 *
 */
bool rvs::exec::is_yaml_properties_collection(
  const std::string& module_name,
  const std::string& property_name) {
  if (module_name == "gpup") {
    if (property_name == "properties")
      return true;

    if (property_name == "io_links-properties")
      return true;
  } else {
    if (module_name == "peqt") {
      if (property_name == "capability") {
        return true;
      }
    } else {
        if (module_name == "gm") {
            if (property_name == "metrics")
            return true;
    }
    }
  }

  return false;
}

