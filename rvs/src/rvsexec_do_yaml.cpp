/********************************************************************************
 *
 * Copyright (c) 2018-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <iomanip>

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
  if (!actions.IsDefined()) {
    rvs::logger::Err("Invalid configuration file !", MODULE_NAME_CAPS);
    return -1;
  }

  /* Number of times to repeat the test */
  for (int i = 0; i < num_times; i++) {

    // for all actions...
    for (YAML::const_iterator it = actions.begin(); it != actions.end(); ++it) {
      const YAML::Node& action = *it;

      sts = 0;
      rvs::logger::log("Action name :" + action["name"].as<std::string>(), rvs::logresults);

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

      // create action executor in .so
      rvs::action* pa = module::action_create(rvsmodule.c_str());
      if (!pa) {
        char buff[1024];
        snprintf(buff, sizeof(buff),
            "action '%s' could not create action object in module '%s'",
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

      // Set Callback
      if(nullptr != app_callback) {
        pif1->callback_set(&rvs::exec::action_callback, (void *)this);
      }

      // execute action
      sts = pif1->run();

      // processing finished, release action object
      module::action_destroy(pa);

      // errors?
      if (sts) {
        rvs::logger::Err("Action failed to run successfully.",
            action["module"].as<std::string>().c_str(),
            action["name"].as<std::string>().c_str());
      }
    }
  }

  /* Process the test results */


  return 0;
}

/**
 * @brief Executes actions listed in .conf file.
 *
 * @return 0 if successful, non-zero otherwise
 *
 */
int rvs::exec::do_yaml(yaml_data_type_t data_type, const std::string& data) {

  int sts = 0;
  YAML::Node config;
  rvs_results_t result = {RVS_STATUS_FAILED, RVS_SESSION_STATE_COMPLETED, (const char *)NULL};

  if(yaml_data_type_t::YAML_FILE == data_type) {

    config = YAML::LoadFile(data);
  }
  else if(yaml_data_type_t::YAML_STRING == data_type) {

    config = YAML::Load(data);
  }
  else {
    return -1;
  }

  // find "actions" map
  const YAML::Node& actions = config["actions"];
  if (!actions.IsDefined()) {
    rvs::logger::Err("Invalid configuration file !", MODULE_NAME_CAPS);
    return -1;
  }

  const char boundary = '|';

  // Header columns
  std::string header = "ROCm Validation Suite (RVS) Summary";
  std::string header1 = "Action Name";
  std::string header2 = "Module";
  std::string header3 = "Result";

  // Define column width for consistent spacing
  int columnWidth = 12;
  int actionColumnWidth = 30;
  int TotalColumnWidth = (actionColumnWidth + 2 * columnWidth + 8);

  // Function to print a horizontal boundary line
  auto printBoundary = [&]() {
    std::cout << "+";
    for (int i = 0; i < (actionColumnWidth + 2 * columnWidth + 8); ++i) {
      std::cout << '-';
    }
    std::cout << "+" << std::endl;
  };

  int padding = ((TotalColumnWidth - 2) / 2) + (header.size() / 2);

  if (rvs::options::has_option("-q")) {

    // Print top boundary
    printBoundary();

    // Print header row
    std::cout << boundary << std::setw(padding) << header << std::setw(TotalColumnWidth - padding + 1) <<  boundary << std::endl;

    // Print top boundary
    printBoundary();

    // Print header row
    std::cout << boundary << " "
      << std::setw(actionColumnWidth) << std::left << header1
      << " | " << std::setw(columnWidth) << std::left << header2
      << " | " << std::setw(columnWidth) << std::left << header3
      << " " << boundary << std::endl;

    // Print inner boundary
    printBoundary();
  }

  /* Number of times to execute the test */
  for (int i = 0; i < num_times; i++) {

    // for all actions...
    for (YAML::const_iterator it = actions.begin(); it != actions.end(); ++it) {
      const YAML::Node& action = *it;

      sts = 0;
      rvs::logger::log("Action name :" + action["name"].as<std::string>(), rvs::logresults);

      // if stop was requested
      if (rvs::logger::Stopping()) {
        char buff[1024];
        snprintf(buff, sizeof(buff),
            "action '%s' was requested to stop",
            action["name"].as<std::string>().c_str());
        result.output_log = buff;
        callback(&result);
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

        result.output_log = buff;
        callback(&result);
        return -1;
      }

      // create action executor in .so
      rvs::action* pa = module::action_create(rvsmodule.c_str());
      if (!pa) {
        char buff[1024];
        snprintf(buff, sizeof(buff),
            "action '%s' could not create action object in module '%s'",
            action["name"].as<std::string>().c_str(),
            rvsmodule.c_str());
        rvs::logger::Err(buff, MODULE_NAME_CAPS);
        result.output_log = buff;
        callback(&result);
        return -1;
      }

      if1* pif1 = dynamic_cast<if1*>(pa->get_interface(1));
      if (!pif1) {
        char buff[1024];
        snprintf(buff, sizeof(buff),
            "action '%s' could not obtain interface if1",
            action["name"].as<std::string>().c_str());
        module::action_destroy(pa);
        result.output_log = buff;
        callback(&result);
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

      // Set Callback
      if(nullptr != app_callback) {
        pif1->callback_set(&rvs::exec::action_callback, (void *)this);
      }

      exec_action action_info;

      action_info.name = action["name"].as<std::string>();
      action_info.module = action["module"].as<std::string>();

      std::transform(action_info.module.begin(),
          action_info.module.end(),
          action_info.module.begin(), ::toupper);

      /***********************************/

      if (rvs::options::has_option("-q")) {
        std::cout << boundary << " "
          << std::setw(actionColumnWidth) << std::left << action_info.name
          << " | " << std::setw(columnWidth) << std::left << action_info.module
          << " | " << std::setw(columnWidth + 2)  << std::right <<  boundary << std::flush;
      }
      /***********************************/

      // execute action
      sts = pif1->run();

      if (rvs::options::has_option("-q")) {
        std::string actionresult = (!sts) ? "PASS" : "FAIL";
        std::string textcolor = (!sts) ? "\033[32m" : "\033[31m";

        std::cout << "\r" << boundary << " "
          << std::setw(actionColumnWidth) << std::left << action_info.name
          << " | " << std::setw(columnWidth) << std::left << action_info.module
          << " | " << textcolor << std::setw(columnWidth) << std::left << actionresult << "\033[0m"
          << " " <<  boundary << std::endl;
      }

      // processing finished, release action object
      module::action_destroy(pa);


      // Action pass fail status !!
      // errors?
      if (sts) {

        char buff[1024];
        snprintf(buff, sizeof(buff),
            "action '%s' failed with error !",
            action["name"].as<std::string>().c_str());
        result.output_log = buff;
        callback(&result);

        rvs::logger::Err("Action failed to run successfully.",
            action["module"].as<std::string>().c_str(),
            action["name"].as<std::string>().c_str());

        action_info.result = false;
      }
      else {
        action_info.result = true;
      }
      action_details.push_back(action_info);
    }
  }

  if (!rvs::options::has_option("-q")) {

    const char boundary = '|';

    // Header columns
    std::string header = "ROCm Validation Suite (RVS) Summary";
    std::string header1 = "Action Name";
    std::string header2 = "Module";
    std::string header3 = "Result";

    // Define column width for consistent spacing
    int columnWidth = 12;
    int actionColumnWidth = 30;
    int TotalColumnWidth = (actionColumnWidth + 2 * columnWidth + 8);

    // Function to print a horizontal boundary line
    auto printBoundary = [&]() {
      std::cout << "+";
      for (int i = 0; i < (actionColumnWidth + 2 * columnWidth + 8); ++i) {
        std::cout << '-';
      }
      std::cout << "+" << std::endl;
    };

    int padding = ((TotalColumnWidth - 2) / 2) + (header.size() / 2);

    // print top boundary
    printBoundary();

    // Print header row
    std::cout << boundary << std::setw(padding) << header << std::setw(TotalColumnWidth - padding + 1) <<  boundary << std::endl;

    // Print top boundary
    printBoundary();

    // Print header row
    std::cout << boundary << " "
      << std::setw(actionColumnWidth) << std::left << header1
      << " | " << std::setw(columnWidth) << std::left << header2
      << " | " << std::setw(columnWidth) << std::left << header3
      << " " << boundary << std::endl;

    // Print inner boundary
    printBoundary();

    for(size_t i = 0; i < action_details.size(); i++) {

      std::string result = (action_details[i].result) ? "PASS" : "FAIL";
      std::string textcolor = (action_details[i].result) ? "\033[32m" : "\033[31m";

      // Print data row
      std::cout << boundary << " "
        << std::setw(actionColumnWidth) << std::left << action_details[i].name
        << " | " << std::setw(columnWidth) << std::left << action_details[i].module
        << " | " << textcolor << std::setw(columnWidth) << std::left << result << "\033[0m"
        << " " <<  boundary << std::endl;
    }

    // Print bottom boundary
    printBoundary();

#if 0
    for(size_t i = 0; i < action_details.size(); i++) {

      std::cout << "Action Name - " << action_details[i].name << std::endl;
      std::cout << "Action Module - " << action_details[i].module << std::endl;
      std::cout << "Action result - " << action_details[i].result << std::endl << std::endl;
    }
#endif
  }
  else {
    printBoundary();
  }

  result.status = RVS_STATUS_SUCCESS;
  result.output_log = "RVS session successfully completed.";
  callback(&result);
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

  rvs::logger::log("Module name :" + module_name, rvs::logresults);

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

      // just set this one property
      if (indexes_provided && it->first.as<std::string>() == "device") {

        std::replace(indexes.begin(), indexes.end(), ',', ' ');

        std::vector <uint16_t> idx;

        // parse key value into std::vector<std::string>
        auto strarray = str_split(indexes, " ");

        // convert str arary into uint16_t array
        rvs_util_strarr_to_uintarr<uint16_t>(strarray, &idx);

        // Check if indexes are gpu indexes or ids
        if(gpu_check_if_gpu_indexes (idx)) {
          sts += pif1->property_set("device_index", indexes);
          sts += pif1->property_set(it->first.as<std::string>(),
              it->second.as<std::string>());
        } else {
          sts += pif1->property_set("device", indexes);
        }
      }
      else {
        sts += pif1->property_set(it->first.as<std::string>(),
            it->second.as<std::string>());
      }
    }
  }

  string parallel;
  if (rvs::options::has_option("-p", &parallel)) {

    if (parallel == "false") {
      sts += pif1->property_set("parallel", "false");
    }
    else if (parallel == "true") {
      sts += pif1->property_set("parallel", "true");
    }
    else if (parallel.empty()) {
      sts += pif1->property_set("parallel", "true");
    }
    else {
      rvs::logger::Err("Invalid value provided for -p (--parallel) option. Valid values: true/false.");
      sts += 1;
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

