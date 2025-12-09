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
#include <thread>
#include <fstream>

#include "include/rvsexec.h"
#include "yaml-cpp/yaml.h"

#include "include/rvsif0.h"
#include "include/rvsif1.h"
#include "include/rvsaction.h"
#include "include/rvsmodule.h"
#include "include/rvsliblogger.h"
#include "include/rvsoptions.h"
#include "include/rvs_util.h"
#include "include/gpu_util.h"

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

  return 0;
}

/**
 * @brief Action test in progress spinner thread
 */
void rvs::exec::in_progress_thread(exec_action action_info) {

  const char boundary = '|';
  const int columnWidth = 14;
  const int actionColumnWidth = 32;

  const string spinner[] = {"||||", "////", "----", "\\\\\\\\"};
  int spinnerIndex = 0;

  while (in_progress) {

    std::cout << "\r" << boundary << " "
      << std::setw(actionColumnWidth) << std::left << action_info.name
      << " | " << std::setw(columnWidth) << std::left << action_info.module
      << " | " << std::setw(columnWidth)  << std::left <<  spinner[spinnerIndex++]
      << "  " << boundary << std::flush;

    spinnerIndex %= 4;

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
}

/**
 * @brief Get installed ROCm version
 */
std::string getROCmVersion(void) {

  std::string line = "N/A";
  std::ifstream inputFile("/opt/rocm/.info/version-rocm");
  if (!inputFile) {
    return line;
  }
  std::getline(inputFile, line);
  inputFile.close();

  return line;
}

/**
 * @brief Get operating system name & version.
 */
std::string getOSNameVersion(void) {

  std::ifstream file("/etc/os-release");
  std::string line;
  std::string osName;

  if (!file.is_open()) {
    return "N/A";
  }

  while (std::getline(file, line)) {

    if (line.find("PRETTY_NAME=") == 0) {
      /* Remove the key and quotes */

      /* Skip "PRETTY_NAME=" */
      osName = line.substr(12);
      if (!osName.empty() && osName.front() == '"' && osName.back() == '"') {
        /* Remove surrounding quotes */
        osName = osName.substr(1, osName.size() - 2);

        /* OS name till braces */
        size_t pos = osName.find('(');
        if (pos != std::string::npos) {
          osName = osName.substr(0, pos);
        }
      }
      break;
    }
  }

  file.close();
  return osName.empty() ? "N/A" : osName;
}


/**
 * @brief Get amdgpu driver version.
 */
std::string getAmdGpuDriverVersion() {

  std::array<char, 128> buffer;
  std::string result;

  // Run the command
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen("dkms status", "r"), pclose);
  if (!pipe) {
    return "N/A";
  }

  // Read the output
  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    result += buffer.data();
  }

  // Example output: "amdgpu/6.14.14-2204008.22.04, 6.8.0-59-generic, x86_64: installed "
  // Extract version (e.g., 6.14.14)
  size_t slashPos = result.find('/');
  size_t commaPos = result.find('-', slashPos);
  if (slashPos != std::string::npos && commaPos != std::string::npos) {
    return result.substr(slashPos + 1, commaPos - slashPos - 1);
  }

  return "N/A";
}

/**
 * @brief Print system overview details  
 */
void systemOverview() {

  // Header column
  std::string header = "System Overview";

  std::string version1 = "RVS version";
  std::string version2 = "ROCm version";
  std::string version3 = "amdgpu version";
  std::string OS = "Operating System";
  std::string gpus = "GPUs";

  // Define column width for consistent spacing
  int columnWidth = 14;;
  int actionColumnWidth = 32;
  int TotalColumnWidth = 69;

  const char boundary = '|';

  // Function to print a horizontal boundary line
  auto printBoundary = [&]() {
    std::cout << "+";
    for (int i = 0; i < TotalColumnWidth; ++i) {
      std::cout << '-';
    }
    std::cout << "+" << std::endl;
  };

  auto printDoubleBoundary = [&]() {
    std::cout << "+";
    for (int i = 0; i < TotalColumnWidth; ++i) {
      std::cout << '=';
    }
    std::cout << "+" << std::endl;
  };

  int padding = (TotalColumnWidth - header.size());
  int padleft = padding / 2;
  int padright = padding - padleft;

  // Print header row
  std::cout << boundary << std::string(padleft, ' ') << header << std::string(padright, ' ') <<  boundary << std::endl;

  // Print top boundary
  printBoundary();

  std::cout << "\r" << boundary << " "
    << std::setw(actionColumnWidth) << std::left << OS
    << " | " << std::setw(actionColumnWidth) << std::left << getOSNameVersion()
    << " " <<  boundary << std::endl;

  std::cout << "\r" << boundary << " "
    << std::setw(actionColumnWidth) << std::left << version1
    << " | " << std::setw(actionColumnWidth) << std::left << RVS_VERSION_STRING
    << " " <<  boundary << std::endl;

  std::cout << "\r" << boundary << " "
    << std::setw(actionColumnWidth) << std::left << version2
    << " | " << std::setw(actionColumnWidth) << std::left << getROCmVersion()
    << " " <<  boundary << std::endl;

  std::cout << "\r" << boundary << " "
    << std::setw(actionColumnWidth) << std::left << version3
    << " | " << std::setw(actionColumnWidth) << std::left << getAmdGpuDriverVersion()
    << " " <<  boundary << std::endl;

  rvs::gpulist::Initialize();

  std::vector<device_info> gpu_info_list = get_gpu_info();

  if(gpu_info_list.size()) {

    std::cout << "\r" << boundary << " "
      << std::setw(actionColumnWidth) << std::left << "GPUs"
      << " | " << std::setw(actionColumnWidth) << std::left << gpu_info_list.size()
      << " " <<  boundary << std::endl;
  }
  else {
    std::cout << "\r" << boundary << " "
      << std::setw(actionColumnWidth) << std::left << "GPUs"
      << " | " << "\033[31m" << std::setw(actionColumnWidth) << std::left << "No GPUs detected !" << "\033[0m"
      << " " <<  boundary << std::endl;
  }

  // Print bottom boundary
  printBoundary();

  std::string GPUdetailsPart1 = "GPU Name - GPU ID";
  std::string GPUdetailsPart2 = "ID - Node ID - BDF";
  padding = (actionColumnWidth - GPUdetailsPart1.size());
  padleft = padding / 2;
  padright = padding - padleft;
  int gpu_index = 0;

  if(gpu_info_list.size() % 2) {

    std::string GPU1 = gpu_info_list[gpu_index].name + " - " + std::to_string(gpu_info_list[gpu_index].gpu_id);
    padright = (actionColumnWidth - GPU1.size());

    std::cout << "\r" << boundary << " " << std::left  << GPUdetailsPart1 << std::string(padding, ' ') << " " <<  boundary 
      << " " << GPU1 << std::string(padright, ' ') << " " <<  boundary 
      << std::endl;

    GPU1 = std::to_string(gpu_index) + " - " + std::to_string(gpu_info_list[gpu_index].node_id) + " - " + 
      gpu_info_list[gpu_index].bus;
    padright = (actionColumnWidth - GPU1.size());

    padding = (actionColumnWidth - GPUdetailsPart2.size());
    std::cout << "\r" << boundary << " " << std::left  << GPUdetailsPart2 << std::string(padding, ' ') << " " <<  boundary 
      << " " << GPU1 << std::string(padright, ' ') << " " <<  boundary 
      << std::endl;

    if (1 == gpu_info_list.size())
      printDoubleBoundary();
    else 
      printBoundary();

    gpu_index++;
  }
  else {

    if (gpu_info_list.size()) {
      std::cout << "\r" << boundary << " " << std::left  << GPUdetailsPart1 << std::string(padding, ' ') << " " <<  boundary 
        << " " << std::string(actionColumnWidth, ' ') << " " <<  boundary 
        << std::endl;
    }
    else {
      std::string msg = "N/A";
      padright = (actionColumnWidth - msg.size());

      std::cout << "\r" << boundary << " " << std::left  << GPUdetailsPart1 << std::string(padding, ' ') << " " <<  boundary 
        << " " << "\033[31m" << msg << std::string(padright, ' ') << " " << "\033[0m" << boundary 
        << std::endl;
    }

    padding = (actionColumnWidth - GPUdetailsPart2.size());
    std::cout << "\r" << boundary << " " << std::left  << GPUdetailsPart2 << std::string(padding, ' ') << " " <<  boundary 
      << " " << std::string(actionColumnWidth, ' ') << " " <<  boundary 
      << std::endl;

    printBoundary();
  }

  for (;gpu_index < gpu_info_list.size();) {

    std::string GPU1 = gpu_info_list[gpu_index].name + " - " + std::to_string(gpu_info_list[gpu_index].gpu_id);
    padleft = (actionColumnWidth - GPU1.size());


    std::string GPU2 = gpu_info_list[gpu_index + 1].name + " - " + std::to_string(gpu_info_list[gpu_index + 1].gpu_id);
    padright = (actionColumnWidth - GPU2.size());

    std::cout << "\r" << boundary 
      << " " << GPU1 << std::string(padleft, ' ') << " " <<  boundary 
      << " " << GPU2 << std::string(padright, ' ') << " " <<  boundary 
      << std::endl;

    GPU1 = std::to_string(gpu_index) + " - " + std::to_string(gpu_info_list[gpu_index].node_id) + " - " + 
      gpu_info_list[gpu_index].bus;
    padleft = (actionColumnWidth - GPU1.size());

    GPU2 = std::to_string(gpu_index + 1) + " - " + std::to_string(gpu_info_list[gpu_index + 1].node_id) + " - " + 
      gpu_info_list[gpu_index + 1].bus;
    padright = (actionColumnWidth - GPU2.size());

    std::cout << "\r" << boundary 
      << " " << GPU1 << std::string(padleft, ' ') << " " <<  boundary 
      << " " << GPU2 << std::string(padright, ' ') << " " <<  boundary 
      << std::endl;

    gpu_index += 2;

    if (gpu_index == gpu_info_list.size())
      printDoubleBoundary();
    else
      printBoundary();
  }
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

  /* Test Summary */

  const char boundary = '|';

  // Header columns
  std::string header = "ROCm Validation Suite (RVS) Summary";
  std::string header1 = "Action Name";
  std::string header2 = "Module";
  std::string header3 = "Result";

  // Define column width for consistent spacing
  int columnWidth = 14;
  int actionColumnWidth = 32;
  int TotalColumnWidth = 69;

  // Function to print a horizontal boundary line
  auto printBoundary = [&]() {
    std::cout << "+";
    for (int i = 0; i < TotalColumnWidth; ++i) {
      std::cout << '-';
    }
    std::cout << "+" << std::endl;
  };

  // Function to print a horizontal boundary line
  auto printDoubleBoundary = [&]() {
    std::cout << "+";
    for (int i = 0; i < TotalColumnWidth; ++i) {
      std::cout << '=';
    }
    std::cout << "+" << std::endl;
  };

  int padding = (TotalColumnWidth - header.size());
  int padleft = padding / 2;
  int padright = padding - padleft;

  /* Quite logging is enabled */
  if (rvs::options::has_option("-q")) {

    // Print top boundary
    printDoubleBoundary();

    // Print header row
    std::cout << boundary << std::string(padleft, ' ') << header << std::string(padright, ' ') <<  boundary << std::endl;

    // Print top boundary
    printDoubleBoundary();

    systemOverview();

    // Print header row
    std::cout << boundary << " "
      << std::setw(actionColumnWidth) << std::left << header1
      << " | " << std::setw(columnWidth) << std::left << header2
      << " | " << std::setw(columnWidth) << std::left << header3
      << "  " << boundary << std::endl;

    // Print inner boundary
    printDoubleBoundary();
  }

  /* Start of action tests */

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

        std::thread in_progress_t;

      if (rvs::options::has_option("-q")) {

        in_progress = true;

        // Start compute workload thread
        in_progress_t = std::thread(&rvs::exec::in_progress_thread, this, action_info);

      }

      // execute action
      sts = pif1->run();

      if (rvs::options::has_option("-q")) {

        in_progress = false;

        in_progress_t.join();

        std::string actionresult = (!sts) ? "PASS" : "FAIL";
        std::string textcolor = (!sts) ? "\033[32m" : "\033[31m";

        std::cout << "\r" << boundary << " "
          << std::setw(actionColumnWidth) << std::left << action_info.name
          << " | " << std::setw(columnWidth) << std::left << action_info.module
          << " | " << textcolor << std::setw(columnWidth) << std::left << actionresult << "\033[0m"
          << "  " <<  boundary << std::endl;
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

        if (!rvs::options::has_option("-q")) {
          rvs::logger::Err("Action failed to run successfully.",
              action["module"].as<std::string>().c_str(),
              action["name"].as<std::string>().c_str());
        }

        action_info.result = false;
      }
      else {
        action_info.result = true;
      }
      action_details.push_back(action_info);
    }
  }
  /* End of action tests */

  /* Quite logging is not enabled  */
  if (!rvs::options::has_option("-q")) {

    const char boundary = '|';

    // Header columns
    std::string header = "ROCm Validation Suite (RVS) Summary";
    std::string header1 = "Action Name";
    std::string header2 = "Module";
    std::string header3 = "Result";

    // Define column width for consistent spacing
    int columnWidth = 14;
    int actionColumnWidth = 32;
    int TotalColumnWidth = 69;

    // Function to print a horizontal boundary line
    auto printBoundary = [&]() {
      std::cout << "+";
      for (int i = 0; i < TotalColumnWidth; ++i) {
        std::cout << '-';
      }
      std::cout << "+" << std::endl;
    };

    // Function to print a horizontal boundary line
    auto printDoubleBoundary = [&]() {
      std::cout << "+";
      for (int i = 0; i < TotalColumnWidth; ++i) {
        std::cout << '=';
      }
      std::cout << "+" << std::endl;
    };

    int padding = (TotalColumnWidth - header.size());
    int padleft = padding / 2;
    int padright = padding - padleft;

    // Print top boundary
    printDoubleBoundary();

    // Print header row
    std::cout << boundary << std::string(padleft, ' ') << header << std::string(padright, ' ') <<  boundary << std::endl;

    // Print top boundary
    printDoubleBoundary();

    systemOverview();

    // Print header row
    std::cout << boundary << " "
      << std::setw(actionColumnWidth) << std::left << header1
      << " | " << std::setw(columnWidth) << std::left << header2
      << " | " << std::setw(columnWidth) << std::left << header3
      << "  " << boundary << std::endl;

    // Print inner boundary
    printDoubleBoundary();

    for(size_t i = 0; i < action_details.size(); i++) {

      std::string result = (action_details[i].result) ? "PASS" : "FAIL";
      std::string textcolor = (action_details[i].result) ? "\033[32m" : "\033[31m";

      // Print data row
      std::cout << boundary << " "
        << std::setw(actionColumnWidth) << std::left << action_details[i].name
        << " | " << std::setw(columnWidth) << std::left << action_details[i].module
        << " | " << textcolor << std::setw(columnWidth) << std::left << result << "\033[0m"
        << "  " <<  boundary << std::endl;
    }

    // Print bottom boundary
    printBoundary();
  }
  else {
    printBoundary();
  }
  if (rvs::logger::to_json()) {
    rvs::lp::JsonEndNodeCreate();
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

