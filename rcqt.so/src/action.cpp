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
#include "include/action.h"

#include <stdlib.h>
#include <sys/utsname.h>
#include <sys/types.h>
#include <unistd.h>

#include <pwd.h>
#include <grp.h>
#include <sys/stat.h>
#include <string.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cstdlib>
#include <array>
#include <map>
#include <vector>
#include <regex>
#include <set>
#include "include/handlerCreator.h"
#include "include/rcutils.h"

#include "include/rvs_key_def.h"
#include "include/rvsloglp.h"

#define JSON_CREATE_NODE_ERROR "JSON cannot create node"
#define JSON_PKGCHK_NODE_NAME "pkgchk"
#define PACKAGE "package"
#define VERSION "version"
#define INTERNAL_ERROR "Internal Error"


#define SONAME  "soname"
#define LDPATH  "ldpath"
#define ARCH    "arch"

#define FILE "file"
#define ETC_PASSWD "/etc/passwd"
#define BUFFER_SIZE 3000

static constexpr auto MODULE_NAME = "rcqt";
static constexpr auto MODULE_NAME_CAPS = "RCQT";

using std::string;
using std::iterator;
using std::endl;
using std::ifstream;
using std::map;
using std::regex;
using std::vector;

rcqt_action::rcqt_action() {
  PACKAGELIST = (getOS() == OSType::Ubuntu) ?"debpackagelist":"rpmpackagelist";
  bjson = false;
  module_name = MODULE_NAME;
}

rcqt_action::~rcqt_action() {
  property.clear();
}

/**
 * @brief Implements action functionality
 *
 * Functionality:
 *
 * - If "package" property is set, the package existance is checked, optionally check for the version if "version" field is set and exit
 * - If "user" property is s, and exits
 * - If "monitor" property is not set or is not set to "true", it stops the Worker thread and exits
 *
 * @return 0 - success. non-zero otherwise
 *
 * */

int rcqt_action::run() {
  string msg;
  bool propchk = false;
  int ret = 0;
  rvs::action_result_t action_result;
  // get the action name
  if (property_get(RVS_CONF_NAME_KEY, &action_name)) {
    msg = "Action name missing";
    rvs::lp::Err(msg, MODULE_NAME_CAPS);

    action_result.state = rvs::actionstate::ACTION_COMPLETED;
    action_result.status = rvs::actionstatus::ACTION_FAILED;
    action_result.output = msg;
    action_callback(&action_result);

    return 1;
  }

  // check for -j flag (json logging)
  if (has_property("cli.-j")) {
      bjson = true;
    }

  if (bjson){
      if (rvs::lp::JsonActionStartNodeCreate(MODULE_NAME, action_name.c_str())){
        rvs::lp::Err("json start create failed", MODULE_NAME_CAPS, action_name);
        return 1;
      }
  }

  // check if package check action is going to trigger
  propchk =  rvs::actionbase::has_property(PACKAGE);
  if (propchk == true)
    ret = pkgchk_run();

  propchk =  rvs::actionbase::has_property(PACKAGELIST);
  if (propchk == true)
    ret = pkglist_run();

  if(bjson){
    rvs::lp::JsonActionEndNodeCreate();
  }

  action_result.state = rvs::actionstate::ACTION_COMPLETED;
  action_result.status = (!ret) ? rvs::actionstatus::ACTION_SUCCESS : rvs::actionstatus::ACTION_FAILED;
  action_result.output = "RCQT Module action " + action_name + " completed";
  action_callback(&action_result);

  return ret;
}

/**
 * Check if the package is installed in the system (optional: check package version )
 * @return 0 - success, non-zero otherwise
 * */

int rcqt_action::pkgchk_run() {

  string package_name;
  string package;
  string msg;

  has_property(PACKAGE, &package_name);
  std::stringstream ss{package_name};
  vector<string> package_list;
  while(ss >> package){
    if(!package.empty())
      package_list.push_back(remSpaces(package));
  }

  handlerCreator creator;
  for( auto pkg : package_list){

    std::cout << "Meta package " << pkg << " :" << std::endl;

    auto handler = creator.getPackageHandler(pkg);
    if(!handler){
      std::cout << "Failed to create handler " << std::endl;
      return -1;
    }

    handler->setCallback(callback, user_param);
    handler->setAction(action_name);
    handler->setPkg(pkg);
    handler->setModule(MODULE_NAME);
    handler->parseManifest();
    handler->validatePackages();

    std::cout << std::endl;
  }

  return 0;
}

/**
 * List all ROCm packages and its version installed in the system.
 * @return 0 - success, non-zero otherwise
 * */

int rcqt_action::pkglist_run() {

  string packagelist;
  string package;
  handlerCreator creator;

  has_property(PACKAGELIST, &packagelist);
  std::stringstream ss{packagelist};
  vector<string> package_list;
  while(ss >> package){
    if(!package.empty())
      package_list.push_back(remSpaces(package));
  }

  auto handler = creator.getPackageHandler();
  if(!handler){
    std::cout << "Failed to create handler " << std::endl;
    return -1;
  }

  handler->setCallback(callback, user_param);
  handler->setPackageList(package_list);
  handler->setAction(action_name);
  handler->setModule(MODULE_NAME);

  handler->listPackageVersion();

  return 0;
}

void rcqt_action::cleanup_logs(){
  rvs::lp::JsonEndNodeCreate();
}

