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

#define MODULE_NAME "rcqt"
#define MODULE_NAME_CAPS "RCQT"
#define JSON_CREATE_NODE_ERROR "JSON cannot create node"
#define JSON_PKGCHK_NODE_NAME "pkgchk"
#define PACKAGE "package"
#define PACKAGELIST "packagelist"
#define VERSION "version"
#define INTERNAL_ERROR "Internal Error"


#define SONAME  "soname"
#define LDPATH  "ldpath"
#define ARCH    "arch"

#define FILE "file"
#define ETC_PASSWD "/etc/passwd"
#define BUFFER_SIZE 3000

#if DRVS_OS_TYPE_NUM == 1
// debian defines
#elseif DRVS_OS_TYPE_NUM == 2
// fedora defines
#endif


using std::string;
using std::iterator;
using std::endl;
using std::ifstream;
using std::map;
using std::regex;
using std::vector;

rcqt_action::rcqt_action() {
  bjson = false;
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
  bool pkgchk_bool = false;
  bool pkglist_bool = false;

  // get the action name
  if (property_get(RVS_CONF_NAME_KEY, &action_name)) {
    rvs::lp::Err("Action name missing", MODULE_NAME_CAPS);
    return 1;
  }

  // check for -j flag (json logging)
  if (property.find("cli.-j") != property.end()) {
    unsigned int sec;
    unsigned int usec;
    rvs::lp::get_ticks(&sec, &usec);
    bjson = true;
    json_rcqt_node = rvs::lp::LogRecordCreate(MODULE_NAME,
      action_name.c_str(), rvs::logresults, sec, usec);
    if (json_rcqt_node == NULL) {
      // log the error
      msg =
      action_name + " " + MODULE_NAME + " "
      + JSON_CREATE_NODE_ERROR;
      rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
      return 1;
    }
  }

  // check if package check action is going to trigger
  pkgchk_bool =  rvs::actionbase::has_property(PACKAGE);
  if (pkgchk_bool == true)
    return pkgchk_run();

  pkglist_bool =  rvs::actionbase::has_property(PACKAGELIST);
  if (pkglist_bool == true)
    return pkglist_run();

  return -1;
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

  /* Junaid: unwanted variable ? */
  // Checking if version field exists
  bool package_found = false;

  handlerCreator creator;
  for( auto pkg : package_list){

    auto handler = creator.getPackageHandler(pkg);
    if(!handler){
      std::cout << "Failed to create handler " << std::endl;
      return -1;
    }

    handler->parseManifest();
    handler->validatePackages();
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

  handler->setPackageList(package_list);
  handler->listPackageVersion();

  return 0;
}

