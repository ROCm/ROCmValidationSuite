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
#include <cstdlib>
#include <array>
#include <map>
#include <vector>
#include <regex>
#include <set>

#include "include/rvs_key_def.h"
#include "include/rvsloglp.h"

#define MODULE_NAME "rcqt"
#define MODULE_NAME_CAPS "RCQT"
#define JSON_CREATE_NODE_ERROR "JSON cannot create node"
#define JSON_PKGCHK_NODE_NAME "pkgchk"
#define JSON_USRCHK_NODE_NAME "usrckh"
#define JSON_KERNELCHK_NODE_NAME "kernelchk"
#define JSON_LDCHK_NODE_NAME "ldchk"
#define PACKAGE "package"
#define VERSION "version"
#define INTERNAL_ERROR "Internal Error"

#define USER "user"
#define GROUP "group"

#define OS_VERSION "os_version"
#define KERNEL_VERSION "kernel_version"

#define SONAME  "soname"
#define LDPATH  "ldpath"
#define ARCH    "arch"

#define FILE "file"
#define ETC_PASSWD "/etc/passwd"
#define ETC_GROUP  "/etc/group"
#define DPKG_FILE  "/var/lib/dpkg/status"
#define PKG_CMD_FILE "installed_pkg.txt"
#define VERSION_FILE "version_file.txt"
#define LDCFG_FILE "rvs_ldcfg_file.txt"
#define STREAM_SIZE 512
#define USR_REG "([^/:]*)"
#define GRP_REG "([^/:]*):([^/:]*):([^/:]*):([^/:]+)"

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
#if 0
  bool pkgchk_bool = false;
  bool usrchk_bool = false;
  bool kernelchk_os_bool = false;
  bool ldcfgchk_so_bool = false;
  bool filechk_bool = false;
#endif

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

#if NEEDED_PLEASE_ENABLE
  // check if package check action is going to trigger
  pkgchk_bool =  rvs::actionbase::has_property(PACKAGE);
  if (pkgchk_bool == true)
    return pkgchk_run();

  // check if usrer check action is going to trigger
  usrchk_bool = rvs::actionbase::has_property(USER);

  if (usrchk_bool == true)
    return usrchk_run();

  // chck if kernel version action is going to trigger
  kernelchk_os_bool = rvs::actionbase::has_property(OS_VERSION);

  if (kernelchk_os_bool)
    return kernelchk_run();

  // check if ldcfg check action is going to trigger
  ldcfgchk_so_bool = rvs::actionbase::has_property(SONAME);

  if (ldcfgchk_so_bool)
    return ldcfgchk_run();

  // check if file check action is going to trigger
  filechk_bool = rvs::actionbase::has_property(FILE);

  if (filechk_bool == true)
    return filechk_run();
#endif

  return -1;
}

/**
 * Check if the package is installed in the system (optional: check package version )
 * @return 0 - success, non-zero otherwise
 * */

int rcqt_action::pkgchk_run() {
  string package_name;
  string msg;

  has_property(PACKAGE, &package_name);
  bool version_exists = false;

  // Checking if version field exists
  string version_name;
  version_exists = has_property(VERSION, &version_name);
  #if RVS_OS_TYPE_NUM == 1
  string command_string = "dpkg --get-selections > ";
  #endif
  #if RVS_OS_TYPE_NUM == 2
  string command_string = "rpm -qa --qf \"%{NAME}\n\" > ";
  #endif
  command_string += PKG_CMD_FILE;

  // We execute the dpkg-querry
  if (system(command_string.c_str()) == -1) {
    rvs::lp::Err("system() error", MODULE_NAME_CAPS, action_name);
    return 1;
  }
  // We read the result from the dpk-querry from the fd[0]
  vector <string> package_vector;
  bool package_found = false;

  std::regex pkg_pattern(package_name);

  ifstream pkg_stream(std::string(PKG_CMD_FILE));
  void *json_child_node = nullptr;
  char file_line[STREAM_SIZE];
  int i = 1;
  string PACKAGE_CONST = "package";
  while (pkg_stream.getline(file_line, STREAM_SIZE)) {
    string line_result = file_line;

    #if RVS_OS_TYPE_NUM == 1
    line_result = line_result.substr(0, line_result.length() - 7);

    line_result.erase(line_result.find_last_not_of(" \n\r\t")+1);
    #endif
    if (regex_match(line_result, pkg_pattern) == true) {
      if (bjson) {
        if (json_rcqt_node != NULL) {
          json_child_node = rvs::lp::CreateNode(json_rcqt_node
          , (PACKAGE_CONST + std::to_string(i++)).c_str());
          rvs::lp::AddString(json_child_node, "package"
          , line_result.c_str());
        }
      }
      if (version_exists) {
        char cmd_buffer[BUFFER_SIZE];
        #if RVS_OS_TYPE_NUM == 1
        snprintf(cmd_buffer, BUFFER_SIZE, \
        "dpkg -s %s | grep Version > %s", line_result.c_str(),
        (std::string(VERSION_FILE)).c_str());
        #endif
        #if RVS_OS_TYPE_NUM == 2
        snprintf(cmd_buffer, BUFFER_SIZE, \
        "rpm -qi %s | grep Version > %s", line_result.c_str(),
        (std::string(VERSION_FILE)).c_str());
        #endif

        if (system(cmd_buffer) == -1) {
          rvs::lp::Err("system() error", MODULE_NAME_CAPS, action_name);
          return 1;
        }
        std::regex version_pattern(version_name);
        ifstream version_stream(std::string(VERSION_FILE));
        char file_line[STREAM_SIZE];
        version_stream.getline(file_line, STREAM_SIZE);
        version_stream.close();
        string group_line_result = file_line;
        #if RVS_OS_TYPE_NUM == 1
        group_line_result = group_line_result
        .substr(9, group_line_result.length() - 9);
        #endif
        #if RVS_OS_TYPE_NUM == 2
        group_line_result.erase(0 , group_line_result.find_first_of(":") + 2);
        #endif
        if (bjson && json_child_node != NULL) {
          rvs::lp::AddString(json_child_node, "group"
          , group_line_result.c_str());
        }
        if (regex_match(group_line_result, version_pattern) == true) {
          string package_exists = "[" + action_name + "] "
          + "rcqt pkgcheck "
          + line_result + " true ";
          rvs::lp::Log(package_exists, rvs::logresults);
          package_found = true;
        } else {
          string pkg_not_exists = "[" + action_name + "] "
          + "rcqt pkgcheck "
          + line_result + " false ";
          rvs::lp::Log(pkg_not_exists, rvs::logresults);
          package_found = true;
        }
      } else {
        string package_exists = "[" + action_name + "] " + "rcqt pkgcheck "
        + line_result + " true";
        rvs::lp::Log(package_exists, rvs::logresults);
        package_found = true;
      }
      if (bjson && json_child_node != NULL)
        rvs::lp::AddNode(json_rcqt_node, json_child_node);
    }
  }
  if (!package_found) {
    string pkg_not_exists = "[" + action_name + "] " + "rcqt pkgcheck "
    + package_name + " false";
    rvs::lp::Log(pkg_not_exists, rvs::logresults);
    if (bjson && json_rcqt_node != NULL) {
      rvs::lp::AddString(json_rcqt_node, "package"
      , "not exist");
    }
  }
  string rm_command_string = std::string("rm ")
  + std::string(PKG_CMD_FILE) +
  (version_exists == true ? " " + std::string(VERSION_FILE) : "");
  // We execute rm command
  if (system(rm_command_string.c_str()) == -1) {
    rvs::lp::Err("system() error", MODULE_NAME_CAPS, action_name);
    return 1;
  }
  if (bjson && json_rcqt_node != NULL)
    rvs::lp::LogRecordFlush(json_rcqt_node);

  return 0;
}

/**
 * Check if the user exists in the system (optional: check for the group membership )
 * @return 0 - success, non-zero otherwise
 * */

int rcqt_action::usrchk_run() {
  string err_msg, msg;
  string user_name;
  if (has_property(USER, &user_name)) {
    bool group_exists = false;
    string group_values_string;

    // Check if gruop exists
    group_exists = has_property(GROUP, &group_values_string);
    vector<string> users_vector;
    ifstream passwd_stream(ETC_PASSWD);
    char file_line[STREAM_SIZE];
    regex usr_pattern(user_name);
    std::map<string, void *> json_map;
    void *json_child_node = nullptr;
    string USER_CONST = "user";
    int i = 1;
    while (passwd_stream.getline(file_line, STREAM_SIZE)) {
      const string line = std::string(file_line);
      std::smatch match;
      const std::regex get_user_pattern(USR_REG);
      if (std::regex_search(line.begin(), line.end()
        , match, get_user_pattern)) {
        string result = match[1];
        if (regex_match(result, usr_pattern) == true) {
          users_vector.push_back(result);
          string user_exists = "[" + action_name + "] " + "rcqt usercheck "
          + result + " true";
          rvs::lp::Log(user_exists, rvs::logresults);
          if (bjson) {
            if (json_rcqt_node != NULL) {
              json_child_node = rvs::lp::CreateNode(json_rcqt_node
              , (USER_CONST + std::to_string(i++)).c_str());
              rvs::lp::AddString(json_child_node, "user"
              , result.c_str());
              json_map[result] = json_child_node;
              rvs::lp::AddNode(json_rcqt_node, json_child_node);
            }
          }
        }
      }
    }
    passwd_stream.close();
    if (users_vector.empty()) {
      string user_not_exists = "[" + action_name + "] " + "rcqt usercheck "
      + user_name + " false";
      rvs::lp::Log(user_not_exists, rvs::logresults);
    }

    if (group_exists) {
      // regex group_patterns;
      vector<string> group_vector;
      group_vector = str_split(group_values_string, ",");
      vector<string> group_found_vector;
      for (vector<string>::iterator it = group_vector.begin();
           it != group_vector.end(); it++) {
        regex group_patterns(*it);
        bool b_group_found = false;
        ifstream group_stream(ETC_GROUP);
        while (group_stream.getline(file_line, STREAM_SIZE)) {
          const string line = std::string(file_line);
          std::smatch match;
          const std::regex get_group_pattern(GRP_REG);
          if (std::regex_search(line.begin(), line.end()
            , match, get_group_pattern)) {
            string result = match[1];
            if (std::regex_match(result, group_patterns)) {
              group_found_vector.push_back(result);
              vector<string> group_users_found =
              str_split(match[4], ",");
              std::set<string>user_group_set(group_users_found.begin()
              , group_users_found.end());
              for (string user_string : users_vector) {
                if (user_group_set.find(user_string)
                  != user_group_set.end()) {
                  string user_group_found = "[" + action_name + "] "
                  + "rcqt usercheck "
                  + user_string + " "
                  + result + \
                  " true";
                  rvs::lp::Log(user_group_found, rvs::logresults);
                  b_group_found = true;
                  if (bjson && json_rcqt_node != NULL) {
                    json_child_node =
                    reinterpret_cast<void*>(json_map[user_string]);
                    rvs::lp::AddString(json_child_node, "group"
                    , result.c_str());
                  }
                } else {
                  string user_group_found = "[" + action_name + "] "
                  + "rcqt usercheck "
                  + user_string + " "
                  + result + \
                  " false";
                rvs::lp::Log(user_group_found, rvs::logresults);
                }
              }
            }
          }
        }
        group_stream.close();
        if (b_group_found == false) {
          for (string user_string : users_vector) {
            string user_group_found = "[" + action_name + "] "
            + "rcqt usercheck "
            + user_string + " "
            + *it + \
            " false";
            rvs::lp::Log(user_group_found, rvs::logresults);
            if (bjson && json_rcqt_node != NULL) {
              json_child_node =
              reinterpret_cast<void*>(json_map[user_string]);
              rvs::lp::AddString(json_child_node, "group"
              , it->c_str());
            }
          }
        }
      }
    }
    if (bjson && json_rcqt_node != NULL)
      rvs::lp::LogRecordFlush(json_rcqt_node);
    return 0;
  }
  return -1;
}

/**
 * Check if the os and kernel version in the system match the givem os and kernel version
 * @return 0 - success, non-zero otherwise
 * */

int rcqt_action::kernelchk_run() {
  string msg;
  string os_version_values;
  string kernel_version_values;

  if (has_property(OS_VERSION, &os_version_values)) {
    // Check kernel version
    if (has_property(KERNEL_VERSION,
      &kernel_version_values) == false) {
      rvs::lp::Err("Kernel version missing in config"
      , MODULE_NAME, action_name);
    return 1;
      }

      /*
       * Fill the os version vector and kernel version vector with
       */
      vector<string> os_version_vector = str_split(os_version_values, ",");

      vector<string> kernel_version_vector =
      str_split(kernel_version_values, ",");

      /* 
       * Parsing /etc/os-release file for pretty name to extract 
       */
      std::ifstream os_version_read("/etc/os-release");
      string os_actual = "";
      string os_file_line;
      bool os_version_correct = false;
      bool os_version_found_in_system = false;
      while (getline(os_version_read, os_file_line)) {
        if (strcasestr(os_file_line.c_str(), "pretty_name") != nullptr) {
          os_version_found_in_system = true;
          os_actual = os_file_line.substr(13, os_file_line.length() - 14);
          if (bjson && json_rcqt_node != nullptr) {
            rvs::lp::AddString(json_rcqt_node, "os version", os_actual);
          }
          vector<string>::iterator os_iter;
          for (os_iter = os_version_vector.begin()
            ; os_iter != os_version_vector.end(); os_iter++) {
            std::regex os_pattern(*os_iter);
          if (regex_match(os_actual, os_pattern) == true) {
            os_version_correct = true;
            break;
          }
            }
            if (os_version_correct == true)
              break;
        }
      }
      os_version_read.close();
      if (os_version_found_in_system == false) {
        rvs::lp::Err("Unable to locate actual OS installed"
        , MODULE_NAME_CAPS, action_name);
        return 1;
      }

      // Get data about the kernel version
      struct utsname kernel_version_struct;
      if (uname(&kernel_version_struct) != 0) {
        rvs::lp::Err("Unable to read kernel version"
        , MODULE_NAME_CAPS, action_name);
        return 1;
      }

      string kernel_actual = kernel_version_struct.release;
      if (bjson && json_rcqt_node != nullptr) {
        rvs::lp::AddString(json_rcqt_node, "kernel version", kernel_actual);
      }
      bool kernel_version_correct = false;

      // Check if the given kernel version matches one from the list
      vector<string>::iterator kernel_iter;
      for (kernel_iter = kernel_version_vector.begin() ; \
        kernel_iter != kernel_version_vector.end(); kernel_iter++) {
        std::regex kernel_pattern(*kernel_iter);
      if (regex_match(kernel_actual, kernel_pattern)) {
        kernel_version_correct = true;
        break;
      }
        }
        string result = "[" + action_name + "] " + "rcqt kernelcheck " + \
        os_actual + " " + kernel_actual + " " + \
        (os_version_correct && kernel_version_correct ? "true" : "false");
        rvs::lp::Log(result, rvs::logresults);
        if (bjson && json_rcqt_node != nullptr) {
          rvs::lp::AddString(json_rcqt_node, "kerelchk"
          , os_version_correct && kernel_version_correct ? "true" : "false");
          rvs::lp::LogRecordFlush(json_rcqt_node);
        }
        return 0;
  }
  return -1;
}

/**
 * Check if the shared object is in the given location with the correct architecture
 * @return 0 - success, non-zero otherwise
 * */

int rcqt_action::ldcfgchk_run() {
  string msg;
  string soname_requested;
  string arch_requested;
  string ldpath_requested;
  if (has_property(SONAME, &soname_requested)) {
    if (has_property(ARCH, &arch_requested) == false) {
      rvs::lp::Err("acrhitecture field missing in config"
      , MODULE_NAME_CAPS, action_name);
      return 1;
    }
    if (has_property(LDPATH, &ldpath_requested) == false) {
      rvs::lp::Err("library path field missing in config"
      , MODULE_NAME_CAPS, action_name);
      return 1;
    }
    string ld_config_result = "[" + action_name + "] " +
    "rcqt ldconfigcheck ";
    struct stat stat_buf;
    if (stat(ldpath_requested.c_str(), &stat_buf) < 0) {
        string arch_fail = ld_config_result + "not found"
        + " NA " + ldpath_requested + " fail";
        rvs::lp::Log(arch_fail, rvs::logresults);
      if (bjson && json_rcqt_node != nullptr) {
        rvs::lp::LogRecordFlush(json_rcqt_node);
      }
      return 0;
    }
    char cmd_buffer[BUFFER_SIZE];
    snprintf(cmd_buffer, BUFFER_SIZE, \
    "ls -p %s | grep -v / > %s", ldpath_requested.c_str(),
             reinterpret_cast<const char*>(LDCFG_FILE));

    if (system(cmd_buffer) == -1) {
      rvs::lp::Err("system() error", MODULE_NAME_CAPS, action_name);
      return 1;
    }
    ifstream lib_stream(std::string(LDCFG_FILE));
    char file_line[STREAM_SIZE];
    vector<string> found_files_vector;
    std::regex file_pattern(soname_requested);
    std::map<string, void *> json_map;
    void *json_child_node = nullptr;
    string LIB_CONST = "lib";
    int i = 1;
    while (lib_stream.getline(file_line, STREAM_SIZE)) {
      if (regex_match(std::string(file_line), file_pattern))
        found_files_vector.push_back(std::string(file_line));
    }
    lib_stream.close();
    string arch_found_string;
    bool arch_found_bool = false;

    for (auto it = found_files_vector.begin()
      ; it != found_files_vector.end(); it++) {
      // Full path of shared object
      string full_ld_path = ldpath_requested + "/" + std::string(*it);
    snprintf(cmd_buffer, sizeof(cmd_buffer)
    , "file %s | grep \"shared object,\" > %s"
    , full_ld_path.c_str(), reinterpret_cast<const char*>(LDCFG_FILE));
    ifstream arch_stream(std::string(LDCFG_FILE));
    if (system(cmd_buffer) == -1) {
      rvs::lp::Err("system() error", MODULE_NAME_CAPS, action_name);
      return 1;
    }
    char file_line[STREAM_SIZE];
    std::regex arch_pattern(arch_requested);
    arch_stream.getline(file_line, STREAM_SIZE);
    string line_result = file_line;
    arch_stream.close();
    if (line_result.empty())
      continue;
    if (bjson) {
      if (json_rcqt_node != NULL) {
        json_child_node = rvs::lp::CreateNode(json_rcqt_node
        , (LIB_CONST + std::to_string(i++)).c_str());
        rvs::lp::AddString(json_child_node, "soname"
        , std::string(*it).c_str());
        rvs::lp::AddNode(json_rcqt_node, json_child_node);
      }
    }
    char arch_cmd_buffer[BUFFER_SIZE];
    snprintf(arch_cmd_buffer, sizeof(arch_cmd_buffer)
    , "objdump -f %s | grep ^architecture> %s"
    , full_ld_path.c_str(), reinterpret_cast<const char*>(LDCFG_FILE));
    if (system(arch_cmd_buffer) == -1) {
      rvs::lp::Err("system() error", MODULE_NAME_CAPS, action_name);
      return 1;
    }
    arch_stream.open(std::string(LDCFG_FILE));
    arch_stream.getline(file_line, STREAM_SIZE);
    arch_found_string = file_line;
    arch_found_string = str_split(arch_found_string, ",")[0];
    arch_found_string.erase(0, arch_found_string
    .find_first_not_of("architecture: ") - 1);
    arch_found_bool = true;
    if (bjson) {
      if (json_rcqt_node != NULL) {
        rvs::lp::AddString(json_child_node, "arch"
        , arch_found_string.c_str());
      }
    }
    if (regex_match(arch_found_string, arch_pattern)) {
      string arch_pass = ld_config_result + *it
      + " " + arch_found_string + " " + ldpath_requested + " pass";
      rvs::lp::Log(arch_pass, rvs::logresults);
    } else {
      string arch_fail = ld_config_result + *it
      + " NA " + ldpath_requested + " fail";
      rvs::lp::Log(arch_fail, rvs::logresults);
    }
    arch_stream.close();
      }
      if (!arch_found_bool) {
        string lib_fail = ld_config_result
        + " not found NA " + ldpath_requested +  " fail";
        rvs::lp::Log(lib_fail, rvs::logresults);
        if (bjson && json_rcqt_node != nullptr) {
          rvs::lp::AddString(json_rcqt_node, "soname", soname_requested);
          rvs::lp::AddString(json_rcqt_node, "ldchk", "false");
        }
      }
      string rm_command_string = std::string("rm ")
      + std::string(LDCFG_FILE);

      // We execute rm command
      if (system(rm_command_string.c_str()) == -1) {
        rvs::lp::Err("system() error", MODULE_NAME_CAPS, action_name);
        return 1;
      }
      if (bjson && json_rcqt_node != nullptr) {
        rvs::lp::LogRecordFlush(json_rcqt_node);
      }
      return 0;
  }
  return -1;
}

// Converts decimal into octal
int rcqt_action::dectooct(int decnum) {
  int rem, i = 1, octnum = 0;
  while (decnum !=0) {
    rem = decnum%8;
    decnum/=8;
    octnum +=rem*i;
    i *=10;
  }
  return octnum;
}
/**
 * Check if the parametrs of the file match the given ones
 * @return 0 - success, non-zero otherwise
 * */ 

int rcqt_action::filechk_run() {
  string exists_string, file, owner, group, msg, check;
  int permission, type;
  bool exists;
  struct stat info;

  map<string, string>::iterator iter;
  iter = property.find("name");
  std::string action_name = iter->second;

  // get from property which file we are checking
  iter = property.find("file");
  file = iter->second;

  // get bool value of property exists
  iter = property.find("exists");
  if (iter == property.end())
    exists = true;
  else
    exists_string = iter->second;
  if (exists_string == "false")
    exists = false;
  else if (exists_string == "true")
    exists = true;
  std::size_t found = file.find_last_of("/\\");
  string file_path = file.substr(0, found);
  string file_requested = file.substr(found+1);
  if (stat(file_path.c_str(), &info) < 0) {
    string check;
    if (exists == false) {
      check = "false";
      msg = "[" + action_name + "] " + "rcqt filecheck "
      + file_path +" DNE " + check;
      rvs::lp::Log(msg, rvs::logresults);
      if (bjson && json_rcqt_node != nullptr) {
        rvs::lp::AddString(json_rcqt_node
        , iter->second, file);
      }
    } else {
      msg = "[" + action_name + "] " + "rcqt filecheck exists false";
      rvs::lp::Log(msg, rvs::logresults);
    }
    if (bjson && json_rcqt_node != nullptr) {
      rvs::lp::LogRecordFlush(json_rcqt_node);
    }

    return 0;
    // if exists property is set to true and file is found,check each parameter
  }
  char cmd_buffer[BUFFER_SIZE];
  snprintf(cmd_buffer, BUFFER_SIZE, \
  "ls %s | grep -v / > %s", file_path.c_str()
  , (std::string(LDCFG_FILE)).c_str());

  if (system(cmd_buffer) == -1) {
    rvs::lp::Err("system() error", MODULE_NAME_CAPS, action_name);
    return 1;
  }
  void *json_child_node = nullptr;
  string FILE_CONST = "file";
  int i = 1;
  std::map<string, void *> json_map;
  ifstream file_stream(std::string(LDCFG_FILE));
  char file_line[STREAM_SIZE];
  vector<string> found_files_vector;
  std::regex file_pattern(file_requested);
  while (file_stream.getline(file_line, STREAM_SIZE)) {
    if (regex_match(std::string(file_line), file_pattern)) {
      found_files_vector.push_back(std::string(file_line));
      if (bjson) {
        if (json_rcqt_node != NULL) {
          json_child_node = rvs::lp::CreateNode(json_rcqt_node
          , (FILE_CONST + std::to_string(i++)).c_str());
          rvs::lp::AddString(json_child_node, "file"
          , file_line);
          json_map[std::string(file_line)] = json_child_node;
          rvs::lp::AddNode(json_rcqt_node, json_child_node);
        }
      }
    }
  }
  file_stream.close();
  if (exists == false && found_files_vector.empty()) {
    check = "true";
    msg = "[" + action_name + "] " + "rcqt filecheck "
    + file_path +" DNE " + check;
    rvs::lp::Log(msg, rvs::logresults);
    if (bjson && json_rcqt_node != nullptr) {
      rvs::lp::AddString(json_rcqt_node
      , "exists", file);
    }
  }
  if (exists == true && found_files_vector.empty()) {
    msg = "[" + action_name + "] " + "rcqt filecheck exists false";
    rvs::lp::Log(msg, rvs::logresults);
  }
  for (auto file_it = found_files_vector.begin();
       file_it != found_files_vector.end(); file_it++) {
    file = file_path + "/" + std::string(*file_it);
  // check if exists property corresponds to real existence of the file
  if (exists == false) {
    if (stat(file.c_str(), &info) < 0)
      check = "true";
    else
      check = "false";
    msg = "[" + action_name + "] " + "rcqt filecheck "+ file +" DNE " + check;
    rvs::lp::Log(msg, rvs::logresults);
    if (bjson && json_rcqt_node != nullptr) {
      json_child_node =
      reinterpret_cast<void*>(json_map[std::string(*file_it)]);
      rvs::lp::AddString(json_child_node
      , "exists", file);
    }
  } else {
    // when exists propetry is true,but file cannot be found
    if (stat(file.c_str(), &info) < 0) {
      msg = "[" + action_name + "] " + "rcqt filecheck "+ file +
      " file is not found";
      rvs::lp::Log(msg, rvs::logerror);
    // if exists property is set to true and file is found,check each parameter
    } else {
      // check if owner is tested
      iter = property.find("owner");
      if (iter != property.end()) {
        // check if value from property is equal to real one
        owner = iter->second;
        std::regex owner_pattern(owner);
        struct passwd p, *result;
        char pbuff[256];
        if ((getpwuid_r(info.st_uid, &p, pbuff, sizeof(pbuff), &result) != 0)) {
          rvs::lp::Err("Error with getpwuid_r", MODULE_NAME_CAPS, action_name);
          return 1;
        }
        if (regex_match(p.pw_name, owner_pattern))
          check = "true";
        else
          check = "false";
        msg = "[" + action_name + "] " + "rcqt filecheck " \
        + (strcmp(check.c_str(), "true") == 0 ? p.pw_name : owner )
        + " " + check;
        rvs::lp::Log(msg, rvs::logresults);
        if (bjson && json_rcqt_node != nullptr) {
          json_child_node =
          reinterpret_cast<void*>(json_map[std::string(*file_it)]);
          rvs::lp::AddString(json_child_node
          , "owner", p.pw_name);
        }
      }
      // check if group is tested
      iter = property.find("group");
      if (iter != property.end()) {
        // check if value from property is equal to real one
        group = iter->second;
        std::regex group_pattern(group);
        struct group g, *result;
        char pbuff[256];
        if ((getgrgid_r(info.st_gid, &g, pbuff, sizeof(pbuff), &result) != 0)) {
          rvs::lp::Err("Error with getgrgid_r", MODULE_NAME, action_name);
          return 1;
        }
        if (regex_match(g.gr_name, group_pattern))
          check = "true";
        else
          check = "false";
        msg = "[" + action_name + "] " + "rcqt filecheck "
        + (strcmp(check.c_str(), "true") == 0 ? g.gr_name : group)
        + " " + check;
        rvs::lp::Log(msg, rvs::logresults);
        if (bjson && json_rcqt_node != nullptr) {
          json_child_node =
          reinterpret_cast<void*>(json_map[std::string(*file_it)]);
          rvs::lp::AddString(json_child_node
          , "group", g.gr_name);
        }
      }
      // check if permissions are tested
      iter = property.find("permission");
      if (iter != property.end()) {
        // check if value from property is equal to real one
        permission = std::atoi(iter->second.c_str());
        if (dectooct(info.st_mode)%1000 == permission)
          check = "true";
        else
          check = "false";
        msg = "[" + action_name + "] " + "rcqt filecheck " + \
        std::to_string(permission)+" "+check;
        rvs::lp::Log(msg, rvs::logresults);
        if (bjson && json_rcqt_node != nullptr) {
          json_child_node =
          reinterpret_cast<void*>(json_map[std::string(*file_it)]);
          rvs::lp::AddString(json_child_node
          , "permissions", std::to_string(permission));
        }
      }
      // check if type is tested
      iter = property.find("type");
      if (iter != property.end()) {
        // check if value from property is equal to real one
        type = std::atoi(iter->second.c_str());
        struct stat buf;
        if (lstat(file.c_str(), &buf) >= 0) {
          if (dectooct(buf.st_mode)/1000 == type)
            check = "true";
          else
            check = "false";
          msg = "[" + action_name + "] " + "rcqt filecheck " + \
          std::to_string(type)+" "+check;
          rvs::lp::Log(msg, rvs::logresults);
          if (bjson && json_rcqt_node != nullptr) {
            json_child_node =
            reinterpret_cast<void*>(json_map[std::string(*file_it)]);
            rvs::lp::AddString(json_child_node
            , "type", std::to_string(type));
          }
        }
      }
    }
  }
  }
  if (bjson && json_rcqt_node != nullptr) {
    rvs::lp::LogRecordFlush(json_rcqt_node);
  }
  string rm_command_string = std::string("rm ")
  + std::string(LDCFG_FILE);

  // We execute rm command
  if (system(rm_command_string.c_str()) == -1) {
    rvs::lp::Err("system() error", MODULE_NAME_CAPS, action_name);
    return 1;
  }
  return 0;
}
