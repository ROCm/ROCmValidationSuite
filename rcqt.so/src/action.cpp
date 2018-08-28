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
#include "action.h"

#include <stdlib.h>
#include <sys/utsname.h>
#include <sys/types.h>
#include <unistd.h>
#include <pwd.h>
#include <grp.h>
#include <sys/stat.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>

#include "rvsloglp.h"

#define MODULE_NAME "rcqt"
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

#define BUFFER_SIZE 3000

using std::cerr;
using std::string;
using std::cin;
using std::cout;
using std::cerr;
using std::iterator;
using std::endl;
using std::ifstream;
using std::map;

action::action() {
  bjson = false;
}

action::~action() {
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

int action::run() {
  int error = 0;
  string msg;
  bool pkgchk_bool = false;
  bool usrchk_bool = false;
  bool kernelchk_os_bool = false;
  bool kernelchk_kernel_bool = false;
  bool ldcfgchk_so_bool = false;
  bool ldcfgchk_arch_bool = false;
  bool ldcfgchk_ldpath_bool = false;
  bool filechk_bool = false;

  // get the action name
  rvs::actionbase::property_get_action_name(&error);
  if (error == 2) {
    msg = "action field is missing in gst module";
    cerr << "RVS-RCQT: " << msg;
    return -1;
  }

  // check for -j flag (json logging)
  if (property.find("cli.-j") != property.end()) {
    unsigned int sec;
    unsigned int usec;
    rvs::lp::get_ticks(&sec, &usec);
    bjson = true;
    json_rcqt_node = rvs::lp::LogRecordCreate(MODULE_NAME,
                            action_name.c_str(), rvs::loginfo, sec, usec);
    if (json_rcqt_node == NULL) {
      // log the error
      msg =
      action_name + " " + MODULE_NAME + " "
      + JSON_CREATE_NODE_ERROR;
      cerr << "RVS-RCQT: " << msg;
    }
  }

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
  kernelchk_kernel_bool = rvs::actionbase::has_property(KERNEL_VERSION);

  if (kernelchk_os_bool && kernelchk_kernel_bool)
    return kernelchk_run();

  // check if ldcfg check action is going to trigger
  ldcfgchk_so_bool = rvs::actionbase::has_property(SONAME);
  ldcfgchk_arch_bool = rvs::actionbase::has_property(ARCH);
  ldcfgchk_ldpath_bool = rvs::actionbase::has_property(LDPATH);

  if (ldcfgchk_so_bool && ldcfgchk_arch_bool && ldcfgchk_ldpath_bool)
    return ldcfgchk_run();

  // check if file check action is going to trigger
  filechk_bool = rvs::actionbase::has_property(FILE);

  if (filechk_bool == true)
    return filechk_run();

  return -1;
}

/**
 * Check if the package is installed in the system (optional: check package version )
 * @return 0 - success, non-zero otherwise
 * */

int action::pkgchk_run() {
  string package_name;
  string msg;
  if (has_property(PACKAGE, &package_name)) {
    bool version_exists = false;

    // Checking if version field exists
    string version_name;
    version_exists = has_property(VERSION, &version_name);
    pid_t pid;
    int fd[2];
    pipe(fd);

    pid = fork();
    if (pid == 0) {
      // Child process

      // Pipe the standard output to the fd[1]
      dup2(fd[1], STDOUT_FILENO);
      dup2(fd[1], STDERR_FILENO);
      char buffer[BUFFER_SIZE];

      snprintf(buffer, BUFFER_SIZE, \
      "dpkg-query -W -f='${Status} ${Version}\n' %s", package_name.c_str());

      // We execute the dpkg-querry
      system(buffer);
      exit(0);
    } else if (pid > 0) {
      // Parent
      char result[BUFFER_SIZE];
      int count;
      close(fd[1]);

      // We read the result from the dpk-querry from the fd[0]
      count = read(fd[0], result, BUFFER_SIZE);

      result[count] = 0;
      string result1 = result;

      // We parse the given result
      string version_value = result1.substr(21, count - 21);
      int ending = version_value.find_first_of('\n');
      if (ending > 0) {
        version_value = version_value.substr(0, ending);
      }
      string passed = "packagecheck " + package_name + " TRUE";
      string failed = "packagecheck " + package_name + " FALSE";

      /* 
       * If result start with dpkg-querry: then we haven't found the package
       * If we get something different, then we confirme that the package is found
       * if version is equal to the required then the test pass
       */

      if (strstr(result, "dpkg-query:") == result) {
        log(failed.c_str(), rvs::logresults);
        if (bjson && json_rcqt_node != nullptr) {
          rvs::lp::AddString(json_rcqt_node, package_name, "not exists");
          rvs::lp::AddString(json_rcqt_node, "pkgchk", "fail");
          rvs::lp::LogRecordFlush(json_rcqt_node);
        }
      } else if (version_exists == false) {
        log(passed.c_str(), rvs::logresults);
        if (bjson && json_rcqt_node != nullptr) {
          rvs::lp::AddString(json_rcqt_node, package_name, "exists");
          rvs::lp::AddString(json_rcqt_node, "pkgchk", "pass");
          rvs::lp::LogRecordFlush(json_rcqt_node);
        }
      } else if (version_name.compare(version_value) == 0) {
        log(passed.c_str(), rvs::logresults);
        if (bjson && json_rcqt_node != nullptr) {
          rvs::lp::AddString(json_rcqt_node, package_name, "exists");
          rvs::lp::AddString(json_rcqt_node, version_name, "exists");
          rvs::lp::AddString(json_rcqt_node, "pkgchk", "pass");
          rvs::lp::LogRecordFlush(json_rcqt_node);
        }
      } else {
        log(failed.c_str(), rvs::logresults);
        if (bjson && json_rcqt_node != nullptr) {
          rvs::lp::AddString(json_rcqt_node, package_name, "exists");
          rvs::lp::AddString(json_rcqt_node, version_name, "not exists");
          rvs::lp::AddString(json_rcqt_node, "pkgchk", "fail");
          rvs::lp::LogRecordFlush(json_rcqt_node);
        }
      }
    } else {
      // fork process error
      cerr << INTERNAL_ERROR << '\n';
      return -1;
    }
    return 0;
  }
  return -1;
}

/**
 * Check if the user exists in the system (optional: check for the group membership )
 * @return 0 - success, non-zero otherwise
 * */

int action::usrchk_run() {
  string err_msg, msg;
  string user_name;
  if (has_property(USER, &user_name)) {
    bool group_exists = false;
    string group_values_string;

    // Check if gruop exists
    group_exists = has_property(GROUP, &group_values_string);

    // Structures for checking group and user
    struct passwd pwd, *result;

    char pwdbuffer[2000];
    int pwdbufflenght = 2000;
    struct group grp, *grprst;
    string user_exists = "[rcqt] usercheck "
        + user_name + " user exists";
    string user_not_exists = "[rcqt] usercheck "
        + user_name + " user not exists";

    // Check for given user
    if (getpwnam_r(user_name.c_str()
      , &pwd, pwdbuffer, pwdbufflenght, &result) != 0) {
      cerr << "Error with getpwnam_r" << endl;
      return -1;
    }
    if (result == nullptr) {
      log(user_not_exists.c_str(), rvs::logresults);
      if (bjson && json_rcqt_node != nullptr) {
        rvs::lp::AddString(json_rcqt_node, user_name, "not exists");
      }
    } else {
      log(user_exists.c_str(), rvs::logresults);
      if (bjson && json_rcqt_node != nullptr) {
        rvs::lp::AddString(json_rcqt_node, user_name, "exists");
      }
    }
    if (group_exists) {
      // Put the group list into vector
      string delimiter = ",";
      vector<string> group_vector;
      group_vector = str_split(group_values_string, delimiter);

      // Check if the group exists
      for (vector<string>::iterator vector_iter = group_vector.begin()
          ; vector_iter != group_vector.end(); vector_iter++) {
        string user_group = "[rcqt] usercheck " + user_name;
        int error_group;

        if ((error_group =  getgrnam_r(vector_iter->c_str()
          , &grp, pwdbuffer, pwdbufflenght, &grprst)) != 0) {
          cerr << "Error with getgrnam_r" << endl;
          return -1;
        }
        if (error_group == EIO) {
          cerr << "IO error" << endl;
          return -1;
        } else if (error_group == EINTR) {
          cerr << "Error sginal was caught during getgrnam_r" << endl;
          return -1;
        } else if (error_group == EMFILE) {
          cerr << "Error file descriptors are currently open" << endl;
          return -1;
        } else if (error_group == ERANGE) {
          cerr << "Error insufficient buffer in getgrnam_r" << endl;
          return -1;
        }
        string err_msg;
        if (grprst == nullptr) {
          err_msg = "group ";
          err_msg += vector_iter->c_str();
          err_msg += " doesn't exist";
          log(err_msg.c_str(), rvs::logerror);
          continue;
        }

        int i;
        int j = 0;

        // Compare if the user group id is equal to the group id
        for (i = 0; grp.gr_mem[i] != NULL; i++) {
          if (strcmp(grp.gr_mem[i], user_name.c_str()) == 0) {
            user_group = user_group + " " + vector_iter->c_str() + " is member";
            log(user_group.c_str(), rvs::logresults);
            if (bjson && json_rcqt_node != nullptr) {
              rvs::lp::AddString(json_rcqt_node
              , user_group + " " + vector_iter->c_str(), "is member");
            }
            j = 1;
            break;
          }
        }

        // If the index is 0 then we user id doesn't match the group id
        if (j == 0) {
          // printf("user is not in the group\n");

          user_group = user_group + " " + vector_iter->c_str() \
          + " is not member";
          log(user_group.c_str(), rvs::logresults);
          if (bjson && json_rcqt_node != nullptr) {
            rvs::lp::AddString(json_rcqt_node
            , user_group + " " + vector_iter->c_str(), "is not member");
          }
          j = 1;
        }
      }
    }
    if (bjson && json_rcqt_node != nullptr) {
      rvs::lp::LogRecordFlush(json_rcqt_node);
    }
    return 0;
  }
  return -1;
}

/**
 * Check if the os and kernel version in the system match the givem os and kernel version
 * @return 0 - success, non-zero otherwise
 * */

int action::kernelchk_run() {
  string msg;
  string os_version_values;
  string kernel_version_values;

  if (has_property(OS_VERSION, &os_version_values)) {
    // Check kernel version
    if (has_property(KERNEL_VERSION,
      &kernel_version_values) == false) {
      cerr << "Kernel version missing in config" << endl;
      return -1;
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
          if (strcmp(os_iter->c_str(), os_actual.c_str()) == 0) {
            os_version_correct = true;
            break;
          }
        }
        if (os_version_correct == true)
          break;
      }
    }
    if (os_version_found_in_system == false) {
      cerr << "Unable to locate actual OS installed" << endl;
      return -1;
    }

    // Get data about the kernel version
    struct utsname kernel_version_struct;
    if (uname(&kernel_version_struct) != 0) {
      cerr << "Unable to read kernel version" << endl;
      return -1;
    }

    string kernel_actual = kernel_version_struct.release;
    if (bjson && json_rcqt_node != nullptr) {
      rvs::lp::AddString(json_rcqt_node, "kernel version", kernel_actual);
    }
    bool kernel_version_correct = false;

    // Check if the given kernel version matches one from the list
    vector<string>::iterator kernel_iter;
    for (kernel_iter = kernel_version_vector.begin() ; \
      kernel_iter != kernel_version_vector.end(); kernel_iter++)
      if (kernel_actual.compare(*kernel_iter) == 0) {
        kernel_version_correct = true;
        break;
      }
    string result = "[rcqt] kernelcheck " + os_actual + \
    " " + kernel_actual + " " + \
    (os_version_correct && kernel_version_correct ? "pass" : "fail");
    log(result.c_str(), rvs::logresults);
    if (bjson && json_rcqt_node != nullptr) {
      rvs::lp::AddString(json_rcqt_node, "kerelchk"
      , os_version_correct && kernel_version_correct ? "pass" : "fail");
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

int action::ldcfgchk_run() {
  string msg;
  string soname_requested;
  string arch_requested;
  string ldpath_requested;
  if (has_property(SONAME, &soname_requested)) {
    if (has_property(ARCH, &arch_requested) == false) {
      cerr << "acrhitecture field missing in config" << endl;
      return -1;
    }
    if (has_property(LDPATH, &ldpath_requested) == false) {
      cerr << "library path field missing in config" << endl;
      return -1;
    }
    // Full path of shared object
    string full_ld_path = ldpath_requested + "/" + soname_requested;

    int fd[2];
    pid_t pid;
    pipe(fd);
    pid = fork();
    if (pid == 0) {
      // child process
      dup2(fd[1], STDOUT_FILENO);
      dup2(fd[1], STDERR_FILENO);
      char buffer[256];
      snprintf(buffer, sizeof(buffer), "objdump -f %s", full_ld_path.c_str());

      system(buffer);
      exit(0);
    } else if (pid > 0) {
      // Parent process
      char result[BUFFER_SIZE];
      close(fd[1]);
      string ld_config_result = "[rcqt] ldconfigcheck ";

      string result_string = result;

      if (strstr(result, "architecture:") != nullptr) {
        vector<string> objdump_lines = str_split(result_string, "\n");
        int begin_of_the_arch_string = 0;
        int end_of_the_arch_string = 0;
        for (uint i = 0; i < objdump_lines.size(); i++) {
          // cout << objdump_lines[i] << "*" << endl;
          if (objdump_lines[i].find("architecture") != string::npos) {
            begin_of_the_arch_string = objdump_lines[i].find(":");
            end_of_the_arch_string = objdump_lines[i].find(",");
            string arch_found = objdump_lines[i]
              .substr(begin_of_the_arch_string + 2
              , end_of_the_arch_string - begin_of_the_arch_string - 2);
            if (bjson && json_rcqt_node != nullptr) {
              rvs::lp::AddString(json_rcqt_node, "soname", soname_requested);
              rvs::lp::AddString(json_rcqt_node, "architecture", arch_found);
            }
            if (arch_found.compare(arch_requested) == 0) {
              string arch_pass = ld_config_result + soname_requested
                + " " + full_ld_path + " " + arch_found + " pass";
              log(arch_pass.c_str(), rvs::logresults);
              if (bjson && json_rcqt_node != nullptr) {
                rvs::lp::AddString(json_rcqt_node, "ldchk", "pass");
              }
            } else {
              string arch_pass = ld_config_result + soname_requested + " "
                + full_ld_path + " " + arch_found + " fail";
              log(arch_pass.c_str(), rvs::logresults);
              if (bjson && json_rcqt_node != nullptr) {
                rvs::lp::AddString(json_rcqt_node, "ldchk", "fail");
              }
            }
          }
        }
      } else {
        string lib_fail = ld_config_result + soname_requested
          + " not found " + "na " + "fail";
        log(lib_fail.c_str(), rvs::logresults);
        if (bjson && json_rcqt_node != nullptr) {
          rvs::lp::AddString(json_rcqt_node, "soname", soname_requested);
          rvs::lp::AddString(json_rcqt_node, "ldchk", "fail");
        }
      }
    } else {
      cerr << "Internal Error" << endl;
      return -1;
    }
    if (bjson && json_rcqt_node != nullptr) {
      rvs::lp::LogRecordFlush(json_rcqt_node);
    }
    return 0;
  }
  return -1;
}

// Converts decimal into octal
int action::dectooct(int decnum) {
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

int action::filechk_run() {
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

  // check if exists property corresponds to real existence of the file
  if (exists == false) {
    if (stat(file.c_str(), &info) < 0)
      check = "true";
    else
      check = "false";
    msg = "[" + action_name + "] " + " filecheck "+ file +" DNE " + check;
    log(msg.c_str(), rvs::logresults);
    if (bjson && json_rcqt_node != nullptr) {
      rvs::lp::AddString(json_rcqt_node
      , "exists", file);
    }
    } else {
    // when exists propetry is true,but file cannot be found
    if (stat(file.c_str(), &info) < 0) {
      log("File is not found", rvs::logerror);
    // if exists property is set to true and file is found,check each parameter
    } else {
      // check if owner is tested
      iter = property.find("owner");
      if (iter != property.end()) {
        // check if value from property is equal to real one
        owner = iter->second;
        struct passwd p, *result;
        char pbuff[256];
        if ((getpwuid_r(info.st_uid, &p, pbuff, sizeof(pbuff), &result) != 0))
          cout << "Error with getpwuid_r" << endl;
        if (p.pw_name == owner)
          check = "true";
        else
          check = "false";
        msg = "[" + action_name + "] " + " filecheck " \
        + owner +" owner:" + check;
        log(msg.c_str(), rvs::logresults);
        if (bjson && json_rcqt_node != nullptr) {
          rvs::lp::AddString(json_rcqt_node
          , "owner", owner);
        }
      }
      // check if group is tested
      iter = property.find("group");
      if (iter != property.end()) {
        // check if value from property is equal to real one
        group = iter->second;
        struct group g, *result;
        char pbuff[256];
        if ((getgrgid_r(info.st_gid, &g, pbuff, sizeof(pbuff), &result) != 0))
          cout << "Error with getgrgid_r" << endl;
        if (g.gr_name == group)
          check = "true";
        else
          check = "false";
        msg = "[" + action_name + "] " + " filecheck " + group+ " group:"+check;
        log(msg.c_str(), rvs::logresults);
        if (bjson && json_rcqt_node != nullptr) {
          rvs::lp::AddString(json_rcqt_node
          , "group", group);
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
        msg = "[" + action_name + "] " + " filecheck " + \
        std::to_string(permission)+" permission:"+check;
        log(msg.c_str(), rvs::logresults);
        if (bjson && json_rcqt_node != nullptr) {
          rvs::lp::AddString(json_rcqt_node
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
          msg = "[" + action_name + "] " + " filecheck " + \
          std::to_string(type)+" type:"+check;
          log(msg.c_str(), rvs::logresults);
          if (bjson && json_rcqt_node != nullptr) {
            rvs::lp::AddString(json_rcqt_node
            , "type", std::to_string(type));
          }
        }
      }
    }
  }
      if (bjson && json_rcqt_node != nullptr) {
      rvs::lp::LogRecordFlush(json_rcqt_node);
    }
  return 0;
}
