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
#include <string.h>

#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>

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

#define BUFFER_SIZE 3000


using std::cerr;
using namespace std;


action::action() {
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

  // get the action name
  rvs::actionbase::property_get_action_name(&error);
  if (error == 2) {
    msg = "action field is missing in gst module";
    log(msg.c_str(), rvs::logerror);

    return -1;
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

  return -1;
}

/**
 * Check if the package is installed in the system (optional: check package version )
 * @param property config file map fields
 * @return 0 - success, non-zero otherwise
 * */

int action::pkgchk_run() {
  string package_name;
  if (has_property(PACKAGE, package_name)) {
    bool version_exists = false;

    // Checking if version field exists
    string version_name;
    version_exists = has_property(VERSION, version_name);
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
      snprintf(buffer, BUFFER_SIZE
          , "dpkg-query -W -f='${Status} ${Version}\n' %s"
          , package_name.c_str());
      // We execute the dpkg-querry
      system(buffer);

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
      } else if (version_exists == false) {
        log(passed.c_str(), rvs::logresults);
      } else if (version_name.compare(version_value) == 0) {
        log(passed.c_str(), rvs::logresults);
      } else {
        log(failed.c_str(), rvs::logresults);
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
 * @param property config file map fields
 * @return 0 - success, non-zero otherwise
 * */

int action::usrchk_run() {
  string err_msg;
  string user_name;
  if (has_property(USER, user_name)) {
    bool group_exists = false;
    string group_values_string;

    // Check if gruop exists
    group_exists = has_property(GROUP, group_values_string);


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
    } else {
      log(user_exists.c_str(), rvs::logresults);
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
        //  return -1;
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
            user_group = user_group + " "
              + vector_iter->c_str() + " is member";

            log(user_group.c_str(), rvs::logresults);
            j = 1;
            break;
          }
        }

        // If the index is 0 then we user id doesn't match the group id
        if (j == 0) {
          // printf("user is not in the group\n");
          user_group = user_group + " "
            + vector_iter->c_str() + " is not member";
          log(user_group.c_str(), rvs::logresults);
        }
      }
    }
    return 0;
  }
  return -1;
}

/**
 * Check if the os and kernel version in the system match the givem os and kernel version
 * @param property config file map fields
 * @return 0 - success, non-zero otherwise
 * */

int action::kernelchk_run() {
  string os_version_values;
  string kernel_version_values;

  if (has_property(OS_VERSION, os_version_values)) {
    // Check kernel version
    if (has_property(KERNEL_VERSION,
      kernel_version_values) == false) {
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
    bool kernel_version_correct = false;

    // Check if the given kernel version matches one from the list
    vector<string>::iterator kernel_iter;
    for (kernel_iter = kernel_version_vector.begin()
        ; kernel_iter != kernel_version_vector.end(); kernel_iter++)
      if (kernel_actual.compare(*kernel_iter) == 0) {
        kernel_version_correct = true;
        break;
      }
      string result = "[rcqt] kernelcheck " + os_actual + " "
        + kernel_actual + " "
        + (os_version_correct && kernel_version_correct ? "pass" : "fail");
    log(result.c_str(), rvs::logresults);
    return 0;
  }

  return -1;
}

/**
 * Check if the shared object is in the given location with the correct architecture
 * @param property config file map fields
 * @return 0 - success, non-zero otherwise
 * */

int action::ldcfgchk_run() {
  string soname_requested;
  string arch_requested;
  string ldpath_requested;
  if (has_property(SONAME, soname_requested)) {
    if (has_property(ARCH, arch_requested) == false) {
      cerr << "acrhitecture field missing in conflig" << endl;
      return -1;
    }

    if (has_property(LDPATH, ldpath_requested) == false) {
      cerr << "libraty path field missing in conflig" << endl;
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
    } else if (pid > 0) {
      // Parent process
      char result[BUFFER_SIZE];
      close(fd[1]);

      read(fd[0], result, BUFFER_SIZE);
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
            if (arch_found.compare(arch_requested) == 0) {
              string arch_pass = ld_config_result + soname_requested
                + " " + full_ld_path + " " + arch_found + " pass";
              log(arch_pass.c_str(), rvs::logresults);
            } else {
              string arch_pass = ld_config_result + soname_requested + " "
                + full_ld_path + " " + arch_found + " fail";
              log(arch_pass.c_str(), rvs::logresults);
            }
          }
        }
      } else {
        string lib_fail = ld_config_result + soname_requested
          + " not found " + "na " + "fail";
        log(lib_fail.c_str(), rvs::logresults);
      }
    } else {
      cerr << "Internal Error" << endl;
      return -1;
    }
    return 0;
  }
  return -1;
}
