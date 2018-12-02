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

#include "rvs_key_def.h"
#include "rvsloglp.h"

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

using std::string;
using std::iterator;
using std::endl;
using std::ifstream;
using std::map;
using std::regex;
using std::vector;

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
  string msg;
  bool pkgchk_bool = false;
  bool usrchk_bool = false;
  bool kernelchk_os_bool = false;
  bool ldcfgchk_so_bool = false;
  bool filechk_bool = false;

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
                            action_name.c_str(), rvs::loginfo, sec, usec);
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
		/*
    pid_t pid;
    int fd[2];
    if (pipe(fd) == -1) {
      rvs::lp::Err("pipe() error", MODULE_NAME_CAPS, action_name);
      return 1;
    }
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
      if (system(buffer) == -1) {
        rvs::lp::Err("system() error", MODULE_NAME_CAPS, action_name);
        return 1;
      }
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
      string passed =  "[" + action_name + "] " + "rcqt packagecheck "
      + package_name + " true";
      string failed =  "[" + action_name + "] " + "rcqt packagecheck "
      + package_name + " false";
				*/
			///std::regex version_pattern(version_name);
      /* 
       * If result start with dpkg-querry: then we haven't found the package
       * If we get something different, then we confirme that the package is found
       * if version is equal to the required then the test pass
       */
/*
		
      if (strstr(result, "dpkg-query:") == result) {
        log(failed.c_str(), rvs::logresults);
        if (bjson && json_rcqt_node != nullptr) {
          rvs::lp::AddString(json_rcqt_node, package_name, "not exists");
          rvs::lp::AddString(json_rcqt_node, "pkgchk", "false");
          rvs::lp::LogRecordFlush(json_rcqt_node);
        }
      } else if (version_exists == false) {
        log(passed.c_str(), rvs::logresults);
        if (bjson && json_rcqt_node != nullptr) {
          rvs::lp::AddString(json_rcqt_node, package_name, "exists");
          rvs::lp::AddString(json_rcqt_node, "pkgchk", "true");
          rvs::lp::LogRecordFlush(json_rcqt_node);
        }
      } else if (regex_match(version_value, version_pattern) == true) {
        log(passed.c_str(), rvs::logresults);
        if (bjson && json_rcqt_node != nullptr) {
          rvs::lp::AddString(json_rcqt_node, package_name, "exists");
          rvs::lp::AddString(json_rcqt_node, version_name, "exists");
          rvs::lp::AddString(json_rcqt_node, "pkgchk", "true");
          rvs::lp::LogRecordFlush(json_rcqt_node);
        }
      } else {
        log(failed.c_str(), rvs::logresults);
        if (bjson && json_rcqt_node != nullptr) {
          rvs::lp::AddString(json_rcqt_node, package_name, "exists");
          rvs::lp::AddString(json_rcqt_node, version_name, "not exists");
          rvs::lp::AddString(json_rcqt_node, "pkgchk", "false");
          rvs::lp::LogRecordFlush(json_rcqt_node);
        }
      }
    } else {
      // fork process error
      rvs::lp::Err("INTERNAL_ERROR", MODULE_NAME_CAPS, action_name);
      return -1;
    }
    */
		pid_t pid;
		int fd[2];
		if (pipe(fd) == -1) {
			rvs::lp::Err("pipe() error", MODULE_NAME_CAPS, action_name);
			return 1;
		}
		pid = fork();
		if (pid == 0) {
			// Child process
			// Pipe the standard output to the fd[1]
			dup2(fd[1], STDOUT_FILENO);
			dup2(fd[1], STDERR_FILENO);
			//char buffer[BUFFER_SIZE];
			string command_string = "dpkg --get-selections > ";
			command_string += PKG_CMD_FILE;
			//snprintf(buffer, BUFFER_SIZE, command_string.c_str());
		 
			// We execute the dpkg-querry
			if (system(command_string.c_str()) == -1) {
				rvs::lp::Err("system() error", MODULE_NAME_CAPS, action_name);
				return 1;
			}
			exit(0);
		} else if (pid > 0) {
			// Parent
			char result_cmnd[BUFFER_SIZE];
			int count;
			close(fd[1]);
			std::cout  << package_name << std::endl;
			// We read the result from the dpk-querry from the fd[0]
			vector <string> package_vector;
			bool package_found = false;
			count = read(fd[0], result_cmnd, BUFFER_SIZE);
			std::regex pkg_pattern(package_name);
			result_cmnd[count] = 0;
			string result_cmnd_string = result_cmnd;
			ifstream pkg_stream(std::string(PKG_CMD_FILE));
			char file_line[STREAM_SIZE];
			while (pkg_stream.getline(file_line, STREAM_SIZE)) {
				string line_result = file_line;
				line_result = line_result.substr(0, line_result.length() - 7);
				line_result.erase(line_result.find_last_not_of(" \n\r\t")+1);
				//cout << line_result << "***" << endl;
				if (regex_match(line_result, pkg_pattern) == true) {
					std::cout << line_result << std::endl;
					if (version_exists) {
						char cmd_buffer[BUFFER_SIZE];
						snprintf(cmd_buffer, BUFFER_SIZE, \
						"dpkg -s %s | grep -i version > %s", line_result.c_str(),(std::string(VERSION_FILE)).c_str());
						std::cout << cmd_buffer << std::endl;
						if (system(cmd_buffer) == -1) {
							rvs::lp::Err("system() error", MODULE_NAME_CAPS, action_name);
							return 1;
						}
						std::regex version_pattern(std::string("Version: ") + version_name);
						ifstream version_stream(std::string(VERSION_FILE));
						char file_line[STREAM_SIZE];
						version_stream.getline(file_line, STREAM_SIZE);
						std::cout << "==="<< std::string(file_line) << std::endl;
						string line_result = file_line;
						if (regex_match(line_result, version_pattern) == true) {
							string package_exists = "[" + action_name + "] " + "rcqt pkgcheck * "
							+ line_result.substr(8, line_result.size()-8) + " true";
							rvs::lp::Log(package_exists, rvs::logresults);
							package_found = true;
						} else {
							string pkg_not_exists = "[" + action_name + "] " + "rcqt pkgcheck &"
							+ package_name + " false";
							rvs::lp::Log(pkg_not_exists, rvs::logresults);
							package_found = true;
						}
					} else {
						string package_exists = "[" + action_name + "] " + "rcqt pkgcheck #"
						+ line_result + " true";
						rvs::lp::Log(package_exists, rvs::logresults);
						package_found = true;
					}
				}
			}
			if (!package_found) {
				string pkg_not_exists = "[" + action_name + "] " + "rcqt pkgcheck "
				+ package_name + " false";
				rvs::lp::Log(pkg_not_exists, rvs::logresults);
			}
			
		}
		//cout << " ***" << endl;
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
		/*
    // Structures for checking group and user
    struct passwd pwd, *result;

    char pwdbuffer[2000];
    int pwdbufflenght = 2000;
    struct group grp, *grprst;
    string user_exists = "[" + action_name + "] " + "rcqt usercheck "
        + user_name + " true";
        string user_not_exists = "[" + action_name + "] " + "rcqt usercheck "
        + user_name + " false";

    // Check for given user
    if (getpwnam_r(user_name.c_str()
      , &pwd, pwdbuffer, pwdbufflenght, &result) != 0) {
      rvs::lp::Err("Error with getpwnam_r", MODULE_NAME_CAPS, action_name);
      return 1;
    }
    if (result == nullptr) {
      log(user_not_exists.c_str(), rvs::logresults);
      if (bjson && json_rcqt_node != nullptr) {
        rvs::lp::AddString(json_rcqt_node, user_name, " false");
      }
    } else {
      log(user_exists.c_str(), rvs::logresults);
      if (bjson && json_rcqt_node != nullptr) {
        rvs::lp::AddString(json_rcqt_node, user_name, " true");
      }
    }
    if (group_exists && result != nullptr) {
      // Put the group list into vector
      string delimiter = ",";
      vector<string> group_vector;
      group_vector = str_split(group_values_string, delimiter);

      // Check if the group exists
      for (vector<string>::iterator vector_iter = group_vector.begin()
          ; vector_iter != group_vector.end(); vector_iter++) {
        string user_group = "rcqt usercheck " + user_name;
        int error_group;

        if ((error_group =  getgrnam_r(vector_iter->c_str()
          , &grp, pwdbuffer, pwdbufflenght, &grprst)) != 0) {
          rvs::lp::Err("Error with getpwnam_r", MODULE_NAME_CAPS, action_name);
          return 1;
        }
        if (error_group == EIO) {
          rvs::lp::Err("IO error", MODULE_NAME_CAPS, action_name);
          return 1;
        } else if (error_group == EINTR) {
          rvs::lp::Err("Error sginal was caught during getgrnam_r"
              , MODULE_NAME_CAPS, action_name);
          return 1;
        } else if (error_group == EMFILE) {
          rvs::lp::Err("Error file descriptors are currently open"
              , MODULE_NAME_CAPS, action_name);
          return 1;
        } else if (error_group == ERANGE) {
          rvs::lp::Err("Error insufficient buffer in getgrnam_r"
              , MODULE_NAME_CAPS, action_name);
          return 1;
        }
        string err_msg;
        if (grprst == nullptr) {
          err_msg = "group ";
          err_msg += vector_iter->c_str();
          err_msg += " does not exist";
          rvs::lp::Err(err_msg, MODULE_NAME, action_name);
          return 1;
        }

        int i;
        int j = 0;

        // Compare if the user group id is equal to the group id
        for (i = 0; grp.gr_mem[i] != NULL; i++) {
          if (strcmp(grp.gr_mem[i], user_name.c_str()) == 0) {
            user_group = "[" + action_name + "] " + user_group + \
            " " + vector_iter->c_str() + " true";
            log(user_group.c_str(), rvs::logresults);
            if (bjson && json_rcqt_node != nullptr) {
              rvs::lp::AddString(json_rcqt_node
              , user_group + " " + vector_iter->c_str(), " true");
            }
            j = 1;
            break;
          }
        }

        // If the index is 0 then we user id doesn't match the group id
        if (j == 0) {
          // printf("user is not in the group\n");

          user_group = "[" + action_name + "] " + user_group + " " \
          + vector_iter->c_str() + " false";
          log(user_group.c_str(), rvs::logresults);
          if (bjson && json_rcqt_node != nullptr) {
            rvs::lp::AddString(json_rcqt_node,
              user_group + " " + vector_iter->c_str(), "false");
          }
          j = 1;
        }
      }
    }
    if (bjson && json_rcqt_node != nullptr) {
      rvs::lp::LogRecordFlush(json_rcqt_node);
    }
    return 0;
		*/
		vector<string> users_vector;
		ifstream passwd_stream(ETC_PASSWD);
		char file_line[STREAM_SIZE];
		regex usr_pattern(user_name);
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
			vector<regex>group_patterns;
			vector<string> group_vector;
			group_vector = str_split(group_values_string, ",");
			vector<string> group_found_vector;
			for (vector<string>::iterator it = group_vector.begin();
						it != group_vector.end(); it++) {
				//cout << *it << endl;
				group_patterns.push_back(std::regex(*it));
			}
			ifstream group_stream(ETC_GROUP);
			while (group_stream.getline(file_line, STREAM_SIZE)) {
				const string line = std::string(file_line);
				std::smatch match;
				const std::regex get_group_pattern(GRP_REG);
				if (std::regex_search(line.begin(), line.end()
							, match, get_group_pattern)) {
					if (std::regex_search(line.begin(), line.end()
						, match, get_group_pattern)) {
						string result = match[1];
						//cout << result << endl;
						
						for (vector<regex>::iterator it = group_patterns.begin();
								it != group_patterns.end(); it++) {
							if (std::regex_match(result, *it)) {
								group_found_vector.push_back(result);
								std::cout << result << std::endl;
								std::cout << "group users " << match[4] << std::endl;
								vector<string> group_users_found =
								str_split(match[4], ",");
								std::set<string>user_group_set (group_users_found.begin()
									, group_users_found.end());
								for (string user_string : users_vector) {
									std::cout << "User " << user_string << std::endl;
									if (user_group_set.find(user_string) 
											!= user_group_set.end()) {
										string user_group_found = "[" + action_name + "] " 
										+ result + \
										" " + user_string+ " true";
										rvs::lp::Log(user_group_found, rvs::logresults);
									} else {
										string user_group_found = "[" + action_name + "] " 
										+ result + \
										" " + user_string+ " false";
										rvs::lp::Log(user_group_found, rvs::logresults);
									}
								}
							}
						}
					}
				}
			}
			if (group_found_vector.empty()) {
				string groups_not_found = "[" + action_name + "] " 
				+ "rcqt" + " group "
				+ group_values_string + " not found";
				rvs::lp::Log(groups_not_found, rvs::logerror);
			}
		}
		
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
          //if (strcmp(os_iter->c_str(), os_actual.c_str()) == 0) {
					if(regex_match(os_actual, os_pattern) == true) {
            os_version_correct = true;
            break;
          }
        }
        if (os_version_correct == true)
          break;
      }
    }
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
    log(result.c_str(), rvs::logresults);
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

int action::ldcfgchk_run() {
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
    char cmd_buffer[BUFFER_SIZE];
		snprintf(cmd_buffer, BUFFER_SIZE, \
		"ls -p %s | grep -v / > %s", ldpath_requested.c_str(), (char*)LDCFG_FILE);
		
		if (system(cmd_buffer) == -1) {
			rvs::lp::Err("system() error", MODULE_NAME_CAPS, action_name);
			return 1;
		}
		ifstream lib_stream(std::string(LDCFG_FILE));
		char file_line[STREAM_SIZE];
		vector<string> found_files_vector;
		std::regex file_pattern(soname_requested);
		while (lib_stream.getline(file_line, STREAM_SIZE)) {
			//std::cout << file_line << std::endl;
			if (regex_match(std::string(file_line), file_pattern))
				found_files_vector.push_back(std::string(file_line));
		}
		string arch_found_string;
		bool arch_found_bool = false;
		string ld_config_result = "[" + action_name + "] " +
		"rcqt ldconfigcheck ";
		for (auto it = found_files_vector.begin(); it != found_files_vector.end(); it++) {
			// Full path of shared object
			string full_ld_path = ldpath_requested + "/" + std::string(*it);
			snprintf(cmd_buffer, sizeof(cmd_buffer), "file %s | grep \"shared object,\" > %s", full_ld_path.c_str(), (char*)LDCFG_FILE);
			ifstream arch_stream(std::string(LDCFG_FILE));
// // // // // // // // // // 			//std::cout << cmd_buffer << std::endl;
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
			char arch_cmd_buffer[BUFFER_SIZE];
			snprintf(arch_cmd_buffer, sizeof(arch_cmd_buffer), "objdump -f %s | grep ^architecture> %s",full_ld_path.c_str(), (char*)LDCFG_FILE);
			if (system(arch_cmd_buffer) == -1) {
				rvs::lp::Err("system() error", MODULE_NAME_CAPS, action_name);
				return 1;
			}
			arch_stream.open(std::string(LDCFG_FILE));
			arch_stream.getline(file_line, STREAM_SIZE);
			arch_found_string = file_line;
			arch_found_string = str_split(arch_found_string, ",")[0];
			arch_found_string.erase(0, arch_found_string.find_first_not_of("architecture: ") - 1);
			arch_found_bool = true;
			//std::cout << arch_found_string << "***" << arch_requested << std::endl;
			if (regex_match(arch_found_string, arch_pattern)) {
				string arch_pass = ld_config_result + *it
				+ " " + arch_found_string + " " + ldpath_requested + " pass";
				log(arch_pass.c_str(), rvs::logresults);
				
			} else {
				string arch_fail = ld_config_result + *it
				+ " NA " + ldpath_requested + " fail";
				log(arch_pass.c_str(), rvs::logresults);
			}
			arch_stream.close();
		}
		if (!arch_found_bool) {
			string lib_fail = ld_config_result
			+ " not found NA " +  + ldpath_requested +  " fail";
			log(lib_fail.c_str(), rvs::logresults);
			if (bjson && json_rcqt_node != nullptr) {
				rvs::lp::AddString(json_rcqt_node, "soname", soname_requested);
				rvs::lp::AddString(json_rcqt_node, "ldchk", "false");
			}
		}
			/*
    // Full path of shared object
    string full_ld_path = ldpath_requested + "/" + soname_requested;

    int fd[2];
    pid_t pid;
    if (pipe(fd) == -1) {
      rvs::lp::Err("pipe() error", MODULE_NAME_CAPS, action_name);
      return 1;
    }
    pid = fork();
    if (pid == 0) {
      // child process
      dup2(fd[1], STDOUT_FILENO);
      dup2(fd[1], STDERR_FILENO);
      char buffer[256];
      snprintf(buffer, sizeof(buffer), "objdump -f %s", full_ld_path.c_str());

      if (system(buffer) == -1) {
        rvs::lp::Err("system() error", MODULE_NAME_CAPS, action_name);
        return 1;
      }
      exit(0);
    } else if (pid > 0) {
      // Parent process
      char result[BUFFER_SIZE];
      close(fd[1]);
      string ld_config_result = "[" + action_name + "] " +
      "rcqt ldconfigcheck ";
      if (read(fd[0], result, BUFFER_SIZE) < 0) {
        rvs::lp::Err("read() error", MODULE_NAME_CAPS, action_name);
        return 1;
      }
      string result_string = result;
			std::cout << "***" << result_string << "***"<< std::endl;
      if (strstr(result, "architecture:") != nullptr) {
        vector<string> objdump_lines = str_split(result_string, "\n");
        int begin_of_the_arch_string = 0;
        int end_of_the_arch_string = 0;
        for (uint i = 0; i < objdump_lines.size(); i++) {
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
                + " " + full_ld_path + " " + arch_found + " true";
              log(arch_pass.c_str(), rvs::logresults);
              if (bjson && json_rcqt_node != nullptr) {
                rvs::lp::AddString(json_rcqt_node, "ldchk", "true");
              }
            } else {
              string arch_pass = ld_config_result + soname_requested + " "
                + full_ld_path + " " + arch_found + " false";
              log(arch_pass.c_str(), rvs::logresults);
              if (bjson && json_rcqt_node != nullptr) {
                rvs::lp::AddString(json_rcqt_node, "ldchk", "false");
              }
            }
          }
        }
      } else {
        string lib_fail = ld_config_result + soname_requested
        + " " + full_ld_path + " " + arch_requested + " false";
        log(lib_fail.c_str(), rvs::logresults);
        if (bjson && json_rcqt_node != nullptr) {
          rvs::lp::AddString(json_rcqt_node, "soname", soname_requested);
          rvs::lp::AddString(json_rcqt_node, "ldchk", "false");
        }
      }
    } else {
      rvs::lp::Err("Internal Error", MODULE_NAME_CAPS, action_name);
      return 1;
    }
    if (bjson && json_rcqt_node != nullptr) {
      rvs::lp::LogRecordFlush(json_rcqt_node);
    }*/
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
	std::cout << file << std::endl;
	std::size_t found = file.find_last_of("/\\");
	std::cout << file.substr(0,found) << '\n';
	std::cout << file.substr(found+1) << '\n';
	string file_path = file.substr(0,found);
	string file_requested = file.substr(found+1);
	char cmd_buffer[BUFFER_SIZE];
	snprintf(cmd_buffer, BUFFER_SIZE, \
	"ls %s | grep -v / > %s", file_path.c_str(), (char*)LDCFG_FILE);
	
	if (system(cmd_buffer) == -1) {
		rvs::lp::Err("system() error", MODULE_NAME_CAPS, action_name);
		return 1;
	}
	ifstream file_stream(std::string(LDCFG_FILE));
	char file_line[STREAM_SIZE];
	vector<string> found_files_vector;
	std::regex file_pattern(file_requested);
	while (file_stream.getline(file_line, STREAM_SIZE)) {
		//std::cout << file_line << std::endl;
		if (regex_match(std::string(file_line), file_pattern))
			found_files_vector.push_back(std::string(file_line));
	}
	if (exists == false && found_files_vector.empty()) {
		check = "true";
		msg = "[" + action_name + "] " + "rcqt filecheck "+ file_path +" DNE " + check;
		log(msg.c_str(), rvs::logresults);
		if (bjson && json_rcqt_node != nullptr) {
			rvs::lp::AddString(json_rcqt_node
			, "exists", file_path);
		}
	}
	if (found_files_vector.empty() && exists == true) {
		check = "false";
		msg = "[" + action_name + "] " + "rcqt filecheck "+ file_path +" DNE " + check;
		log(msg.c_str(), rvs::logresults);
		if (bjson && json_rcqt_node != nullptr) {
			rvs::lp::AddString(json_rcqt_node
			, "exists", file_path);
		}
	}
	for (auto file_it = found_files_vector.begin();
			 file_it != found_files_vector.end(); file_it++) {
		file = file_path + "/" + std::string(*file_it);
		std::cout << file << std::endl;
  // check if exists property corresponds to real existence of the file
  if (exists == false) {
    if (stat(file.c_str(), &info) < 0)
      check = "true";
    else
      check = "false";
    msg = "[" + action_name + "] " + "rcqt filecheck "+ file +" DNE " + check;
    log(msg.c_str(), rvs::logresults);
    if (bjson && json_rcqt_node != nullptr) {
      rvs::lp::AddString(json_rcqt_node
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
        //if (p.pw_name == owner)
        if (regex_match(p.pw_name, owner_pattern))
          check = "true";
        else
          check = "false";
        msg = "[" + action_name + "] " + "rcqt filecheck " \
        + owner +" " + check;
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
				std::regex group_pattern(group);
        struct group g, *result;
        char pbuff[256];
        if ((getgrgid_r(info.st_gid, &g, pbuff, sizeof(pbuff), &result) != 0)) {
          rvs::lp::Err("Error with getgrgid_r", MODULE_NAME, action_name);
          return 1;
        }
        //if (g.gr_name == group)
        if (regex_match(g.gr_name, group_pattern))
          check = "true";
        else
          check = "false";
        msg = "[" + action_name + "] " + "rcqt filecheck " + group+ " " + check;
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
        msg = "[" + action_name + "] " + "rcqt filecheck " + \
        std::to_string(permission)+" "+check;
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
          msg = "[" + action_name + "] " + "rcqt filecheck " + \
          std::to_string(type)+" "+check;
          log(msg.c_str(), rvs::logresults);
          if (bjson && json_rcqt_node != nullptr) {
            rvs::lp::AddString(json_rcqt_node
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
  return 0;
}
