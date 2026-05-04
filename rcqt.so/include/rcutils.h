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

#ifndef UTILS_H
#define UTILS_H
#include <vector>
#include <map>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <chrono>

const std::string ORDEP{"ORDEP:"};
enum class OSType {
  Ubuntu,
  Centos,
  SLES,
  RHEL,
  Oracle,
  Azure,
  Amazon,
  Alibaba,
  None
};

const std::string os_release_file {"/etc/os-release"};
const std::string name_key {"NAME"};
const std::string id_key {"ID"};
const std::map<std::string, OSType> op_systems {
  {"ubuntu", OSType::Ubuntu},
  {"centos", OSType::Centos},
  {"sles", OSType::SLES},
  {"red hat enterprise linux", OSType::RHEL},
  {"oracle linux server", OSType::Oracle},
  {"microsoft azure linux", OSType::Azure},
  {"amazon linux", OSType::Amazon},
  {"alibaba cloud linux", OSType::Alibaba},
};

struct package_info{
	std::string name{};
	std::string version{};
};

// common funtions
std::string get_last_word(const std::string& input);
OSType getOSOrId();
OSType getOS(std::string keyname);
std::string remSpaces(std::string str);
std::string pfilename(const std::string& package);

bool getPackageInfo(const std::string& package,
    const std::string& packagemgr,
    const std::string& command,
    const std::string& option,
    std::stringstream &ss);

#endif
