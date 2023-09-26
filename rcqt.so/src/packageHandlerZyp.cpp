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

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include "include/packageHandlerZyp.h"
#include "include/zypPackageInfo.h"

PackageHandlerZyp::PackageHandlerZyp(std::string pkgname): PackageHandler{} {

  const std::vector<std::string> cmd {std::string("info"), std::string("--requires"), std::string("info")};

	metaInfo.reset(new ZypPackageInfo(pkgname, cmd));
  metaInfo->fillPkgInfo();
  m_manifest = metaInfo->getFileName();
}

PackageHandlerZyp::PackageHandlerZyp(): PackageHandler{} {

  const std::vector<std::string> cmd {std::string("info"), std::string("--requires"), std::string("info")};

	metaInfo.reset(new ZypPackageInfo(cmd));
}

bool PackageHandlerZyp::pkgrOutputParser(const std::string& s_data, package_info& info){

  std::stringstream data{s_data};
  std::string line;
  bool found = false;
  int pos = 0;

  while(std::getline(data, line)){

    if( line.find("Name") != std::string::npos){
      info.name = get_last_word(line);
      found = true;
    }
    else if(line.find("Version") != std::string::npos){

      info.version = get_last_word(line);
      pos = info.version.find("-");
      info.version = info.version.substr(0, pos);

      if(found) 
        return found;
    }
  }

  return found;
}

std::string PackageHandlerZyp::getInstalledVersion(const std::string& package){

  package_info pinfo;
  std::stringstream ss;
  bool status;

  status = getPackageInfo(package, metaInfo->getPackageMgrName(), metaInfo->getInfoCmdName(), "", ss);
  if (true != status) {
    std::cout << "getPackageInfo failed !!!" << std::endl;
    return std::string{};
  }

  auto res = pkgrOutputParser(ss.str(), pinfo);
  if(!res){
//    std::cout << "error in parsing" << std::endl;
    return std::string{};
  }
  return pinfo.version;
}

