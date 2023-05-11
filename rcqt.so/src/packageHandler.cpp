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

#include <fstream>
#include <sstream>
#include <iostream>
#include "include/packageHandler.h"

bool PackageHandler::parseManifest(){

  if(m_manifest.empty()) {
    std::cout  << "Error: Manifest is not filled !!!" << std::endl;
    return false;
  }

	std::ifstream ifs {m_manifest};
	std::string line;

	while(std::getline(ifs, line)){
		if(line.empty())
			continue;
		std::istringstream iss{line};
		std::string name, ver;
		iss >> name >> ver;
		if(!iss.eof()){
			std::cout  << "error line format" << line << std::endl; //crude validation
			return false;
		}
		m_pkgversionmap.emplace(name, ver);
	}
	return true;
}

void PackageHandler::validatePackages(){

  std::string msg;

	auto pkgmap = getPackageMap();
	if(pkgmap.empty()){
		std::cout << "no packages to validate in the file " << std::endl;
		return;
	}

	int totalPackages = 0, missingPackages = 0, badVersions = 0,
		installedPackages = 0;
	for (const auto& val: pkgmap){
		++totalPackages;
		auto inputname    = val.first;
		auto inputversion = val.second;
		auto installedvers = getInstalledVersion(inputname);
		if(installedvers.empty()){
			++missingPackages;
			std::cout << "Error: package " << inputname << " not installed " <<
					std::endl;
			continue;
		}

		if( inputversion.compare(installedvers)){
			++badVersions;
			std::cout << "Error: version mismatch for package " << inputname <<
					" expected version: " << inputversion << " but installed " <<
					installedvers << std::endl;
		} else {
			++installedPackages;
			std::cout << "Package " << inputname << " installed version is " << 
					installedvers << std::endl;
		}
	}

  msg = "RCQT complete : \n";
  msg += "\tTotal Packages to validate    : " + std::to_string(totalPackages) + "\n" + \
         "\tValid Packages                : " + std::to_string(installedPackages) + "\n" + \
         "\tMissing Packages              : " + std::to_string(missingPackages) + "\n" + \
         "\tPackages version mismatch     : " + std::to_string(badVersions) + "\n";

  std::cout << msg;

  if(nullptr != callback) {
    rvs::action_result_t action_result;

    action_result.state = rvs::actionstate::ACTION_COMPLETED;
    action_result.status = rvs::actionstatus::ACTION_SUCCESS;
    action_result.output = msg.c_str();
    callback(&action_result, user_param);
  }

	return ;
}

void PackageHandler::listPackageVersion(){

  std::string msg;

	auto pkglist = getPackageList();
	if(pkglist.empty()){
		std::cout << "no packages in the list" << std::endl;
		return;
	}
	int totalPackages = 0, missingPackages = 0, installedPackages = 0;

	for (const auto& pkgname: pkglist){

    ++totalPackages;

    auto installedversion = getInstalledVersion(pkgname);
    if(!installedversion.empty()){
			++installedPackages;
			std::cout << "Package " << pkgname << " installed version is " << installedversion << std::endl;
    }
    else {
      ++missingPackages;
      std::cout << "Package " << pkgname << " not installed " << std::endl;
      continue;
    }
	}

	msg = "RCQT complete : \n";
	msg += "\t Total Packages installed : " + std::to_string(installedPackages) + "\n";

  std::cout << msg;

  if(nullptr != callback) {
    rvs::action_result_t action_result;

    action_result.state = rvs::actionstate::ACTION_COMPLETED;
    action_result.status = rvs::actionstatus::ACTION_SUCCESS;
    action_result.output = msg.c_str();
    callback(&action_result, user_param);
  }

	return;
}

