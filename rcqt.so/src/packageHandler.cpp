/********************************************************************************
 *
 * Copyright (c) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "include/rvsloglp.h"


const std::string metapack{"Meta Package"};

// expected versions usually either = or >=. 
bool isVersionMismatch(std::string expected, std::string installed){
  if(expected.find("+") == std::string::npos){
    return expected.compare(installed) != 0;
  }
  expected = expected.substr(0, expected.size()-1);
  auto exp = std::stof(expected);
  auto ins = std::stof(installed);
  return ins < exp;
}
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
  void *json_node =json_list_create(std::string(m_metapkg) ,rvs::loginfo);
	auto pkgmap = getPackageMap();
	if(pkgmap.empty()){
		std::cout << "Meta package not installed or no dependencies present !!!" << std::endl;
		return;
	}

	int totalPackages = 0, missingPackages = 0, badVersions = 0,
		installedPackages = 0;
	std::string cumulate_deps;
	for (const auto& val: pkgmap){
		++totalPackages;
                auto pkname = val.first.find(ORDEP) == std::string::npos ? val.first : val.first.substr(ORDEP.size());
		void *pkg_node = json_node_create( module_name, m_metapkg, rvs::loginfo);
                if (pkg_node){
                    if (val.first.find(ORDEP) != std::string::npos)
                        
                    rvs::lp::AddString(pkg_node, "depname", pkname);
                    rvs::lp::AddString(pkg_node, "expected", val.second);
		}
		auto inputname    = val.first;
		auto inputversion = val.second;
		auto installedvers = getInstalledVersion(inputname);
		if(installedvers.empty()){
			++missingPackages;
			std::cout << "Error: package " << inputname << " not installed " <<
					std::endl;
			rvs::lp::AddString(pkg_node, "installed", "N/A");
			rvs::lp::AddNode(json_node, pkg_node);
			continue;
		}

		if(isVersionMismatch(inputversion, installedvers)){
			++badVersions;
			std::cout << "Error: version mismatch for package " << inputname <<
					" expected version: " << inputversion << " but installed " <<
					installedvers << std::endl;
		} else {
			++installedPackages;
			auto tmp = inputname.find(ORDEP);
			if(tmp != std::string::npos)
				inputname = inputname.substr(ORDEP.size()); // Remove ordep identifier
			std::cout << "Package " << inputname << " installed version is " << 
					installedvers << std::endl;
			//rvs::lp::AddString(pkg_node, "installed",installedvers);
		}
                rvs::lp::AddString(pkg_node, "installed", installedvers);
                rvs::lp::AddNode(json_node, pkg_node);

	}

  msg = "Meta package validation complete : \n";
  msg += "\tTotal packages validated     : " + std::to_string(totalPackages) + "\n" + \
         "\tInstalled packages           : " + std::to_string(installedPackages) + "\n" + \
         "\tMissing packages             : " + std::to_string(missingPackages) + "\n" + \
         "\tVersion mismatch packages    : " + std::to_string(badVersions) + "\n";

  std::cout << msg;

  void *res_node = json_node_create(std::string(module_name),
                        m_metapkg, rvs::loginfo);
  rvs::lp::AddString(res_node, "Total Packages Validated", std::to_string(totalPackages));
  rvs::lp::AddString(res_node, "Installed Packages", std::to_string(installedPackages));
  rvs::lp::AddString(res_node, "Missing Packages", std::to_string(missingPackages));
  rvs::lp::AddString(res_node, "Mismatched Packages", std::to_string(badVersions));
  rvs::lp::AddString(res_node, "pass", totalPackages == installedPackages ? "true" : "false");
  rvs::lp::AddNode(json_node, res_node);
  rvs::lp::LogRecordFlush(json_node, rvs::logresults);

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
  std::vector<std::string> kv_pairs;
  auto pkglist = getPackageList();
  if(pkglist.empty()){
    std::cout << "no packages in the list" << std::endl;
    return;
  }
  int totalPackages = 0, missingPackages = 0, installedPackages = 0;
  for (const auto& pkgname: pkglist){

    ++totalPackages;

    kv_pairs.emplace_back(pkgname);
    auto installedversion = getInstalledVersion(pkgname);
    if(!installedversion.empty()){
      ++installedPackages;
      std::cout << "Package " << pkgname << " installed version is " << installedversion << std::endl;
      kv_pairs.emplace_back(installedversion);
    }
    else {
      ++missingPackages;
      std::cout << "Package " << pkgname << " not installed " << std::endl;
      kv_pairs.emplace_back("N/A");
      continue;
    }
  }
  kv_pairs.emplace_back("Installed Packages");
  kv_pairs.emplace_back(std::to_string(installedPackages));
  kv_pairs.emplace_back("Missing Packages");
  kv_pairs.emplace_back(std::to_string(missingPackages));
  kv_pairs.emplace_back("pass");
  kv_pairs.emplace_back(missingPackages == 0 ? "true" : "false");
  msg = "Packages install validation complete : \n";
  msg += "\tMissing packages      : " + std::to_string(missingPackages) + "\n";
  msg += "\tInstalled packages    : " + std::to_string(installedPackages) + "\n";

  std::cout << msg;
  log_to_json(rvs::logresults, kv_pairs);

  if(nullptr != callback) {
    rvs::action_result_t action_result;

    action_result.state = rvs::actionstate::ACTION_COMPLETED;
    action_result.status = rvs::actionstatus::ACTION_SUCCESS;
    action_result.output = msg.c_str();
    callback(&action_result, user_param);
  }

  return;
}


void PackageHandler::log_to_json(int log_level, std::vector<std::string> kvlist, void* parent) {
    if  (kvlist.size() == 0 || kvlist.size() %2 != 0){
            return;
    }
    void *json_node = json_node_create(std::string(module_name),
        action_name.c_str(), log_level);
    if (json_node) {
         for (int i =0; i< kvlist.size()-1; i +=2){
           rvs::lp::AddString(json_node, kvlist[i], kvlist[i+1]);
      }
      }
      rvs::lp::LogRecordFlush(json_node, log_level);
}
