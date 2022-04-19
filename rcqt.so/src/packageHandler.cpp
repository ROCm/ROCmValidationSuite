#include <fstream>
#include <assert.h>
#include <sstream>
#include <iostream>
#include "include/packageHandler.h"

bool PackageHandler::parseManifest(){
	assert(!m_manifest.empty());
	std::ifstream ifs {m_manifest};
	std::cout << "Manifest name is " << m_manifest << std::endl;
	assert(ifs.good());
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
	//std::cout << m_pkgversionmap.size() << " packages to validate " << std::endl;
	return true;
}

void PackageHandler::validatePackages(){
  std::cout << "File name is " << m_manifest << std::endl;
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
	std::cout << "RCQT complete : " << std::endl;
	std::cout << "\tTotal Packages to validate    : " << totalPackages     << std::endl;
	std::cout << "\tValid Packages                : " << installedPackages << std::endl;
	std::cout << "\tMissing Packages              : " << missingPackages   << std::endl;
	std::cout << "\tPackages version mismatch     : " << badVersions       << std::endl;
	return ;	
}

void PackageHandler::listPackageVersion(){

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
			std::cout << "Package " << pkgname << " installed version is " << 
					installedversion << std::endl;
    }
    else {
      ++missingPackages;
      std::cout << "Package " << pkgname << " not installed " <<
        std::endl;
      continue;
    }
	}

	std::cout << "RCQT complete : " << std::endl;
	std::cout << "\t Total Packages installed : " << installedPackages << std::endl;
	return;
}

