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
