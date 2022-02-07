#include <iostream>
#include <sstream>
#include <fstream>
#include "include/rpmPackageInfo.h"

bool RpmPackageInfo::readMetaPackageInfo(std::string ss){

  std::stringstream inss{ss};
  std::string line;
  bool found = false;
  size_t pos = std::string::npos;
  std::ofstream os;

  os.open(getFileName() , std::ofstream::out | std::ofstream::app);
	std::cout << "DEBUG: file is " << getFileName() << std::endl;

  while(std::getline(inss, line)){

    pos = line.find("=");

    if(std::string::npos != pos){

      /* Get depend package and its version */
      auto firstSpacePos = line.find(" ");

      auto depPackageName = line.substr(0, firstSpacePos);
      auto depPackageVersion = line.substr(pos + 2);

      /* Write depend package name and its version to file */
      os << depPackageName << " " << depPackageVersion << std::endl;

      pos = std::string::npos;

      if (false == found){
        found = true;
      }
    }
  }

  os.close();

  return found;
}
