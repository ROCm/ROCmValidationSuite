#include <iostream>
#include <sstream>
#include <fstream>
#include "include/zypPackageInfo.h"

bool ZypPackageInfo::readMetaPackageInfo(std::string ss){

  std::stringstream inss{ss};
  std::string line;
  bool found = false;
  size_t pos = std::string::npos;
  std::ofstream os;

  os.open(getFileName() , std::ofstream::out | std::ofstream::app);

  while(std::getline(inss, line)){

    pos = line.find("Requires");

    if(std::string::npos != pos){

      while(std::getline(inss, line)){

        pos = line.find("=");

        if(std::string::npos != pos){

          /* Get dependent package and its version */

          size_t firstCharPos = line.find_first_not_of(" ");
          size_t midSpacePos = line.find(" ", firstCharPos);

          auto depPackageName = line.substr(firstCharPos, midSpacePos - firstCharPos);

          midSpacePos = line.find_last_of(" ");
          auto depPackageVersion = line.substr(midSpacePos + 1);

          /* Write dependent package name and its version to file */
          os << depPackageName << " " << depPackageVersion << std::endl;

          pos = std::string::npos;

          if (false == found){
            found = true;
          }
        }
      }
    }
  }

  os.close();

  return found;
}
