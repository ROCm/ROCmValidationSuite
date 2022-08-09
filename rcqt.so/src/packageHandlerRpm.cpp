#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include "include/packageHandlerRpm.h"
#include "include/rpmPackageInfo.h"

PackageHandlerRpm::PackageHandlerRpm(std::string pkgname): PackageHandler{} {

  const std::vector<std::string> cmd {std::string("-qR"), std::string(""), std::string("-qi")};

	metaInfo.reset(new RpmPackageInfo(pkgname, cmd));
  metaInfo->fillPkgInfo();
  m_manifest = metaInfo->getFileName();
}

PackageHandlerRpm::PackageHandlerRpm(): PackageHandler{} {

  const std::vector<std::string> cmd {std::string("-qR"), std::string(""), std::string("-qi")};

	metaInfo.reset(new RpmPackageInfo(cmd));
}

bool PackageHandlerRpm::pkgrOutputParser(const std::string& s_data, package_info& info){
  std::stringstream data{s_data};
  // first line tells if we need to proceed or not.
  std::string line;
  bool found = false;
  while(std::getline(data, line)){
    if(line.find("Version") != std::string::npos){
      info.version = get_last_word(line);
      found = true;
    } else if( line.find("Package") != std::string::npos){
      info.name = get_last_word(line);
      if(found) // prevent further processing
        return found;
    }
  }
  return found;
}

std::string PackageHandlerRpm::getInstalledVersion(const std::string& package){

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
    std::cout << "error in parsing" << std::endl;
    return std::string{};
  }
  return pinfo.version;
}
