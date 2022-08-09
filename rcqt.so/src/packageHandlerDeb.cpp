#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include "include/packageHandlerDeb.h"
#include "include/debPackageInfo.h"

PackageHandlerDeb::PackageHandlerDeb(std::string pkgname): PackageHandler{}{
  
  const std::vector<std::string> cmd {std::string("--status"), std::string(""), std::string("--status")};

	metaInfo.reset(new DebPackageInfo(pkgname, cmd));
  metaInfo->fillPkgInfo();
  m_manifest = metaInfo->getFileName();
}

PackageHandlerDeb::PackageHandlerDeb(): PackageHandler{}{
  
  const std::vector<std::string> cmd {std::string("--status"), std::string(""), std::string("--status")};

	metaInfo.reset(new DebPackageInfo(cmd));
}

bool PackageHandlerDeb::pkgrOutputParser(const std::string& s_data, package_info& info){
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

std::string PackageHandlerDeb::getInstalledVersion(const std::string& package){

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

