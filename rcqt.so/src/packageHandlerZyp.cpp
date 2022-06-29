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
    std::cout << "error in parsing" << std::endl;
    return std::string{};
  }
  return pinfo.version;
}

