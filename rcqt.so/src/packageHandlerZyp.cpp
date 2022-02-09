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
	metaInfo.reset(new ZypPackageInfo(pkgname,
                  std::string("zypper"), std::string("info"), std::string("--requires")));
  metaInfo->fillPkgInfo();
  m_manifest = metaInfo->getFileName();
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
  int read_pipe[2]; // From child to parent
  int exit_status;
	package_info pinfo;
  if(pipe(read_pipe) == -1){
    perror("Pipe");
    return pinfo.version;
  }
  pid_t process_id = fork();
  if(process_id < 0){
    perror("Fork");
    return pinfo.version;

  }else if(process_id == 0) {
    dup2(read_pipe[1], 1);
    close(read_pipe[0]);
    close(read_pipe[1]);
    execlp("zypper", "zypper", "info", package.c_str(), NULL);
  } else {
    // parent:
    int status;
    waitpid(process_id, &status,0);
    std::stringstream ss;
    close(read_pipe[1]);
    {
      char arr[4096];
      int n = read(read_pipe[0], arr, sizeof(arr));
      ss.write(arr, n);

    }
    //std::cout << ss.str() << std::endl;
    close(read_pipe[0]);
    std::string ver_string{};
    auto res = pkgrOutputParser(ss.str(), pinfo);
    if(!res){
        std::cout << "error in parsing" << std::endl;
        return std::string{};
    }
    //std::cout << pinfo.name << " and " << pinfo.version << std::endl;
    return pinfo.version;
  }
}

void PackageHandlerZyp::validatePackages(){
  std::cout << "MANOJ: file nameis " << m_manifest << std::endl;
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
