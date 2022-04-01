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

	metaInfo.reset(new ZypPackageInfo(pkgname, std::string("zypper"), cmd));
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

#if 0
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
#endif

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

