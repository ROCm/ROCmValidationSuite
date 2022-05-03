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

#if 0
void getPackageInfo(const std::string& package, std::stringstream &ss){

  int read_pipe[2]; // From child to parent
  int exit_status;
	package_info pinfo;
  if(pipe(read_pipe) == -1){
    perror("Pipe");
//    return pinfo.version;
  }

  std::cout << "Rpm getInstalled" << std::endl;

  pid_t process_id = fork();
  if(process_id < 0){
    perror("Fork");
//    return pinfo.version;

  }else if(process_id == 0) {
    dup2(read_pipe[1], 1);
    close(read_pipe[0]);
    close(read_pipe[1]);
    execlp("rpm", "rpm", "-qi", package.c_str(), NULL);
  } else {
    // parent:
    int status;
    waitpid(process_id, &status,0);
    close(read_pipe[1]);
    {
      char arr[4096];
      int n = read(read_pipe[0], arr, sizeof(arr));
      ss.write(arr, n);

    }
    //std::cout << ss.str() << std::endl;
    close(read_pipe[0]);
  }
}

#endif

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
