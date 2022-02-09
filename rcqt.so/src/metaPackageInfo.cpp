#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

#include "include/metaPackageInfo.h"

bool PackageInfo::fillPkgInfo(){
  if(m_filename.empty())
    return false;
	//std::stringstream ss;
  std::ofstream os{m_filename};
  int read_pipe[2]; // From child to parent
  int exit_status;
        //package_info pinfo;
  if(pipe(read_pipe) == -1){
    perror("Pipe");
    return false;
  }
  pid_t process_id = fork();
  if(process_id < 0){
    perror("Fork");
    return false;

  }else if(process_id == 0) {
    dup2(read_pipe[1], 1);
    close(read_pipe[0]);
    close(read_pipe[1]);
		//std::cout<< "in child" << std::endl;
    execlp(m_pkgmgrname.c_str(), m_pkgmgrname.c_str(), m_cmd1name.c_str(),  m_cmd2name.c_str(), m_pkgname.c_str(), NULL);
  } else {
    // parent:
    int status;
    waitpid(process_id, &status,0);
    std::stringstream ss;
    close(read_pipe[1]);
    { 
      char arr[8192];
      int n = read(read_pipe[0], arr, sizeof(arr));
      ss.write(arr, n);
    }
    
    close(read_pipe[0]);
    // handle ss
    readMetaPackageInfo(ss.str());
		//std::cout << ss.str() << std::endl;
		return true;
    }
}

