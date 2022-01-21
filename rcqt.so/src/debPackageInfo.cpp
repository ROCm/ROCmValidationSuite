#include <iostream>
#include <sstream>
#include <fstream>
#include "include/debPackageInfo.h"

// decide on output of package manager
std::pair<std::string, std::string>
DebPackageInfo::getNameVers(std::string word){
  auto it = word.find("(=");
  if(it != std::string::npos){
    auto pname = word.substr(0, it);
    auto pvers = word.substr(it+2);
    //remove last ) as it wont help
    pvers = pvers.substr(0, pvers.size()-1);
    return {remSpaces(pname), remSpaces(pvers)};
  }
}

void DebPackageInfo::readDepLine(const std::string& depLine){
  std::stringstream ss{std::string{depLine}};
  std::ofstream os;
	os.open(getFileName() , std::ofstream::out | std::ofstream::app);
	std::cout << "DEBUG: file is " << getFileName() << std::endl;
  std::string word;
  while(std::getline(ss, word, ',')){
    std::pair<std::string, std::string> wp = getNameVers(word);
    os << wp.first << " " <<wp.second << std::endl;
  }

}

bool DebPackageInfo::readMetaPackageInfo(std::string ss){
  std::stringstream inss{ss};
	//std::ofstream os{m_filename};
	std::string line;
	bool found = false;
	while(std::getline(inss, line)){
		if(!found){
			if(line.find("Depends:") != std::string::npos){
				found = true;
				auto itr = line.find(":");
        line= line.substr(itr+1);
				readDepLine(line);
				return true;
			} else{
				continue;
			}
		}
	}
	return found;
}
/*
bool DebPackageInfo::fillPkgInfo(){
	if(m_filename.empty())
		return false;
  std::ofstream	os{m_filename};
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
    execlp("dpkg", "dpkg", "--status", package.c_str(), NULL);
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
	  readMetaPackageInfo(ss);	
		}
}
*/
