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
    auto hyp = pvers.find("-");
    if (hyp != std::string::npos){
      pvers = pvers.substr(0,hyp);
    }
    return {remSpaces(pname), remSpaces(pvers)};
  }
}

void DebPackageInfo::readDepLine(const std::string& depLine){
  std::stringstream ss{std::string{depLine}};
  std::ofstream os;
	os.open(getFileName() , std::ofstream::out | std::ofstream::app);
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
