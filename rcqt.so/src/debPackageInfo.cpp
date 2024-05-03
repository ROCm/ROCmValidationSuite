#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include "include/debPackageInfo.h"

std::vector<std::string> dependency_patterns{"(=", "(>="};

std::string findPattern(std::string word){
  for (auto pt : dependency_patterns){
    if (word.find(pt) != std::string::npos) {
	return pt;
    }
  }
  return std::string{};
}


std::pair<std::string, std::string>
DebPackageInfo::getNameVers(std::string word){
  std::string pname{word};
  std::string pvers{"0+"};
  auto ordep = word.find("|");
  if ( ordep != std::string::npos){
    return {ORDEP+remSpaces(pname.substr(0, ordep))+","+remSpaces(pname.substr(ordep+1)), "0+"};
  }
  auto pat  = findPattern(word);
  if (pat.size() != 0 ){
    auto it = word.find(pat);
      if(it != std::string::npos){
        pname = word.substr(0, it);
        pvers = word.substr(it+pat.size());
        //remove last ) as it wont help
        pvers = pvers.substr(0, pvers.size()-1);
        auto hyp = pvers.find("-");
        if (hyp != std::string::npos){
          pvers = pvers.substr(0,hyp);
        }
	if (pat.compare("(>=") == 0)
          pvers+="+";
    }
  } 
  return {remSpaces(pname), remSpaces(pvers)};
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
