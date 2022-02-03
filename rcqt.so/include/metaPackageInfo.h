#ifndef PACKAGE_INFO_H
#define PACKAGE_INFO_H
#include <cstdio>
#include "include/rcutils.h"
class PackageInfo{

public:
	PackageInfo(const std::string& pkgname, const std::string& pkgmgr, const std::string& cmd):m_pkgname{pkgname}, m_pkgmgrname{pkgmgr}, m_cmdname{cmd}{
    if(!pkgname.empty())
			m_filename = pfilename(pkgname);	
	}
	virtual ~PackageInfo(){
		if(!m_filename.empty()){
			auto res = std::remove(m_filename.c_str());
		}
	}
	PackageInfo(const PackageInfo& ) = default;
  bool fillPkgInfo(); // reads from package and writes to a file
  virtual bool readMetaPackageInfo(std::string ss) = 0;
	std::string getFileName(){
		return m_filename;
	}
  void setPkg(const std::string& pkgname){
		if(!m_filename.empty()){
      auto res = std::remove(m_filename.c_str());
		}
		  m_filename = pfilename(pkgname);
	}
	std::string getPackageName(){
		return m_pkgname;
	}
private:
	std::string m_pkgname;
	std::string m_filename;
  std::string m_pkgmgrname; // dpkg or yum or zypper
  std::string m_cmdname; // option for package manager
};

#endif
