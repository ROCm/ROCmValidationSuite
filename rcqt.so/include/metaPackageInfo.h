#ifndef PACKAGE_INFO_H
#define PACKAGE_INFO_H

#include <cstdio>
#include "include/rcutils.h"

class PackageInfo {

public:
  PackageInfo(const std::string& pkgname, const std::string& pkgmgr,
      const std::string& cmd1, const std::string& cmd2, const std::string& cmd3) :
    m_pkgname{pkgname}, m_pkgmgrname{pkgmgr}, m_dependcmd1name{cmd1}, m_dependcmd2name{cmd2}, m_infocmdname{cmd3} {
      if(!pkgname.empty())
        m_filename = pfilename(pkgname);	
    }

  PackageInfo(const std::string& pkgmgr,
      const std::string& cmd1, const std::string& cmd2, const std::string& cmd3) :
    m_pkgmgrname{pkgmgr}, m_dependcmd1name{cmd1}, m_dependcmd2name{cmd2}, m_infocmdname{cmd3} {
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
	
  std::string getPackageMgrName(){
		return m_pkgmgrname;
	}

  std::string getInfoCmdName(){
		return m_infocmdname;
	}

private:
	std::string m_pkgname;
	std::string m_filename;
  std::string m_pkgmgrname; // dpkg or yum or zypper
  std::string m_dependcmd1name; // command option1 to find dependent packages.
  std::string m_dependcmd2name; // command option2 to find dependent packages.
  std::string m_infocmdname; // command option to find package info.
};

#endif
