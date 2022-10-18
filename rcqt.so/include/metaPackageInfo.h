/********************************************************************************
 *
 * Copyright (c) 2018-2022 ROCm Developer Tools
 *
 * MIT LICENSE:
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to do
 * so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#ifndef PACKAGE_INFO_H
#define PACKAGE_INFO_H

#include <cstdio>
#include "include/rcutils.h"

class PackageInfo {

public:
  PackageInfo(const std::string& pkgname, const std::string& pkgmgr, const std::vector <std::string> cmd) :
    m_pkgname{pkgname}, m_pkgmgrname{pkgmgr}, m_dependcmd1name{cmd[0]}, m_dependcmd2name{cmd[1]}, m_infocmdname{cmd[2]} {
      if(!pkgname.empty())
        m_filename = pfilename(pkgname);	
    }

  PackageInfo(const std::string& pkgmgr, const std::vector <std::string> cmd) :
    m_pkgmgrname{pkgmgr}, m_dependcmd1name{cmd[0]}, m_dependcmd2name{cmd[1]}, m_infocmdname{cmd[2]} {
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
