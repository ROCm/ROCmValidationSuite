#ifndef DEB_PACKAGE_INFO_H
#define DEB_PACKAGE_INFO_H
#include "metaPackageInfo.h"

class DebPackageInfo : public PackageInfo{
public:
	DebPackageInfo(const std::string& pkgname,
                 const std::string& pkgmgr,
								 const std::string& cmd1,
								 const std::string& cmd2,
								 const std::string& cmd3):PackageInfo(pkgname,
                                      pkgmgr,
                                      cmd1,
                                      cmd2,
                                      cmd3){}
	bool readMetaPackageInfo(std::string ss) override;
	std::pair<std::string, std::string> getNameVers(std::string word);
	void readDepLine(const std::string& depLine);
};
#endif
