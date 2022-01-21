#ifndef DEB_PACKAGE_INFO_H
#define DEB_PACKAGE_INFO_H
#include "metaPackageInfo.h"

class DebPackageInfo : public PackageInfo{
public:
	DebPackageInfo(const std::string& pkgname,
                 const std::string& pkgmgr,
								 const std::string& cmd):PackageInfo(pkgname,
                                      pkgmgr,
                                      cmd){}
	bool readMetaPackageInfo(std::string ss) override;
	std::pair<std::string, std::string> getNameVers(std::string word);
	void readDepLine(const std::string& depLine);
};
#endif
