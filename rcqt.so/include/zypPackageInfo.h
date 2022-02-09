#ifndef ZYP_PACKAGE_INFO_H
#define ZYP_PACKAGE_INFO_H

#include "metaPackageInfo.h"

class ZypPackageInfo : public PackageInfo {
public:
	ZypPackageInfo(const std::string& pkgname,
                 const std::string& pkgmgr,
								 const std::string& cmd1,
								 const std::string& cmd2):PackageInfo(pkgname,
                                      pkgmgr,
                                      cmd1,
                                      cmd2){}
	bool readMetaPackageInfo(std::string ss) override;
};
#endif
