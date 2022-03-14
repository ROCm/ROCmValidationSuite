#ifndef RPM_PACKAGE_INFO_H
#define RPM_PACKAGE_INFO_H

#include "metaPackageInfo.h"

class RpmPackageInfo : public PackageInfo {
public:
	RpmPackageInfo(const std::string& pkgname,
                 const std::string& pkgmgr,
								 const std::string& cmd1,
								 const std::string& cmd2,
								 const std::string& cmd3):PackageInfo(pkgname,
                                      pkgmgr,
                                      cmd1,
                                      cmd2,
                                      cmd3){}
	bool readMetaPackageInfo(std::string ss) override;
};
#endif
