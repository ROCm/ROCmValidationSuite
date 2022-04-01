#ifndef RPM_PACKAGE_INFO_H
#define RPM_PACKAGE_INFO_H

#include "metaPackageInfo.h"

class RpmPackageInfo : public PackageInfo {
  public:
    RpmPackageInfo(const std::string& pkgname,
        const std::string& pkgmgr,
        const std::vector <std::string> cmd):PackageInfo(pkgname,
          pkgmgr,
          cmd[0],
          cmd[1],
          cmd[2]){}
    bool readMetaPackageInfo(std::string ss) override;
};
#endif
