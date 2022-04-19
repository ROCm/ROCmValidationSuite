#ifndef ZYP_PACKAGE_INFO_H
#define ZYP_PACKAGE_INFO_H

#include "metaPackageInfo.h"

class ZypPackageInfo : public PackageInfo {

  public:

    ZypPackageInfo(const std::string& pkgname,
        const std::string& pkgmgr,
        const std::vector <std::string> cmd):PackageInfo(pkgname,
          pkgmgr,
          cmd[0],
          cmd[1],
          cmd[2]){}

    ZypPackageInfo(const std::string& pkgmgr,
        const std::vector <std::string> cmd):PackageInfo(pkgmgr,
          cmd[0],
          cmd[1],
          cmd[2]){}

    bool readMetaPackageInfo(std::string ss) override;
};
#endif
