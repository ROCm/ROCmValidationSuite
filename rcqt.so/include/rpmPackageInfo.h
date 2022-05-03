#ifndef RPM_PACKAGE_INFO_H
#define RPM_PACKAGE_INFO_H

#include "metaPackageInfo.h"

class RpmPackageInfo : public PackageInfo {

  public:
    RpmPackageInfo(const std::string& pkgname,
        const std::vector <std::string> cmd):PackageInfo(pkgname,
          std::string("rpm"),
          cmd){}

    RpmPackageInfo(const std::vector <std::string> cmd):PackageInfo(
          std::string("rpm"),
          cmd){}

    bool readMetaPackageInfo(std::string ss) override;
};
#endif
