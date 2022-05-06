#ifndef ZYP_PACKAGE_INFO_H
#define ZYP_PACKAGE_INFO_H

#include "metaPackageInfo.h"

class ZypPackageInfo : public PackageInfo {

  public:

    ZypPackageInfo(const std::string& pkgname,
        const std::vector <std::string> cmd):PackageInfo(pkgname,
          std::string("zypper"),
          cmd){}

    ZypPackageInfo(const std::vector <std::string> cmd):PackageInfo(
          std::string("zypper"),
          cmd){}

    bool readMetaPackageInfo(std::string ss) override;
};
#endif
