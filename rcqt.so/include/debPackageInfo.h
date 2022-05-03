#ifndef DEB_PACKAGE_INFO_H
#define DEB_PACKAGE_INFO_H
#include "metaPackageInfo.h"

class DebPackageInfo : public PackageInfo{
  public:
    DebPackageInfo(const std::string& pkgname,
        const std::vector <std::string> cmd):PackageInfo(pkgname,
          std::string("dpkg"),
          cmd){}

    DebPackageInfo(const std::vector <std::string> cmd):PackageInfo(
          std::string("dpkg"),
          cmd){}

    bool readMetaPackageInfo(std::string ss) override;
    std::pair<std::string, std::string> getNameVers(std::string word);
    void readDepLine(const std::string& depLine);
};
#endif
