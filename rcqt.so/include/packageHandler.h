#ifndef PACKAGE_HANDLER_H
#define PACKAGE_HANDLER_H

#include <string>
#include <map>
#include <memory>
#include "include/rcutils.h"
#include "metaPackageInfo.h"

class PackageHandler{
public:
	PackageHandler() = default;
	void setManifest(std::string manifest){
		m_manifest = manifest;
	}
	std::string getManifest() const{
		return m_manifest;
	}
	virtual bool parseManifest(); // and load m_pkgversionmap
	void  validatePackages();
	const std::map<std::string, std::string>& getPackageMap() const{
		return m_pkgversionmap;
	}
	virtual std::string getInstalledVersion(const std::string& package) = 0;
	virtual bool pkgrOutputParser(const std::string& s_data, package_info& info) = 0;
  void setPkg(const std::string& pkg) {
		/*
		if(metaInfo)
			metaInfo->setPkg(pkg);
      metaInfo->fillPkgInfo();
		  m_manifest = metaInfo->getFileName();
  */
	}	
	virtual ~PackageHandler(){}
	
private:
	std::map<std::string, std::string> m_pkgversionmap;
protected:
	std::unique_ptr<PackageInfo> metaInfo;
	std::string m_manifest;	
};
#endif
