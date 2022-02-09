#ifndef PACKAGE_HANDLER_ZYP_H
#define PACKAGE_HANDLER_ZYP_H

#include "packageHandler.h"

class PackageHandlerZyp: virtual public PackageHandler{
public:
	PackageHandlerZyp(std::string pkgname = "");
	void validatePackages() override;
	bool pkgrOutputParser(const std::string& s_data, 
																	package_info& info)	override;
	std::string getInstalledVersion(const std::string& package) override;	
	~PackageHandlerZyp(){}
};
#endif
