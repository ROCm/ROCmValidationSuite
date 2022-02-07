
#ifndef PACKAGE_HANDLER_RPM_H
#define PACKAGE_HANDLER_RPM_H
#include "packageHandler.h"

class PackageHandlerRpm: virtual public PackageHandler{
public:
	PackageHandlerRpm(std::string pkgname = "");
	void validatePackages() override;
	bool pkgrOutputParser(const std::string& s_data, 
																	package_info& info)	override;
	std::string getInstalledVersion(const std::string& package) override;	
	~PackageHandlerRpm(){}
};
#endif
