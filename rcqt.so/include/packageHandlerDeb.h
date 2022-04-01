#ifndef PACKAGE_HANDLER_DEB_H
#define PACKAGE_HANDLER_DEB_H
#include "include/packageHandler.h"

class PackageHandlerDeb: virtual public PackageHandler{
public:
	PackageHandlerDeb(std::string pkgname = ""); 
	bool pkgrOutputParser(const std::string& s_data, 
																	package_info& info)	override;
	std::string getInstalledVersion(const std::string& package) override;	
	~PackageHandlerDeb(){}
};
#endif
