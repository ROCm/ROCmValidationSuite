#include "include/handlerCreator.h"
#include <iostream>
std::unique_ptr<PackageHandler> handlerCreator::getPackageHandler(const std::string& pkg){
	auto osName = getOS();
	std::cout << "nameees " << osName << std::endl;
	if(osName.find("ubuntu") != std::string::npos)
		return std::unique_ptr<PackageHandler>(new PackageHandlerDeb{pkg});
	else
		return nullptr;
}
