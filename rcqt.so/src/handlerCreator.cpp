#include "include/handlerCreator.h"
#include <iostream>
PackageHandler* handlerCreator::getPackageHandler(const std::string& pkg){
	auto osName = getOS();
	PackageHandler* lptr = nullptr;
	if(OSType::Ubuntu == osName){
		lptr = new PackageHandlerDeb{pkg};
	}
	
	return lptr;
}
