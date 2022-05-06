#include "include/handlerCreator.h"
#include <iostream>

PackageHandler* handlerCreator::getPackageHandler(const std::string& pkg){

  auto osName = getOS();
  PackageHandler* lptr = nullptr;

  if(OSType::Ubuntu == osName){
    lptr = new PackageHandlerDeb{pkg};
  }
  else if (OSType::Centos == osName) {
    lptr = new PackageHandlerRpm{pkg};
  }
  else if (OSType::SLES == osName) {
    lptr = new PackageHandlerZyp{pkg};
  }
  return lptr;
}

PackageHandler* handlerCreator::getPackageHandler(){

  auto osName = getOS();
  PackageHandler* lptr = nullptr;

  if(OSType::Ubuntu == osName){
    lptr = new PackageHandlerDeb{};
  }
  else if (OSType::Centos == osName) {
    lptr = new PackageHandlerRpm{};
  }
  else if (OSType::SLES == osName) {
    lptr = new PackageHandlerZyp{};
  }
  return lptr;
}
