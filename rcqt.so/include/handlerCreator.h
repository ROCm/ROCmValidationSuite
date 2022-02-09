#ifndef HANDLER_CREATOR_H
#define HANDLER_CREATOR_H

#include <memory>
#include "include/packageHandlerDeb.h"
#include "include/packageHandlerRpm.h"
#include "include/packageHandlerZyp.h"
#include "include/rcutils.h"

class handlerCreator{
public:
	handlerCreator() = default;
	virtual ~handlerCreator() = default;
	PackageHandler* getPackageHandler(const std::string& pkg);
};
#endif
