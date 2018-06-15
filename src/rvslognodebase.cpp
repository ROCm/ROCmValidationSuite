
#include "rvslognodebase.h"

rvs::LogNodeBase::LogNodeBase(const std::string& rName, const LogNodeBase* pParent) 
: Name(rName),
Parent(pParent),
Type(eLN::Unknown)
{
}

rvs::LogNodeBase::LogNodeBase(const char* rName, const LogNodeBase* pParent)
: Name(rName),
Parent(pParent),
Type(eLN::Unknown)
{
}

rvs::LogNodeBase::~LogNodeBase() {
}
