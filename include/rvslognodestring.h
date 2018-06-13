
#ifndef _LOGNODESTRING_H
#define _LOGNODESTRING_H

#include "rvslognodebase.h"

namespace rvs 
{
	

class LogNodeString : public LogNodeBase
{
	
public:

	LogNodeString( const std::string& Name, const std::string& Val, const LogNodeBase* Parent = nullptr);
	LogNodeString( const char* Name, const char* Val, const LogNodeBase* Parent = nullptr);
	
	virtual ~LogNodeString();
	
	virtual std::string ToJson(const std::string& Lead = "");
	
protected:	
	
	std::string Value;
};

}	// namespace rvs 

#endif // _LOGNODESTRING_H