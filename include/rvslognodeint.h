
#ifndef _LOGNODEINT_H
#define _LOGNODEINT_H

#include "rvslognodebase.h"

namespace rvs 
{
	

class LogNodeInt : public LogNodeBase
{
	
public:

	LogNodeInt( const std::string& Name, const int Val, const LogNodeBase* pParent = nullptr);
	LogNodeInt( const char* Name, const int Val, const LogNodeBase* pParent = nullptr);
	
	virtual ~LogNodeInt();

	virtual std::string ToJson(const std::string& Lead = "");
	
protected:
	
	int Value;
};

}	// namespace rvs 

#endif // _LOGNODEINT_H