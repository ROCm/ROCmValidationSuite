
#ifndef _LOGNODE_H
#define _LOGNODE_H

#include "rvslognodebase.h"

#include <vector>
#include <memory>

namespace rvs 
{
	
class LogNode : public LogNodeBase
{
	
public:

	LogNode( const std::string& Name, const LogNodeBase* Parent = nullptr);
	LogNode( const char* Name, const LogNodeBase* Parent = nullptr);
	virtual ~LogNode();

	virtual std::string ToJson(const std::string& Lead = "");
	
public:
	void Add(LogNodeBase* spChild);
	
public:
	std::vector<LogNodeBase*> Child;
	
	
};

}	// namespace rvs 

#endif // _LOGNODE_H