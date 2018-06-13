
#ifndef _LOGNODEREC_H
#define _LOGNODEREC_H

#include "rvslognode.h"

#include <vector>


namespace rvs 
{
	
class LogNodeRec : public LogNode
{
	
public:

	LogNodeRec( const std::string& Name, const int LogLevel, const unsigned int Sec, const unsigned int uSec, const LogNodeBase* Parent = nullptr);
	LogNodeRec( const char* Name, const int LogLevel, const unsigned int Sec, const unsigned int uSec, const LogNodeBase* Parent = nullptr);
	virtual ~LogNodeRec();

	virtual std::string ToJson(const std::string& Lead = "");
	
public:
	const int LogLevel();
	
protected:
	int Level;
	const int sec;
	const int usec;
};

}	// namespace rvs 

#endif // _LOGNODEREC_H