#ifndef _RVSLOGNODEBASE_H_
#define _RVSLOGNODEBASE_H_

#include <string>

#define RVSENDL "\n"
#define RVSINDENT "  "

namespace rvs 
{
	
typedef enum eLN {
	Unknown = 0,
	List    = 1,
	String  = 2,
	Integer = 3,
	Record  = 4
} T_LNTYPE;
	
class LogNodeBase
{
public:
	virtual ~LogNodeBase();
	virtual std::string ToJson(const std::string& Lead = "") = 0;
	
protected:	
	LogNodeBase(const std::string& rName, const LogNodeBase* pParent = nullptr);
	LogNodeBase(const char* rName, const LogNodeBase* pParent = nullptr);
	
protected:
	const LogNodeBase* 	Parent;
	std::string 		Name;
	T_LNTYPE 			Type;
};




}	// namespace rvs

#endif // _RVSLOGNODEBASE_H_