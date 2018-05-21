
#ifndef RVSLIB_H_
#define RVSLIB_H_

#include <map>
#include <string>

namespace rvs
{
namespace lib
{

class actionbase
{
public:
	virtual ~actionbase();
	
protected:
	actionbase();
	
public:
    virtual int   	property_set(const char*, const char*);
	virtual int   	run(void) = 0;

protected:
	std::map<std::string, std::string>	property;
};


	
}	// namespace lib
}	// namespace rvs


#endif // RVSLIB_H_