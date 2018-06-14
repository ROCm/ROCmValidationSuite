
#ifndef RVSLIB_H_
#define RVSLIB_H_

#include <map>
#include <string>


namespace rvs
{

struct init_stuct
{
	
};

namespace lib
{

class actionbase
{
public:
	virtual ~actionbase();
	
protected:
	actionbase();
	void sleep(const unsigned int ms);
	
public:
  virtual int   	property_set(const char*, const char*);
  virtual int   	run(void) = 0;
  bool has_property(const std::string& key, std::string& val);
  bool has_property(const std::string& key);

protected:
	std::map<std::string, std::string>	property;
};


	
}	// namespace lib
}	// namespace rvs


#endif // RVSLIB_H_