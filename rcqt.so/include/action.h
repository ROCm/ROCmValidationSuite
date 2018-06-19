

#ifndef ACTION_H_
#define ACTION_H_

#include "rvslib.h"
#include <string>
#include <vector>

class action : public rvs::lib::actionbase
{
public:
	action();
	virtual ~action();
	
	virtual int property_set(const char*, const char*);
	virtual int run(void);
  virtual void split_string(std::vector <std::string> &group_array, char delimiter, std::string string_of_groups);
	
protected:
	
};

#endif /* ACTION_H_ */
