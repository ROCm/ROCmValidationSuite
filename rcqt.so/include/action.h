

#ifndef ACTION_H_
#define ACTION_H_

#include "rvslib.h"

class action : public rvs::lib::actionbase
{
public:
	action();
	virtual ~action();
	
	virtual int property_set(const char*, const char*);
	virtual int run(void);
	
protected:
	
};

#endif /* ACTION_H_ */
