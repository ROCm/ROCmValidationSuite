

#ifndef ACTION_H_
#define ACTION_H_

#include "rvslib.h"

class action : public rvs::lib::actionbase
{
public:
	action();
	virtual ~action();
	
	virtual int run(void);
	
protected:
	int do_gpu_list(void);
};

#endif /* ACTION_H_ */
