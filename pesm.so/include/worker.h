
#ifndef _WORKER_H_
#define _WORKER_H_

#include <string>
#include "rvsthreadbase.h"

	
class Worker : public rvs::ThreadBase {

public:
	Worker();
	~Worker();
	
	void stop(void);
	void set_name(const std::string& name) { action_name = name; }
	void json(const bool flag) { bjson = flag; }
	const std::string& get_name(void) { return action_name; }
	
protected:
	virtual void run(void);
	
protected:
	bool		bjson;
	bool 		brun;
	std::string	action_name;
};
	


#endif // _WORKER_H_