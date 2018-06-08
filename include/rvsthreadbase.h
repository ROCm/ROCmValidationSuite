
#ifndef _RVSTHREAD_BASE_H_
#define _RVSTHREAD_BASE_H_

#include <thread>

namespace rvs
{

class ThreadBase {
protected:
	ThreadBase();
	~ThreadBase();
	
public:
	void start();
	void detach();
	void join();
	void sleep(const unsigned int ms);
	
protected:
	void runinternal(void);
	virtual void run() = 0;
	
protected:
	std::thread t;
	
};

}


#endif // _RVSTHREAD_BASE_H_