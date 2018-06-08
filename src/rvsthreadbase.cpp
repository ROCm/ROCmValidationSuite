
#include "rvsthreadbase.h"

#include <chrono>
//#include <iostream>


rvs::ThreadBase::ThreadBase() : t() {
}
	
rvs::ThreadBase::~ThreadBase() {
}

void rvs::ThreadBase::runinternal() {
//	std::cout << "In rvs::ThreadBase::runinternal()" << std::endl;
	run();
}

void rvs::ThreadBase::start() {
//	std::cout << "In rvs::ThreadBase::start()" << std::endl;
	t = std::thread(&rvs::ThreadBase::runinternal, this);
}

void rvs::ThreadBase::detach() {
//	std::cout << "In rvs::ThreadBase::detach()" << std::endl;
	t.detach();
}

void rvs::ThreadBase::join() {
//	std::cout << "In rvs::ThreadBase::join()" << std::endl;
	t.join();
	
}

void rvs::ThreadBase::sleep(const unsigned int ms) {
	std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}
