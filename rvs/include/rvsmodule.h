/*
 * rvsmodule.h
 *
 *  Created on: Apr 27, 2018
 *      Author: ubuntu
 */

#ifndef RVSMODULE_H_
#define RVSMODULE_H_

class rvsif0;
class rvsif1;

class rvsmodule
{
public:
	static rvsmodule* create(const char* modulename);
	static int        destroy(rvsmodule* pmodule);
	virtual void*     get_interface(int);

protected:
	rvsmodule();
	virtual ~rvsmodule();
	int init_interfaces(void);

protected:
	void* psolib;
	rvsif0* pif0;
	rvsif1* pif1;


};





#endif /* RVSMODULE_H_ */
