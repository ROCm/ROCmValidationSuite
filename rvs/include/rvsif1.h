/*
 * rvsif1.h
 *
 *  Created on: May 6, 2018
 *      Author: ubuntu
 */

#ifndef RVSIF1_H_
#define RVSIF1_H_

#include "rvsmodule_if.h"

class rvsif1
{
public:
	virtual int   init(void);
	virtual int   run(int, char**);
	virtual char* get_errstring(void);
	virtual int   get_errint(void);
	virtual int   terminate(void);

private:
	rvsif1();
	virtual ~rvsif1();
	virtual int init_rvs_interface(void* psolib);

private:
	t_intvoid         rvs_module_init;
	t_intintcharpp    rvs_module_run;
	t_charpvoid       rvs_module_get_errstring;
	t_intvoid         rvs_module_get_errint;
	t_intvoid         rvs_module_terminate;

friend class rvsmodule;
};


#endif /* RVSIF1_H_ */
