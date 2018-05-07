/*
 * rvsif0.h
 *
 *  Created on: May 6, 2018
 *      Author: ubuntu
 */

#ifndef RVSIF0_H_
#define RVSIF0_H_


#include "rvsmodule_if.h"

class rvsif0
{
public:
	virtual void    get_version(int*, int*, int*);
	virtual char*   get_name(void);
	virtual char*   get_description(void);
	virtual int     has_interface(int);

private:
	rvsif0();
	virtual ~rvsif0();
	virtual int init_rvs_interface(void* psolib);

private:
	t_voidintpintpintp rvs_module_get_version;
	t_charpvoid        rvs_module_get_name;
	t_charpvoid        rvs_module_get_description;
	t_intint           rvs_module_has_interface;

friend class rvsmodule;

};


#endif /* RVSIF0_H_ */
