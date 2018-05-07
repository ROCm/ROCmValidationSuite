/*
 * rvsmodule.cpp
 *
 *  Created on: May 4, 2018
 *      Author: ubuntu
 */

#include <stdio.h>
#include <iostream>
#include <dlfcn.h>

#include "rvsif0.h"
#include "rvsif1.h"
#include "rvsmodule.h"


rvsmodule::rvsmodule()
: psolib(NULL),
  pif0(NULL),
  pif1(NULL)
{
}

rvsmodule::~rvsmodule()
{
}


rvsmodule* rvsmodule::create(const char* name)
{
	// create instance
	rvsmodule* p = new rvsmodule();

	// try loading .so module
	p->psolib = dlopen(name, RTLD_LAZY);

	// error?
	if (!p->psolib)
	{
	   // clean up and return NULL
	   delete p;
	   return NULL;
	}


	// initialize interfaces:
	if(p->init_interfaces())
	{
		delete p;
		return NULL;
	}


	return p;
}

int rvsmodule::init_interfaces(void)
{
	int sts;
	pif0 = new rvsif0();
	sts = pif0->init_rvs_interface(psolib);
	if(sts)
	{
	    delete pif0;
	    // rvs modul must implmnt intrface 0, so report an error
	    return 1;
	}

	pif1 = new rvsif1();
	sts = pif1->init_rvs_interface(psolib);
	if(sts)
	{
		// rvs interface not found:
	    delete pif1;
	    pif1 = NULL;
	}

    return 0;
}

int rvsmodule::destroy(rvsmodule* pmodule)
{
	if (!pmodule)
		return 1;

	dlclose(pmodule->psolib);

	delete pmodule;

	return 0;
}

void* rvsmodule::get_interface(int iid)
{
	switch(iid)
	{
	case 0:
	    return pif0;
	case 1:
	    return pif1;
	default:
		return NULL;
	}
}



