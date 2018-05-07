/*
 * rvsif1.cpp
 *
 *  Created on: May 6, 2018
 *      Author: ubuntu
 */

#include <stdio.h>
#include <dlfcn.h>
#include "rvsif1.h"

rvsif1::rvsif1()
: rvs_module_init(NULL),
  rvs_module_run(NULL),
  rvs_module_get_errstring(NULL),
  rvs_module_get_errint(NULL),
  rvs_module_terminate(NULL)
{
}

rvsif1::~rvsif1()
{
}


int rvsif1::init_rvs_interface(void* psolib)
{
	char* error;

	if (!psolib)
	{
	   return 2;
	}

	rvs_module_init = (t_intvoid)(dlsym(psolib, "rvs_module_init"));
	if ((error = dlerror()) != NULL)
	{
	   return 1;
	}

	rvs_module_run = (t_intintcharpp)(dlsym(psolib, "rvs_module_run"));
	if ((error = dlerror()) != NULL)
	{
	   return 1;
	}

	rvs_module_get_errstring = (t_charpvoid)(dlsym(psolib, "rvs_module_get_errstring"));
	if ((error = dlerror()) != NULL)
	{
	   return 1;
	}

	rvs_module_get_errint = (t_intvoid)(dlsym(psolib, "rvs_module_get_errint"));
	if ((error = dlerror()) != NULL)
	{
	   return 1;
	}

	rvs_module_terminate = (t_intvoid)(dlsym(psolib, "rvs_module_terminate"));
	if ((error = dlerror()) != NULL)
	{
	   return 1;
	}

	return 0;
}


int rvsif1::init(void)
{
	return (*rvs_module_init)();
}

int rvsif1::run(int Argc, char* Argv[])
{
	return (*rvs_module_run)(Argc, Argv);
}

char* rvsif1::get_errstring(void)
{
	return (*rvs_module_get_errstring)();
}

int rvsif1::get_errint(void)
{
	return (*rvs_module_get_errint)();
}

int rvsif1::terminate(void)
{
	return (*rvs_module_terminate)();
}

