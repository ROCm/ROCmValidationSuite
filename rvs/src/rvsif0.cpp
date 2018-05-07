/*
 * rvsif0.cpp
 *
 *  Created on: May 6, 2018
 *      Author: ubuntu
 */

#include <stdio.h>
#include <dlfcn.h>
#include "rvsif0.h"

rvsif0::rvsif0()
:rvs_module_get_version(NULL),
 rvs_module_get_name(NULL),
 rvs_module_get_description(NULL),
 rvs_module_has_interface(NULL)
{}

rvsif0::~rvsif0() {}

int rvsif0::init_rvs_interface(void* psolib)
{
	char* error;

	if (!psolib)
	{
	   return 2;
	}

	rvs_module_get_version = (t_voidintpintpintp)(dlsym(psolib, "rvs_module_get_version"));
	if ((error = dlerror()) != NULL)
	{
	   return 1;
	}

	rvs_module_get_name = (t_charpvoid)(dlsym(psolib, "rvs_module_get_name"));
	if ((error = dlerror()) != NULL)
	{
	   return 1;
	}

	rvs_module_get_description = (t_charpvoid)(dlsym(psolib, "rvs_module_get_description"));
	if ((error = dlerror()) != NULL)
	{
	   return 1;
	}

	rvs_module_has_interface = (t_intint)(dlsym(psolib, "rvs_module_has_interface"));
	if ((error = dlerror()) != NULL)
	{
	   return 1;
	}

	return 0;
}

void  rvsif0::get_version(int* Major, int* Minor, int* Patch)
{
	(*rvs_module_get_version)(Major, Minor, Patch);
}

char* rvsif0::get_name(void)
{
	return (*rvs_module_get_name)();
}

char* rvsif0::get_description(void)
{
	return (*rvs_module_get_description)();
}

int rvsif0::has_interface(int iid)
{
	return (*rvs_module_has_interface)(iid);
}




