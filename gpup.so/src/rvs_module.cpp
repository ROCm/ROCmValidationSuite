/*
 * rvs_module.cpp
 *
 *  Created on: Apr 27, 2018
 *      Author: ubuntu
 */




#include <stdio.h>

extern "C" void  rvs_module_get_version(int* Major, int* Minor, int* Revision)
{
	puts("\nHello from shared library - rvs_module_get_version()\n");
	*Major = 1;
	*Minor = 0;
	*Revision = 0;
}

extern "C" int rvs_module_has_interface(int iid)
{
	switch(iid)
	{
	case 0:
		return 1;
    }

	return 0;
}

extern "C" char* rvs_module_get_name(void)
{
   return (char*)"gpup";
}

extern "C" char* rvs_module_get_description(void)
{
   return (char*)"ROCm Validation Suite GPUP module";
}
