

#include "rvsif0.h"

rvs::if0::if0()
:rvs_module_get_version(nullptr),
 rvs_module_get_name(nullptr),
 rvs_module_get_description(nullptr),
 rvs_module_has_interface(nullptr)
{
}

rvs::if0::~if0() 
{
}

rvs::if0::if0(const if0& rhs)
{
	*this = rhs;
}

rvs::if0& rvs::if0::operator=(const rvs::if0& rhs) // copy assignment
{
	// self-assignment check
    if (this != &rhs) 
	{
		ifbase::operator=(rhs);
		rvs_module_get_version 		= rhs.rvs_module_get_version;
		rvs_module_get_name			= rhs.rvs_module_get_name;
		rvs_module_get_description	= rhs.rvs_module_get_description;
		rvs_module_has_interface	= rhs.rvs_module_has_interface;
		rvs_module_get_config		= rhs.rvs_module_get_config;
		rvs_module_get_output		= rhs.rvs_module_get_output;
    }
    
    return *this;
}

rvs::ifbase* rvs::if0::clone(void)
{
	return new rvs::if0(*this);
}


void  rvs::if0::get_version(int* Major, int* Minor, int* Patch)
{
	(*rvs_module_get_version)(Major, Minor, Patch);
}

char* rvs::if0::get_name(void)
{
	return (*rvs_module_get_name)();
}

char* rvs::if0::get_description(void)
{
	return (*rvs_module_get_description)();
}

int rvs::if0::has_interface(int iid)
{
	return (*rvs_module_has_interface)(iid);
}

char* rvs::if0::get_config()
{
	return (*rvs_module_get_config)();
}
char* rvs::if0::get_output()
{
	return (*rvs_module_get_output)();
}



