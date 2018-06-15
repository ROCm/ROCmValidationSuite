

#include "rvs_module.h"
#include "action.h"
#include "rvsloglp.h"

int log(const char* pMsg, const int level)
{
	return rvs::lp::Log(pMsg, level);
}

extern "C" void  rvs_module_get_version(int* Major, int* Minor, int* Revision)
{
	*Major = BUILD_VERSION_MAJOR;
	*Minor = BUILD_VERSION_MINOR;
	*Revision = BUILD_VERSION_PATCH;
}

extern "C" int rvs_module_has_interface(int iid)
{
	switch(iid)
	{
	case 0:
    case 1:
		return 1;
    }

	return 0;
}

extern "C" char* rvs_module_get_name(void)
{
   return (char*)"smqt";
}

extern "C" char* rvs_module_get_description(void)
{
   return (char*)"SBIOS Mapping Qualification Tool";
}

extern "C" char* rvs_module_get_config(void)
{
	return (char*)"bar1_req_size(integer), bar1_base_addr_min(integer), bar1_base_addr_max(integer), bar2_req_size(integer), bar2_base_addr_min(integer), bar2_base_addr_max(integer), bar4_req_size(integer), bar4_base_addr_min(integer), bar4_base_addr_max(integer), bar5_req_size(integer)";
}

extern "C" char* rvs_module_get_output(void)
{
	return (char*)"bar1_size(integer), bar1_base_addr(integer), bar2_size(integer), bar2_base_addr(integer), bar4_size(integer), bar4_base_addr(integer), bar5_size(integer), pass(bool)";
}



extern "C" int   rvs_module_init(void* pMi)
{
  rvs::lp::Initialize(static_cast<T_MODULE_INIT*>(pMi));
	return 0;
}

extern "C" int   rvs_module_terminate(void)
{
	return 0;
}

extern "C" char* rvs_module_get_errstring(int error)
{
	return  (char*)"General Error";
}

extern "C" void* rvs_module_action_create(void)
{
	return static_cast<void*>(new action);
}

extern "C" int   rvs_module_action_destroy(void* pAction)
{
	delete static_cast<rvs::lib::actionbase*>(pAction);
	return 0;
}

extern "C" int rvs_module_action_property_set(void* pAction, const char* Key, const char* Val)
{
	return static_cast<rvs::lib::actionbase*>(pAction)->property_set(Key, Val);
}

extern "C" int rvs_module_action_run(void* pAction)
{
	return static_cast<rvs::lib::actionbase*>(pAction)->run();
}
