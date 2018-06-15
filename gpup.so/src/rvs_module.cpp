
#include "rvs_module.h"
#include "action.h"
#include "rvsloglp.h"

int log(const char* pMsg, const int level)
{
	return rvs::lp::Log(pMsg, level);
}

extern "C" void  rvs_module_get_version(int* Major, int* Minor, int* Revision)
{
	*Major = 1;
	*Minor = 0;
	*Revision = 0;
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
   return (char*)"gpup";
}

extern "C" char* rvs_module_get_description(void)
{
   return (char*)"ROCm Validation Suite GPUP module";
}

extern "C" char* rvs_module_get_config(void)
{
	return (char*)"module (string), version (string), installed (bool), user (string), groups (collection of strings), file (string), owner (string), group (string), permission (int), type (int), exists (bool)";
}

extern "C" char* rvs_module_get_output(void)
{
	return (char*)"pass (bool)";
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


