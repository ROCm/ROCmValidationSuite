
#ifndef RVSMODULE_IF1_H_
#define RVSMODULE_IF1_H_

extern "C"
{



extern 	char* rvs_module_get_errstring(int error);
extern 	int   rvs_module_action_property_set(void* Action, const char* Key, const char* Val);
extern 	int   rvs_module_action_run(void* Action);

// define function pointer types to ease late binding usage
typedef char* (*t_rvs_module_get_errstring)(int error);
typedef int   (*t_rvs_module_action_property_set)(void* Action, const char* Key, const char* Val);
typedef int   (*t_rvs_module_action_run)(void* Action);

}

#endif  // RVSMODULE_IF1_H_
