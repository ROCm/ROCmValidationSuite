
#ifndef RVSMODULE_IF_H_
#define RVSMODULE_IF_H_

extern "C"
{
extern	int   rvs_module_init(void*);
extern 	int   rvs_module_terminate(void);
extern 	void* rvs_module_action_create(void);
extern 	int   rvs_module_action_destroy(void*);

// define function pointer types to ease late binding usage
typedef int   (*t_rvs_module_init)(void*);
typedef int   (*t_rvs_module_terminate)(void);
typedef void* (*t_rvs_module_action_create)(void);
typedef int   (*t_rvs_module_action_destroy)(void*);

	
}

#endif  // RVSMODULE__F_H_
