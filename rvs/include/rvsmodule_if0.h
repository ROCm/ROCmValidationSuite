
#ifndef RVSMODULE_IF0_H_
#define RVSMODULE_IF0_H_

extern "C"
{

extern void  rvs_module_get_version(int* Major, int* Minor, int* Revision);
extern char* rvs_module_get_name(void);
extern char* rvs_module_get_description(void);
extern int   rvs_module_has_interface(int InterfaceID);


// define function pointer types to ease late binding usage
typedef void  (*t_rvs_module_get_version)(int*, int*, int*);
typedef char* (*t_rvs_module_get_name)(void);
typedef char* (*t_rvs_module_get_description)(void);
typedef int   (*t_rvs_module_has_interface)(int);

}

#endif  // RVSMODULE_IF0_H_
