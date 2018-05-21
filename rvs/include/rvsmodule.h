

#ifndef RVSMODULE_H_
#define RVSMODULE_H_

#include "rvsmodule_if.h"
	
namespace rvs 
{

class ifbase;
class action;


class module
{
	typedef std::pair<std::string,module*> t_mmpair;
	
	// collection related members
public:	
    static int     		initialize(const char* pConfigName);
    static action*		action_create(const char* pModuleShortName);
	static int       	action_destroy(action*);
    static int         	terminate();

protected:
	static 	module* 	find_create_module(const char* pShortName);
	// YAML configuration
	static YAML::Node	config;
	
	// short name -> rvsmodule*
    static std::map<std::string,module*>		modulemap;
	
	// short name -> .so filename
    static std::map<std::string,std::string>	filemap;

protected:
						module(const char* pModuleShortName, void* pSoLib);
	virtual 			~module();

protected:
			void* 		psolib;
			std::string name;
	
			int 		initialize();
			int  		terminate_internal();
protected:	
			void* 		action_create();
			int        	action_destroy_internal(action*);
			
protected:	
			int 		init_interfaces(void);
			int 		init_interface_method(void** ppfunc, const char* pMethodName);
			int 		init_interfaces(action*);
			int 		init_interface_0(void);
			int 		init_interface_1(void);
			
protected:
			std::map<int,std::shared_ptr<ifbase>> ifmap;

			t_rvs_module_init			rvs_module_init;
			t_rvs_module_terminate		rvs_module_terminate;
			t_rvs_module_action_create	rvs_module_action_create;
			t_rvs_module_action_destroy	rvs_module_action_destroy;
	
	

};


} // namespace rvs


#endif /* RVSMODULE_H_ */
