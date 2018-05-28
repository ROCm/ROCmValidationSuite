

#include <stdio.h>
#include <iostream>
#include <dlfcn.h>
#include "yaml-cpp/yaml.h"

#include "rvsliblogger.h"
#include "rvsif0.h"
#include "rvsif1.h"
#include "rvsaction.h"
#include "rvsmodule.h"

std::map<std::string,rvs::module*>	rvs::module::modulemap;
std::map<std::string,std::string>	rvs::module::filemap;
YAML::Node							rvs::module::config;

using namespace std;

rvs::module::module(const char* pModuleShortName, void* pSoLib)
: psolib(pSoLib),
name(pModuleShortName)
{
}

rvs::module::~module()
{
}

int rvs::module::initialize(const char* pConfig)
{
	// load list of supported modules from config file
	YAML::Node config = YAML::LoadFile(pConfig);
	
	// verify that that the file format is supported
    YAML::const_iterator it=config.begin();
	if( it == config.end())
    {
        cerr << "ERROR: unsupported file format. Version string not found." << endl;
        return -1;
    }
		
    std::string key=it->first.as<std::string>();
    std::string value=it->second.as<std::string>();
	
	if(key != "version")
	{
		cerr << "ERROR: unsupported file format. Version string not found." << endl;
		return -1;
	}
	
    if(value!="1")
    {
        cerr << "ERROR: version is not 1" << endl;
        return -1;
    }
    
    // load nam-file pairs:
    for(it++; it!=config.end(); ++it) 
    {
        key=it->first.as<std::string>();
        value=it->second.as<std::string>();
//        std::cout << "name: " << key << endl; 
//        std::cout << "lib: " << value << endl;
        
		filemap.insert( pair<string,string>(key, value));
    }
	
	return 0;
}

rvs::module* rvs::module::find_create_module(const char* name)
{
	module* m = NULL;
	
	// find module based on short name
	auto it = modulemap.find(std::string(name));
	
	// not found...
	if (it == modulemap.end())
	{
		// ... try opening .so
		
		// first find proper .so filename
		auto it = filemap.find(std::string(name));
		
		// not found...
		if( it == filemap.end())
		{
			// this should never happen if .config is OK
			cerr << "ERROR: module '" << name << "' not found in configuration." << endl;
			return NULL;
		}
		
		// open .so 
		void* psolib = dlopen(it->second.c_str(), RTLD_LAZY);
		
		// error?
		if( !psolib)
		{
			cerr << "ERROR: could not load .so '" << it->second.c_str() << "' reason: " << dlerror() << endl;
			return NULL;	// fail
		}

		// create module object
		m = new rvs::module(name, psolib);
		if(!m)
		{
			dlclose(psolib);
			return NULL;
		}
		
		// initialize API function pointers
		if(m->init_interfaces())
		{
			cerr << "ERROR: could not init interfaces for '" << it->second.c_str() << "'" << endl;
			dlclose(psolib);
			delete m;
			return nullptr;
		}
		
		// initialize newly loaded module
		if(m->initialize())
		{
			cerr << "ERROR: could not initialize '" << it->second.c_str() << "'" << endl;
			dlclose(psolib);
			delete m;
			return nullptr;
		}
		
		// add to map
		modulemap.insert(t_mmpair(name, m));
	}
	else
	{
		m = it->second;
	}

	return m;
}

int rvs::module::initialize()
{
	return (*rvs_module_init)((void*)LoggerCallback);
}


rvs::action* rvs::module::action_create(const char* name)
{
	// find module
	rvs::module* m = module::find_create_module(name);
	if( !m)
	{
		cerr << "ERROR: module '" << name << "' not available." << endl;
		return nullptr;
	}
	
	// create lib action objct
	void* plibaction = m->action_create();
	if(!plibaction)
	{
		cerr << "ERROR: module '" << name << "' could not create lib action." << endl;		
		return nullptr;
	}
	
	// create action proxy object
	rvs::action* pa = new rvs::action(name, plibaction);
	if(!pa)
	{
		cerr << "ERROR: module '" << name << "' could not create action proxy." << endl;	
		return nullptr;
	}
	
	// create interfaces for the proxy
	// clone from module and assign libaction ptr
	
	for(auto it = m->ifmap.begin(); it != m->ifmap.end(); it++)
	{
		std::shared_ptr<rvs::ifbase> sptrif(it->second->clone());
		sptrif->plibaction = plibaction;
		pa->ifmap.insert(rvs::action::t_impair(it->first, sptrif));
	}

	return pa;
	
}

void* rvs::module::action_create()
{
	return (*rvs_module_action_create)();
}


int rvs::module::action_destroy(rvs::action* paction)
{
	// find module
	rvs::module* m = module::find_create_module(paction->name.c_str());
	if( !m)
		return -1;
	
	return m->action_destroy_internal(paction);
}

int rvs::module::action_destroy_internal(rvs::action* paction)
{
	int sts = (*rvs_module_action_destroy)(paction->plibaction);
	delete paction;
	
	return sts;
}

int rvs::module::terminate()
{
	for(auto it = rvs::module::modulemap.begin(); it != rvs::module::modulemap.end(); it++) 
    {
		it->second->terminate_internal();
		dlclose(it->second->psolib);
		delete it->second;
	}
	
	modulemap.clear();

	return 0;
}

int rvs::module::terminate_internal()
{
	return (*rvs_module_terminate)();
}


int rvs::module::init_interfaces()
{
	// init global helper methods for this library
	if( init_interface_method( (void**)(&rvs_module_init), "rvs_module_init"))
		return -1;
		
	if( init_interface_method( (void**)(&rvs_module_terminate), "rvs_module_terminate"))
		return -1;
		
	if( init_interface_method( (void**)(&rvs_module_action_create), "rvs_module_action_create"))
		return -1;
		
	if( init_interface_method( (void**)(&rvs_module_action_destroy), "rvs_module_action_destroy"))
		return -1;
		
	if( init_interface_0())
		return -1;
		
	if( init_interface_1())
		return -1;
	
	return 0;
}

int rvs::module::init_interface_method(void** ppfunc, const char* pMethodName)
{
	if (!psolib)
	{
		cerr << "ERROR: psolib is null. " << endl;
		return -1;
	}
	void* pf = dlsym(psolib, pMethodName);
	if (!pf)
	{
		cerr << "ERROR: could not find .so method '" << pMethodName << "'" << endl;
	}
	
	*ppfunc = pf;
	
	return 0;
}

int rvs::module::init_interface_0(void)
{
	rvs::if0* pif0 = new rvs::if0();
	if(!pif0)
		return -1;
	
	int sts = 0;

	if( init_interface_method( (void**)(&(pif0->rvs_module_get_version)), "rvs_module_get_version"))
		sts--;

	if( init_interface_method( (void**)(&(pif0->rvs_module_get_name)), "rvs_module_get_name"))
		sts--;

	if( init_interface_method( (void**)(&(pif0->rvs_module_get_description)), "rvs_module_get_description"))
		sts--;

	if( init_interface_method( (void**)(&(pif0->rvs_module_has_interface)), "rvs_module_has_interface"))
		sts--;

	if( init_interface_method( (void**)(&(pif0->rvs_module_get_config)), "rvs_module_get_config"))
		sts--;

	if( init_interface_method( (void**)(&(pif0->rvs_module_get_output)), "rvs_module_get_output"))
		sts--;

	if(sts)
	{
		delete pif0;
		return sts;
	}

	std::shared_ptr<rvs::ifbase> sptr( (rvs::ifbase*)pif0);
	ifmap.insert(rvs::action::t_impair(0, sptr));
	
	return 0;
}

int rvs::module::init_interface_1(void)
{
	rvs::if1* pif1 = new rvs::if1();
	if(!pif1)
		return -1;
	
	int sts = 0;
	if( init_interface_method( (void**)(&(pif1->rvs_module_action_property_set)), "rvs_module_action_property_set"))
		sts--;

	if( init_interface_method( (void**)(&(pif1->rvs_module_action_run)), "rvs_module_action_run"))
		sts--;

	if( init_interface_method( (void**)(&(pif1->rvs_module_get_errstring)), "rvs_module_get_errstring"))
		sts--;

	if(sts)
	{
		delete pif1;
		return sts;
	}

	std::shared_ptr<rvs::ifbase> sptr( (rvs::ifbase*)pif1);
	ifmap.insert(rvs::action::t_impair(1, sptr));
	
	return 0;
}





