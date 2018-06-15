#include "rvsexec.h"

#include <iostream>
#include <memory>
#include "yaml-cpp/yaml.h"

#include "rvsif0.h"
#include "rvsif1.h"
#include "rvsaction.h"
#include "rvsmodule.h"
#include "rvsliblogger.h"
#include "rvsoptions.h"

/*** Example rvs.conf file structure

actions:
- name: action_1
  device: all
  module: gpup
  properties:
    mem_banks_count:
  io_links-properties:
    version_major:
- name: action_2
  module: gpup
  device: all

***/

using namespace rvs;

int rvs::exec::do_yaml(const string& config_file)
{
	int sts = 0;
	
	YAML::Node config = YAML::LoadFile(config_file);
	
	// find "actions" map
	const YAML::Node& actions = config["actions"];
	if1* pif1 = nullptr;
	
	// for all actions...
	for (YAML::const_iterator it = actions.begin(); it != actions.end(); ++it) 
	{
		const YAML::Node& action = *it;
		
		// find module name 
		string rvsmodule = action["module"].as<std::string>();
		
		// not found or empty
		if( rvsmodule == "") 
		{	
			// report error and go to next action
			cerr << "ERROR: action '"<< action["name"].as<std::string>() << "' does not specify module." << endl;
			continue;
		}
		
		// create action excutor in .so
		rvs::action* pa = module::action_create(rvsmodule.c_str());
		if(!pa)
		{
			cerr << "ERROR: action '"<< action["name"].as<std::string>() << "' could not crate action object in module '" << rvsmodule.c_str() << "'"<< endl;
			continue;
		}
			
		// obtain interface to set parameters and execute action
		if1* pif1 = (if1*)(pa->get_interface(1));
		if(!pif1)
		{
			cerr << "ERROR: action '"<< action["name"].as<std::string>() << "' could not obtain interface IF1"<< endl;
			module::action_destroy(pa);
			continue;
		}
			
		// load action properties from yaml file
		sts += do_yaml_properties(action, rvsmodule, pif1);
		if(sts)
		{
			module::action_destroy(pa);
			return sts;
		}
		
		// set also command line options:
		for (auto clit = rvs::options::get().begin(); clit != rvs::options::get().end(); ++clit) {
			string p(clit->first);
			p = "cli." + p;
			pif1->property_set(p, clit->second);
		}
		
		// execute action
		sts = pif1->run();

		// procssin finished, release action object
		module::action_destroy(pa);

		// errors?
		if(sts)
		{
			// cancel actions and return
			return sts;
		}
		
	}	
	
	
	return 0;
}

int rvs::exec::do_yaml_properties(const YAML::Node& node, const string& module_name, rvs::if1* pif1)
{
	int sts = 0;
	
	// for all child nodes
	for (YAML::const_iterator it = node.begin(); it != node.end(); it++) 
	{
		const YAML::Node& child = *it;
		
		// if property is collection of module specific properties,
		if(is_yaml_properties_collection(module_name, it->first.as<string>()))
		{
			// pass properties collection to .so action object
			sts += do_yaml_properties_collection(it->second, it->first.as<string>(), pif1);
		}
		else
		{
			// just set this one propertiy
			sts += pif1->property_set(it->first.as<string>(), it->second.as<string>());
		}
	}
	
	return sts;
	
}

int rvs::exec::do_yaml_properties_collection(const YAML::Node& node, const string& parent_name, if1* pif1)
{
	int sts = 0;
	
	// for all child nodes
	for (YAML::const_iterator it = node.begin(); it != node.end(); it++) 
	{
		// prepend dot separated parent name and pass property to module
		sts += pif1->property_set(parent_name + "." + it->first.as<string>(),
		it->second.IsNull() ? string("") : it->second.as<string>());
	}
	
	return sts;
	
}

// check if particular property for a module is collection property
bool rvs::exec::is_yaml_properties_collection(const string& module_name, const string& property_name)
{
	if( module_name == "gpup")
	{
		if(property_name == "properties")
			return true;
		
		if(property_name == "io_links-properties")
			return true;
	}
	else
		if( module_name == "peqt"){
		if(property_name == "capability")
			return true;
	}

	return false;
	
}

