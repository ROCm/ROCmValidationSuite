
#include <iostream>
#include <memory>
#include "yaml-cpp/yaml.h"

#include "rvsif0.h"
#include "rvsif1.h"
#include "rvsaction.h"
#include "rvsmodule.h"
#include "rvsliblogger.h"

#include "rvsexec.h"
#define VER "BUILD_VERSION_STRING"


rvs::exec::exec(const rvs::exec::options_t& rOptions)
{
	options = rOptions;
}

rvs::exec::~exec()
{
}

bool  rvs::exec::has_option(const char* pOptions, string& val)
{
	auto it = options.find(string(pOptions));
	if( it == options.end())
		return false;
	
	val = it->second;
	return true;
}

int rvs::exec::run()
{
	int 	sts;
	string 	val;
	
	// check -h options
	if( has_option("-h", val))
	{
		do_help();
		return 0;
	}
		
	// check -v options
	if( has_option("-v", val))
	{
		do_version();
		return 0;
	}
		
	// check -d options
	if( has_option("-d", val))
	{
		int level;
		try 
		{ 
			level = std::stoi(val);
		}
		catch(...)
		{ 
			cerr << "ERROR: syntax error: logging level not integer: " << val <<endl;
			return -1;
		}
		if( level < 0 || level > 5)
		{
			cerr << "ERROR: syntax error: logging level not in range [0..5]: " << val <<endl;
			return -1;
		}
//		rvs::lib::logger::log_level(level);
		lib::logger::log_level(level);
	}
	
	string config_file;
	if( has_option("-c", val))
	{
		config_file = val;
	}
	else
	{
		config_file = "conf/rvs.conf";
	}
		
    rvs::module::initialize("./rvsmodules.config");

	if( has_option("-t", val))
	{
        cout<< endl << "ROCm Validation Suite (version " << LIB_VERSION_STRING << ")" << endl << endl;
        cout<<"Modules available:"<<endl;
		rvs::module::do_list_modules();
		return 0;
	}
	
	try
	{
		sts = do_yaml(config_file);
	} catch(...)
	{
		cerr << "Error parsing configuration file: " << config_file << endl;
	}

	rvs::module::terminate();
	
	return sts;
}

void rvs::exec::do_version()
{
	cout << LIB_VERSION_STRING << endl;
}

void rvs::exec::do_help()
{
	cout << "No help available." << endl;
}

