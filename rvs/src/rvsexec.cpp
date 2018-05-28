
#include <iostream>
#include <memory>
#include "yaml-cpp/yaml.h"

#include "rvsif0.h"
#include "rvsif1.h"
#include "rvsaction.h"
#include "rvsmodule.h"
#include "rvsliblogger.h"

#include "rvsexec.h"



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
		do_list_modules();
		return 0;
	}
	
	try
	{
		sts = do_yaml(config_file);
	} catch(...)
	{
		cerr << "Error parsing configuration file: " << config_file << endl;
	}

// 	rvs::action* pa = rvs::module::action_create("gpup");
// 	if(!pa)
// 	{
// 		cerr << "ERROR: could not create 'gpup' action." << endl;
// 		return -1;
// 	}
// 	
// 	rvs::if0* pif0 = dynamic_cast<rvs::if0*>(pa->get_interface(0));
// 	if(!pif0)
// 	{
// 		cerr << "ERROR: could not get interface 'IF0'." << endl;
// 		return -1;
// 	}
// 	
// 	std::cout << "Module: " << pif0->get_name() << endl;
// 	std::cout << "Description: " << pif0->get_description() << endl;
// 	std::cout << "config: " << pif0->get_config() << endl;
// 	std::cout << "output: " << pif0->get_output() << endl;
// 
// 	rvs::if1* pif1 = dynamic_cast<rvs::if1*>(pa->get_interface(1));
// 	if(!pif0)
// 	{
// 		cerr << "ERROR: could not get interface 'IF1'." << endl;
// 		return -1;
// 	}
// 	
// // 	sts = pif1->property_set("simd_count mem_banks_count caches_count io_links_count cpu_core_id_base simd_id_base max_waves_per_simd lds_size_in_kb gds_size_in_kb wave_front_size array_count simd_arrays_per_engine cu_per_simd_array simd_per_cu max_slots_scratch_cu vendor_id device_id location_id drm_render_minor max_engine_clk_fcompute local_mem_size fw_version capability max_engine_clk_ccompute");
// 	pif1->property_set("simd_count", "");
// 	pif1->property_set("cpu_cores_count", "" );
// 
// 	sts = pif1->run();
// 	
// 	rvs::module::action_destroy(pa);

	rvs::module::terminate();
	
	return sts;
}

void rvs::exec::do_version()
{
	cout << "1.0.0" << endl;
}

void rvs::exec::do_help()
{
	cout << "No help available." << endl;
}

void rvs::exec::do_list_modules()
{
// 	rvs::action* pa = rvs::module::action_create("gpup");
// 	if(!pa)
// 	{
// 		cerr << "ERROR: could not create 'gpup' action." << endl;
// 		return -1;
// 	}
// 	
// 	rvs::if0* pif0 = dynamic_cast<rvs::if0*>(pa->get_interface(0));
// 	if(!pif0)
// 	{
// 		cerr << "ERROR: could not get interface 'IF0'." << endl;
// 		return -1;
// 	}
// 	
// 	std::cout << "Module: " << pif0->get_name() << endl;
// 	std::cout << "Description: " << pif0->get_description() << endl;
// 	std::cout << "config: " << pif0->get_config() << endl;
// 	std::cout << "output: " << pif0->get_output() << endl;

	
}

