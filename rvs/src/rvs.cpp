//============================================================================
// Name        : rvs.cpp
// Author      : HDL-DH
// Version     :
// Copyright   : (c) 2018
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <stdio.h>
#include <iostream>
#include <dlfcn.h>
#include <string.h>

#include <fstream>
#include "yaml-cpp/yaml.h"

#include "rvsif0.h"
#include "rvsif1.h"
#include "rvsaction.h"
#include "rvsmodule.h"
#include "rvscli.h"
#include "rvsliblogger.h"



using namespace std;

// TOTO: move into rvs::module
// int listmodules(void)
// {
//     YAML::Node config = YAML::LoadFile("./rvsmodules.config");
//     
//     YAML::const_iterator it=config.begin();
//     std::string key=it->first.as<std::string>();
//     std::string value=it->second.as<std::string>();
//     if(value!="1")
//     {
//         printf("ERROR: version is not 1");
//         return -1;
//     }
//     
//     for(it++; it!=config.end(); ++it) 
//     {
//         std::string key=it->first.as<std::string>();
//         std::string value=it->second.as<std::string>();
//         std::cout << "Key: " << key <<"\n"; 
//         std::cout << "Value: " << value << "\n";
//         
//         rvs::module* module = rvs::module::create(value.c_str());
//         rvs::if0* pif0 = (rvs::if0*)rvs::module->get_interface(0);
// 
//         
//         fprintf(stdout, "Module: %s\n", pif0->get_name() );
// 
//         fprintf(stdout, "Description: %s\n", pif0->get_description() );
//         int Major,Minor,Patch;
//         pif0->get_version(&Major,&Minor,&Patch);
//         std::cout << "Version: " << Major << "." << Minor << "." << Patch << "\n";
//         rvs::module::destroy(module);
//     }
//     return 0;
// }

int main(int Argc, char**Argv)
{
	int sts;
	rvs::cli cli;
	
// 	sts =  cli.parse(Argc, Argv);
// 	if(sts)
// 	{
// 		cout<< "ERROR: error parsing command line:" << cli.get_error_string() << endl;
// 		return -1;
// 	}
    
    rvs::module::initialize("./rvsmodules.config");

	rvs::action* pa = rvs::module::action_create("gpup");
	if(!pa)
	{
		cerr << "ERROR: could not create 'gpup' action." << endl;
		return -1;
	}
	
	rvs::if0* pif0 = dynamic_cast<rvs::if0*>(pa->get_interface(0));
	if(!pif0)
	{
		cerr << "ERROR: could not get interface 'IF0'." << endl;
		return -1;
	}
	
	std::cout << "Module: " << pif0->get_name() << endl;
	std::cout << "Description: " << pif0->get_description() << endl;

	rvs::if1* pif1 = dynamic_cast<rvs::if1*>(pa->get_interface(1));
	if(!pif0)
	{
		cerr << "ERROR: could not get interface 'IF1'." << endl;
		return -1;
	}
	
// 	sts = pif1->property_set("simd_count mem_banks_count caches_count io_links_count cpu_core_id_base simd_id_base max_waves_per_simd lds_size_in_kb gds_size_in_kb wave_front_size array_count simd_arrays_per_engine cu_per_simd_array simd_per_cu max_slots_scratch_cu vendor_id device_id location_id drm_render_minor max_engine_clk_fcompute local_mem_size fw_version capability max_engine_clk_ccompute");
	pif1->property_set("simd_count", "");
	pif1->property_set("cpu_cores_count", "" );

	sts = pif1->run();
	
	rvs::module::action_destroy(pa);

	//dummy peqt check
	rvs::action* peqt_action = rvs::module::action_create("peqt");
	if(!peqt_action)
	{
		cerr << "ERROR: could not create 'peqt' action." << endl;
		return -1;
	}
	
	rvs::if0* peqt_if0 = dynamic_cast<rvs::if0*>(peqt_action->get_interface(0));
	if(!peqt_if0)
	{
		cerr << "ERROR: could not get interface 'IF0'." << endl;
		return -1;
	}
	
	std::cout << "Module: " << peqt_if0->get_name() << endl;
	std::cout << "Description: " << peqt_if0->get_description() << endl;

	rvs::if1* peqt_if1 = dynamic_cast<rvs::if1*>(peqt_action->get_interface(1));
	if(!peqt_if1)
	{
		cerr << "ERROR: could not get interface 'IF1'." << endl;
		return -1;
	}
	

	sts = peqt_if1->run();
	
	rvs::module::action_destroy(peqt_action);


	rvs::module::terminate();
    
	return 0;
}




