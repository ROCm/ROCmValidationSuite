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

#include <fstream>
#include "yaml-cpp/yaml.h"

#include "rvsif0.h"
#include "rvsif1.h"
#include "rvsmodule.h"



using namespace std;


int main(int Argc, char**Argv) {
	cout << "Hello from RVS..." << endl; 

    YAML::Node config = YAML::LoadFile("./rvsmodules.config");

//     std::cout << "config.Type(): " << config.Type() << "\n";
//     std::cout << "config size: " << config.size() << "\n";
//     std::cout << "config[0].Type(): " << config["version"].Type() << "\n";
 

    for(YAML::const_iterator it=config.begin(); it!=config.end(); ++it) 
    {
        std::cout << "Key: " << it->first.as<std::string>() << "   Value: " << it->second.as<std::string>() << "\n";
    }
    

	rvsmodule* module = rvsmodule::create("libgpup.so.1");
	if(!module)
	{
		fprintf(stdout, "Could not load module.\n");
		return 1;
	}

	rvsif0* pif0 = (rvsif0*)module->get_interface(0);


	fprintf(stdout, "Module: %s\n", pif0->get_name() );

	fprintf(stdout, "Description: %s\n", pif0->get_description() );

	rvsmodule::destroy(module);

	return 0;
}




