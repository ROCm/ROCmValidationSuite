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
#include "rvsmodule.h"



using namespace std;

int listmodules(void)
{
    YAML::Node config = YAML::LoadFile("./rvsmodules.config");
    
    YAML::const_iterator it=config.begin();
    std::string key=it->first.as<std::string>();
    std::string value=it->second.as<std::string>();
    if(value!="1")
    {
        printf("ERROR: version is not 1");
        return -1;
    }
    
    for(it++; it!=config.end(); ++it) 
    {
        std::string key=it->first.as<std::string>();
        std::string value=it->second.as<std::string>();
        std::cout << "Key: " << key <<"\n"; 
        std::cout << "Value: " << value << "\n";
        
        rvsmodule* module = rvsmodule::create(value.c_str());
        rvsif0* pif0 = (rvsif0*)module->get_interface(0);

        
        fprintf(stdout, "Module: %s\n", pif0->get_name() );

        fprintf(stdout, "Description: %s\n", pif0->get_description() );
        int Major,Minor,Patch;
        pif0->get_version(&Major,&Minor,&Patch);
        std::cout << "Version: " << Major << "." << Minor << "." << Patch << "\n";
        rvsmodule::destroy(module);
    }
    return 0;
}

int main(int Argc, char**Argv)
{
    for (int i=1;i<Argc;i++)
    {
        if (!strcmp(Argv[i],"--modules"))
        {
        int status=listmodules();
        }
    }
    
	return 0;
}




