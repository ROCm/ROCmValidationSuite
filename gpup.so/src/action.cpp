
#include <iostream>
#include <stdlib.h>
#include <dirent.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <map>
#include "rvs_module.h"
#include "action.h"
#include "gpu_util.h"

#define KFD_SYS_PATH_NODES "/sys/class/kfd/kfd/topology/nodes"

using namespace std;


action::action()
{
}

action::~action()
{
	property.clear();
}

int action::property_set(const char* Key, const char* Val)
{
	return rvs::lib::actionbase::property_set(Key, Val);
}

int action::run(void)
{    
        ifstream f_id,f_prop,f_link_prop;
        char path[256];
        string prop_name, action_name = "[]";
        int gpu_id, num_links;
        unsigned long int prop_val;
        string msg,s;
        std::map<string,string>::iterator it;
        
        //Discover the number of nodes: Inside nodes folder there are only folders that represent the node number
        int num_nodes = gpu_num_subdirs((char*)KFD_SYS_PATH_NODES, (char*)"");
                
        for(int node_id=0; node_id<num_nodes; node_id++){
                snprintf(path, 256, "%s/%d/gpu_id", KFD_SYS_PATH_NODES, node_id);
                f_id.open(path);
                snprintf(path, 256, "%s/%d/properties", KFD_SYS_PATH_NODES, node_id);
                f_prop.open(path);
    
                // TODO check also if device name is "all" and if property name is "all" 
                for (it=property.begin(); it!=property.end(); ++it){
                    s = it->first;
                    if( s == "name")
                         action_name = it->second;
                }               
                
                f_id >> gpu_id;
                
                if ( gpu_id != 0){
                    for (it=property.begin(); it!=property.end(); ++it){
                    //Discover the number of io_links: Inside iolinks folder there are only folders that represent the link number
                    // TODO in case that value for io_links_count is provided use it instead of counting files 
                    snprintf(path, 256, "%s/%d/io_links", KFD_SYS_PATH_NODES, node_id);
                    num_links = gpu_num_subdirs((char*)path, (char*)"");
                    
                    s = it->first;

                    if( s.find(".")!= std::string::npos && s.substr(0,s.find(".")) == "properties"){
                    while(f_prop >> prop_name){
                        if (prop_name == s.substr(s.find(".")+1)){
                        f_prop >> prop_val;
                        msg = action_name + " gpup " + std::to_string(gpu_id) + " "+ prop_name + " " + std::to_string(prop_val);
                        log( msg.c_str(), rvs::logresults);
                        break;
                        }
                        f_prop >> prop_val;
                    }
                    }
                    f_prop.clear();
                    f_prop.seekg( 0, std::ios::beg );
                    
                     if( s.find(".")!= std::string::npos && s.substr(0,s.find(".")) == "io_links-properties"){                 
                     for(int link_id=0; link_id<num_links; link_id++){
                         snprintf(path, 256, "%s/%d/io_links/%d/properties", KFD_SYS_PATH_NODES, node_id, link_id);
                         f_link_prop.open(path);
                         while(f_link_prop >> prop_name){
                         if (prop_name == s.substr(s.find(".")+1)){
                                 f_link_prop >> prop_val;
                                 msg = action_name + " gpup " + std::to_string(gpu_id) + " " + std::to_string(link_id) + " "+ prop_name + " " + std::to_string(prop_val);
                                 log( msg.c_str(), rvs::logresults);
                                 break;
                             }
                             f_prop >> prop_val;
                         }
                         f_link_prop.close();
                     }
                     }
                    }
                }
                f_id.close();
                f_prop.close();
            }        
        return 0;
}