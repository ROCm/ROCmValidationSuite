
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

#define KFD_SYS_PATH_NODES "/sys/class/kfd/kfd/topology/nodes"

using namespace std;

static int num_subdirs(char *dirpath, char *prefix)
{
    int count = 0;
    DIR *dirp;
    struct dirent *dir;
    int prefix_len = strlen(prefix);

    dirp = opendir(dirpath);
    if (dirp){
        while ((dir = readdir(dirp)) != 0){
            if((strcmp(dir->d_name, ".") == 0) ||
            (strcmp(dir->d_name, "..") == 0))
            continue;
            if(prefix_len &&
            strncmp(dir->d_name, prefix, prefix_len))
            continue;
        count++;
        }
        closedir(dirp);
    }
    return count;		
}

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
        int gpu_id, prop_val, num_links;
        string msg;
        std::map<string,string>::iterator it;
        
        it=property.find("action_name");
        if ( it != property.end() ){
            action_name = it->second;
            property.erase(it);
        }
        //Discover the number of nodes: Inside nodes folder there are only folders that represent the node number
        int num_nodes = num_subdirs((char*)KFD_SYS_PATH_NODES, (char*)"");
        
            for(int node_id=0; node_id<num_nodes; node_id++){
                snprintf(path, 256, "%s/%d/gpu_id", KFD_SYS_PATH_NODES, node_id);
                f_id.open(path);
                snprintf(path, 256, "%s/%d/properties", KFD_SYS_PATH_NODES, node_id);
                f_prop.open(path);
    
                f_id >> gpu_id;
                
                if ( gpu_id != 0){
                    for (it=property.begin(); it!=property.end(); ++it){
                    //Discover the number of io_links: Inside iolinks folder there are only folders that represent the link number
                    // TODO in case that value for io_links_count is provided use it instead of counting files 
                    snprintf(path, 256, "%s/%d/io_links", KFD_SYS_PATH_NODES, node_id);
                    num_links = num_subdirs((char*)path, (char*)"");
                    
                    while(f_prop >> prop_name){
                        if (prop_name == it->first){
                        f_prop >> prop_val;
                        msg = action_name + " gpup " + std::to_string(gpu_id) + " "+ prop_name + " " + std::to_string(prop_val);
                        log( msg.c_str(), rvs::logresults);
                        break;
                        }
                    }
                    for(int link_id=0; link_id<num_links; link_id++){
                        snprintf(path, 256, "%s/%d/io_links/%d/properties", KFD_SYS_PATH_NODES, node_id, link_id);
                        f_link_prop.open(path);
                        while(f_link_prop >> prop_name){
                            if (prop_name == it->first){
                                f_link_prop >> prop_val;
                                msg = action_name + " gpup " + std::to_string(gpu_id) + " " + std::to_string(link_id) + " "+ prop_name + " " + std::to_string(prop_val);
                                log( msg.c_str(), rvs::logresults);
                                break;
                            }
                        }
                        f_link_prop.close();
                    } 
                    }
                }
                f_id.close();
                f_prop.close();
            }        
        return 0;
}