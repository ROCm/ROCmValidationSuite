#include "gpu_util.h"

#include <stdlib.h>
#include <dirent.h>
#include <string.h>
#include <fstream>

using namespace std;

int gpu_num_subdirs(char* dirpath, char* prefix)
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

/**
 * gets all GPUS location_id
 * @param gpus_location_id the vector that will store all the GPU location_id
 * @return
 */
void gpu_get_all_location_id( std::vector<unsigned short int>& gpus_location_id)
{
	ifstream f_id,f_prop;
	char path[KFD_PATH_MAX_LENGTH];

    std::string prop_name;
    int gpu_id;
    unsigned short int prop_val;


    //Discover the number of nodes: Inside nodes folder there are only folders that represent the node number
    int num_nodes = gpu_num_subdirs((char*)KFD_SYS_PATH_NODES, (char*)"");

    //get all GPUs device id
    for(int node_id=0; node_id<num_nodes; node_id++){
    	snprintf(path, KFD_PATH_MAX_LENGTH, "%s/%d/gpu_id", KFD_SYS_PATH_NODES, node_id);
        f_id.open(path);
        snprintf(path, KFD_PATH_MAX_LENGTH, "%s/%d/properties", KFD_SYS_PATH_NODES, node_id);
        f_prop.open(path);

        f_id >> gpu_id;

        if (gpu_id != 0){
            while(f_prop >> prop_name){
                if (prop_name == "location_id"){
                f_prop >> prop_val;
                gpus_location_id.push_back(prop_val);
                break;
                }
            }
        }

        f_id.close();
        f_prop.close();
     }
}
