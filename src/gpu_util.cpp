/********************************************************************************
 *
 * Copyright (c) 2018 ROCm Developer Tools
 *
 * MIT LICENSE:
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to do
 * so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include "gpu_util.h"

#include <stdlib.h>
#include <dirent.h>
#include <string.h>
#include <fstream>
#include <map>

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

/**
 * gets all GPUS gpu_id
 * @param gpus_id the vector that will store all the GPU gpu_id
 * @return
 */
void gpu_get_all_gpu_id(std::vector<unsigned short int>& gpus_id)
{
    ifstream f_id,f_prop;
    char path[KFD_PATH_MAX_LENGTH];

    int gpu_id;

    //Discover the number of nodes: Inside nodes folder there are only folders that represent the node number
    int num_nodes = gpu_num_subdirs((char*)KFD_SYS_PATH_NODES, (char*)"");

    //get all GPUs device id
    for(int node_id=0; node_id<num_nodes; node_id++){
        snprintf(path, KFD_PATH_MAX_LENGTH, "%s/%d/gpu_id", KFD_SYS_PATH_NODES, node_id);
        f_id.open(path);

        f_id >> gpu_id;

        if (gpu_id != 0)
            gpus_id.push_back(gpu_id);

        f_id.close();
     }
}

