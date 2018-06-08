#ifndef _GPU_UTIL_H_
#define _GPU_UTIL_H_

#include <vector>

#define KFD_SYS_PATH_NODES              "/sys/class/kfd/kfd/topology/nodes"
#define KFD_PATH_MAX_LENGTH             256

extern int  gpu_num_subdirs(char* dirpath, char* prefix);
extern void gpu_get_all_location_id(std::vector<unsigned short int>& gpus_location_id);


#endif	// _GPU_UTIL_H_