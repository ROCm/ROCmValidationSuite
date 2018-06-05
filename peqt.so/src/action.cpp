#include <stdlib.h>
#include <dirent.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <map>
#include <vector>
#include <algorithm>
#include "rvsliblogger.h"
#include "rvs_module.h"
#include "action.h"

#ifdef __cplusplus
extern "C" {
#endif
#include <pci/pci.h>
#include <linux/pci.h>
#ifdef __cplusplus
}
#endif


#define KFD_SYS_PATH_NODES              "/sys/class/kfd/kfd/topology/nodes"
#define KFD_PATH_MAX_LENGTH             256
#define PCI_DEV_NUM_CAPABILITIES        10

using namespace std;

const char* pcie_cap_names[] = {"link_cap_max_speed", "link_cap_max_width", "link_stat_cur_speed", "link_stat_neg_width", "slot_pwr_limit_value", "slot_physical_num", "device_id", "vendor_id", "kernel_driver", "dev_serial_num"};

static int num_subdirs(char *dirpath, char *prefix);
static void get_all_gpu_location_id(std::vector<unsigned short int> &gpus_location_id);

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


/**
 * gets all GPUS location_id
 * @param gpus_location_id the vector that will store all the GPU location_id
 * @return
 */
static void get_all_gpu_location_id(std::vector<unsigned short int> &gpus_location_id)
{
	ifstream f_id,f_prop;
	char path[KFD_PATH_MAX_LENGTH];

    string prop_name;
    int gpu_id;
    unsigned short int prop_val;


    //Discover the number of nodes: Inside nodes folder there are only folders that represent the node number
    int num_nodes = num_subdirs((char*)KFD_SYS_PATH_NODES, (char*)"");

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
    string prop_name, msg, action_name = "[]";
    char buff[1024];

    void (*arr_prop_pfunc_names[]) (struct pci_dev *dev, char *) = {get_link_cap_max_speed, get_link_cap_max_width, get_link_stat_cur_speed, get_link_stat_neg_width, get_slot_pwr_limit_value, get_slot_physical_num, get_device_id, get_vendor_id, get_kernel_driver, get_dev_serial_num};

	std::map<string,string>::iterator it;
 	std::vector<unsigned short int> gpus_location_id;

    struct pci_access *pacc;
    struct pci_dev *dev;


    //get the action name
    it=property.find("name");
    if ( it != property.end() ){
        action_name = it->second;
        property.erase(it);
    }

    //get all GPU location_id (Note: we're not using device_id as the unique identifier of the GPU because multiple GPUs can have the same ID ... this is also true for the case of the machine where we're working)
    //therefore, what we're using is the location_id which is unique and points to the sysfs
    get_all_gpu_location_id(gpus_location_id);

    //get the pci_access structure
    pacc = pci_alloc();
    //initialize the PCI library
    pci_init(pacc);
    //get the list of devices
    pci_scan_bus(pacc);

    //iterate over devices
    for (dev = pacc->devices; dev; dev = dev->next){
        pci_fill_info(dev, PCI_FILL_IDENT | PCI_FILL_BASES | PCI_FILL_CLASS | PCI_FILL_EXT_CAPS | PCI_FILL_CAPS | PCI_FILL_PHYS_SLOT); //fil in the info

        //computes the actual dev's location_id (sysfs entry)
        unsigned short int dev_location_id = ((((unsigned short int)(dev->bus)) << 8) | (dev->func));

        //check if this pci_dev corresponds to one of AMD GPUs
        std::vector<unsigned short int>::iterator it_gpu = find(gpus_location_id.begin(), gpus_location_id.end(), dev_location_id);

        if(it_gpu != gpus_location_id.end()){
            //that should be an AMD GPU

            for (it=property.begin(); it!=property.end(); ++it){
                string prop_name = it->first.substr(it->first.find_last_of(".") + 1); //skip the "capability."                
                for(unsigned char i = 0; i < PCI_DEV_NUM_CAPABILITIES; i++)
                    if(prop_name == pcie_cap_names[i]){
                        (*arr_prop_pfunc_names[i])(dev, buff);
                        msg = action_name + " peqt " + pcie_cap_names[i] + " " +  buff;
                        log( msg.c_str(), rvs::loginfo);
                    }
            }
        }
    }

    pci_cleanup(pacc);

    return 0;
}
