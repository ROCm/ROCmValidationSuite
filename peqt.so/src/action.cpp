// Copyright [year] <Copyright Owner> ... goes here
#include <stdlib.h>
#include <dirent.h>
#include <string>
#include <vector>
#include <regex>
#include <iostream>
#include <map>

#ifdef __cplusplus
extern "C" {
#endif
#include <pci/pci.h>
#include <linux/pci.h>
#ifdef __cplusplus
}
#endif

#include "pci_caps.h"
#include "gpu_util.h"
#include "rvs_util.h"
#include "rvsliblogger.h"
#include "rvs_module.h"
#include "action.h"

#define PCI_DEV_NUM_CAPABILITIES        13

#define YAML_DEVICE_PROPERTY_ERROR      "Error while parsing <device> property!"
#define YAML_DEVICE_PROP_DELIMITER      " "

#define RVS_CONF_NAME_KEY               "name"
#define RVS_CONF_DEVICE_KEY             "device"
#define RVS_CONF_DEVICEID_KEY           "deviceid"

const char* pcie_cap_names[] =
        { "link_cap_max_speed", "link_cap_max_width", "link_stat_cur_speed",
                "link_stat_neg_width", "slot_pwr_limit_value",
                "slot_physical_num", "device_id", "vendor_id", "kernel_driver",
                "dev_serial_num", "pwr_base_pwr", "pwr_rail_type",
                "atomic_op_completer" };

action::action() {
}

action::~action() {
    property.clear();
}

int action::property_set(const char* Key, const char* Val) {
    return rvs::lib::actionbase::property_set(Key, Val);
}

int action::run(void) {
    string prop_name, msg, action_name = "[]";
    char buff[1024];
    bool pci_infra_ok = true;  // PCI qualification result
    bool amd_gpus_found = false;
    bool device_property_ok = true;
    bool device_all_selected = false;
    bool device_id_filtering = false;
    u16 config_deviceid;

    // array of pointer to function corresponding to each capability
    void (*arr_prop_pfunc_names[])(struct pci_dev *dev, char *) = {
        get_link_cap_max_speed, get_link_cap_max_width,
        get_link_stat_cur_speed, get_link_stat_neg_width,
        get_slot_pwr_limit_value, get_slot_physical_num,
        get_device_id, get_vendor_id, get_kernel_driver,
        get_dev_serial_num, get_pwr_base_pwr,
        get_pwr_rail_type, get_atomic_op_completer
    };

    std::map<string, string>::iterator it;  // properties map iterator
    std::vector<u16> gpus_location_id;  // AMD GPUs location_id (PCI related)
    std::vector<u16> gpus_gpu_id;  // AMD GPUs gpu_id
    std::vector<string> device_prop_gpu_id_list;  // the list of all gpu_id
                                                // in the <device> property

    struct pci_access *pacc;
    struct pci_dev *dev;

    // get the action name
    it = property.find(RVS_CONF_NAME_KEY);
    if (it != property.end()) {
        action_name = it->second;
        property.erase(it);
    }

    // get gpus id
    it = property.find(RVS_CONF_DEVICE_KEY);
    if (it != property.end()) {
        if (it->second == "all") {
            device_all_selected = true;
        } else {
            // split the list of gpu_id
            device_prop_gpu_id_list = str_split(it->second,
            YAML_DEVICE_PROP_DELIMITER);
        }
        property.erase(it);
    } else {
        // no device (individual gpu_id or all) selected
        msg = action_name + " peqt " + YAML_DEVICE_PROPERTY_ERROR;
        log(msg.c_str(), rvs::logerror);
        return 0;  // TODO(Tudor) check what to return 0 or -1?
    }

    // get the deviceid
    it = property.find(RVS_CONF_DEVICEID_KEY);
    if (it != property.end()) {
        if(it->second != "") {
            config_deviceid = std::stoi(it->second);
            device_id_filtering = true;
        }
        property.erase(it);
    }

    // get all gpu id
    gpu_get_all_gpu_id(gpus_gpu_id);

    // get all GPU location_id (Note: we're not using device_id as the
    // unique identifier of the GPU because multiple GPUs can have the same ID
    // this is also true for the case of the machine where we're working)
    // therefore, what we're using is the location_id which is unique and
    // points to the sysfs
    // this list is "synced" with gpus_gpu_id
    gpu_get_all_location_id(gpus_location_id);

    // get the pci_access structure
    pacc = pci_alloc();
    // initialize the PCI library
    pci_init(pacc);
    // get the list of devices
    pci_scan_bus(pacc);

    // iterate over devices
    for (dev = pacc->devices; dev; dev = dev->next) {
        // fill in the info
        pci_fill_info(dev,
                PCI_FILL_IDENT | PCI_FILL_BASES |
                PCI_FILL_CLASS | PCI_FILL_EXT_CAPS |
                PCI_FILL_CAPS | PCI_FILL_PHYS_SLOT);

        // computes the actual dev's location_id (sysfs entry)
        u16 dev_location_id = ((((u16) (dev->bus)) << 8) | (dev->func));

        // check if this pci_dev corresponds to one of AMD GPUs
        std::vector<u16>::iterator it_gpu = find(gpus_location_id.begin(),
                gpus_location_id.end(), dev_location_id);

        if (it_gpu != gpus_location_id.end()) {
            // that should be an AMD GPU
            amd_gpus_found = true;

            // check for deviceid filtering
            if(!device_id_filtering || (device_id_filtering && dev->device_id == config_deviceid)) {

                // check if the GPU is part of the PCIe check
                // (either device: all or the gpu_id is the device: <gpu id> list

                bool cur_gpu_selected = false;
                if (device_all_selected) {
                    cur_gpu_selected = true;
                } else {
                    // search for this gpu in the list provided under the
                    // <device> property get the actual position in
                    // the location_id list
                    u16 index_loc_id = std::distance(gpus_location_id.begin(),
                            it_gpu);
                    // get the gpu_id for the same position
                    u16 gpu_id = gpus_gpu_id.at(index_loc_id);

                    // check if this gpu_id is in the list provided
                    // within the <device> property
                    std::vector<string>::iterator it_gpu_id = find(
                            device_prop_gpu_id_list.begin(),
                            device_prop_gpu_id_list.end(), std::to_string(gpu_id));

                    if (it_gpu_id != device_prop_gpu_id_list.end())
                        cur_gpu_selected = true;
                }

                if (cur_gpu_selected) {
                    for (it = property.begin(); it != property.end(); ++it) {
                        // skip the "capability."
                        string prop_name = it->first.substr(
                                it->first.find_last_of(".") + 1);
                        for (unsigned char i = 0; i < PCI_DEV_NUM_CAPABILITIES; i++)
                            if (prop_name == pcie_cap_names[i]) {
                                (*arr_prop_pfunc_names[i])(dev, buff);
                                msg = action_name + " peqt " + pcie_cap_names[i]
                                        + " " + buff;
                                log(msg.c_str(), rvs::loginfo);
                                if (it->second != "") {
                                    regex prop_regex(it->second);
                                    if (!regex_match(buff, prop_regex)) {
                                        pci_infra_ok = false;
                                    }
                                }
                            }
                    }
                }
            }
        }
    }

    if (!amd_gpus_found)
        pci_infra_ok = false;

    msg = action_name + " peqt " + (pci_infra_ok ? "TRUE" : "FALSE");
    log(msg.c_str(), rvs::logresults);

    pci_cleanup(pacc);

    return 0;
}
