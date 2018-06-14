// Copyright [year] <Copyright Owner> ... goes here
#include <string>
#include <vector>
#include <regex>
#include <map>

#ifdef __cplusplus
extern "C" {
#endif
#include <pci/pci.h>
#ifdef __cplusplus
}
#endif

#include "pci_caps.h"
#include "gpu_util.h"
#include "rvs_util.h"
#include "rvsliblogger.h"
#include "rvs_module.h"
#include "action.h"

#define CHAR_BUFF_MAX_SIZE              1024
#define PCI_DEV_NUM_CAPABILITIES        13

#define YAML_DEVICE_PROPERTY_ERROR      "Error while parsing <device> property!"
#define YAML_DEVICEID_PROPERTY_ERROR    "Error while parsing <deviceid> property!"
#define YAML_REGULAR_EXPRESSION_ERROR   "Regular expression error"
#define KFD_QUERYING_ERROR              "An error occurred while querying the GPU properties"
#define PCI_ALLOC_ERROR                 "pci_alloc() error"
#define YAML_DEVICE_PROP_DELIMITER      " "

#define RVS_CONF_NAME_KEY               "name"
#define RVS_CONF_DEVICE_KEY             "device"
#define RVS_CONF_DEVICEID_KEY           "deviceid"

#define MODULE_NAME                     "peqt"

// collection of allowd PCIe capabilities
const char* pcie_cap_names[] =
        { "link_cap_max_speed", "link_cap_max_width", "link_stat_cur_speed",
                "link_stat_neg_width", "slot_pwr_limit_value",
                "slot_physical_num", "device_id", "vendor_id", "kernel_driver",
                "dev_serial_num", "pwr_base_pwr", "pwr_rail_type",
                "atomic_op_completer" };

// array of pointer to function corresponding to each capability
void (*arr_prop_pfunc_names[])(struct pci_dev *dev, char *) = {
    get_link_cap_max_speed, get_link_cap_max_width,
    get_link_stat_cur_speed, get_link_stat_neg_width,
    get_slot_pwr_limit_value, get_slot_physical_num,
    get_device_id, get_vendor_id, get_kernel_driver,
    get_dev_serial_num, get_pwr_base_pwr,
    get_pwr_rail_type, get_atomic_op_completer
};

using std::vector;
using std::string;
using std::map;

/**
 * default class constructor
 */
action::action() {
}

/**
 * class destructor
 */
action::~action() {
    property.clear();
}

/**
 * checks if input string is a positive integer number
 * @param str_val the input string
 * @return true if string is a positive integer number, false otherwise
 */
static bool is_positive_integer(const std::string& str_val) {
    return !str_val.empty()
            && std::find_if(str_val.begin(), str_val.end(),
                    [](char c) {return !std::isdigit(c);}) == str_val.end();
}

/**
 * adds a (key, value) pair to the module's properties collection
 * @param Key one of the keys specified in the RVS SRS
 * @param Val key's value
 * @return add result
 */
int action::property_set(const char* Key, const char* Val) {
    return rvs::lib::actionbase::property_set(Key, Val);
}

/**
 * gets the gpu_id list from the module's properties collection
 * @param error pointer to a memory location where the error code will be stored
 * @return true if "all" is selected, false otherwise
 */
bool action::property_get_device(int *error) {
    map<string, string>::iterator it;  // module's properties map iterator
    *error = 0;  // init with 'no error'
    it = property.find(RVS_CONF_DEVICE_KEY);
    if (it != property.end()) {
        if (it->second == "all") {
            property.erase(it);
            return true;
        } else {
            // split the list of gpu_id
            device_prop_gpu_id_list = str_split(it->second,
            YAML_DEVICE_PROP_DELIMITER);
            property.erase(it);

            if (device_prop_gpu_id_list.empty()) {
                *error = 1;  // list of gpu_id cannot be empty
            } else {
                for (vector<string>::iterator it_gpu_id =
                        device_prop_gpu_id_list.begin();
                        it_gpu_id != device_prop_gpu_id_list.end(); ++it_gpu_id)
                    if (!is_positive_integer(*it_gpu_id)) {
                        *error = 1;
                        break;
                    }
            }
            return false;
        }

    } else {
        *error = 1;
        return false;  // when error is set, it doesn't really matter whether the method returns true or false
    }
}

/**
 * gets all PCIe capabilities for a given gpu
 * @param dev pointer to pci_dev corresponding to the current GPU
 * @return false if regex check failed, true otherwise
 */
bool action::get_gpu_all_pcie_capabilities(struct pci_dev *dev) {
    char buff[CHAR_BUFF_MAX_SIZE];
    string prop_name, msg;
    bool pci_infra_qual_result = true;
    map<string, string>::iterator it;  // module's properties map iterator

    for (it = property.begin(); it != property.end(); ++it) {
        // skip the "capability."
        string prop_name = it->first.substr(it->first.find_last_of(".") + 1);
        for (unsigned char i = 0; i < PCI_DEV_NUM_CAPABILITIES; i++)
            if (prop_name == pcie_cap_names[i]) {
                // call the capability's corresponding function
                (*arr_prop_pfunc_names[i])(dev, buff);

                // log the capability's value
                msg = action_name + " " + MODULE_NAME + " " + pcie_cap_names[i] + " " + buff;
                log(msg.c_str(), rvs::loginfo);

                // check for regex match
                if (it->second != "") {
                    try {
                        regex prop_regex(it->second);
                        if (!regex_match(buff, prop_regex)) {
                            pci_infra_qual_result = false;
                        }
                    } catch (const std::regex_error& e) {
                        // log the regex error
                        msg = action_name + " " + MODULE_NAME
                                + " "
                                + YAML_REGULAR_EXPRESSION_ERROR
                                + " at '" + it->second + "'";
                        log(msg.c_str(), rvs::logerror);
                    }
                }
            }
    }

    return pci_infra_qual_result;
}

/**
 * gets the action name from the module's properties collection
 */
void action::property_get_action_name(void)
{
    action_name = "[]";
    map<string, string>::iterator it = property.find(RVS_CONF_NAME_KEY);
    if (it != property.end()) {
        action_name = it->second;
        property.erase(it);
    }
}

/**
 * gets the deviceid from the module's properties collection
 * @param error pointer to a memory location where the error code will be stored
 * @return deviceid value if valid, -1 otherwise
 */
int action::property_get_deviceid(int *error) {
    map<string, string>::iterator it = property.find(RVS_CONF_DEVICEID_KEY);
    int deviceid = -1;
    *error = 0;  // init with 'no error'

    if (it != property.end()) {
        if (it->second != "") {
            if(is_positive_integer(it->second)) {
                deviceid = std::stoi(it->second);
            } else {
                *error = 1;  // we have something but it's not a number
            }
        } else {
            *error = 1;  // we have an empty string
        }
        property.erase(it);
    }

    return deviceid;
}

/**
 * runs the whole PEQT logic
 * @return run result
 */
int action::run(void) {
    string msg;
    map<string, string>::iterator it;  // module's properties map iterator
    bool pci_infra_qual_result = true;  // PCI qualification result
    bool amd_gpus_found = false;
    bool device_all_selected = false;
    bool device_id_filtering = false;
    int error = 0;
    unsigned short int deviceid;
    vector<unsigned short int> gpus_location_id;  // stores all the AMD GPU location_id
    vector<unsigned short int> gpus_id;  // stores all the AMD GPUs gpu_id

    struct pci_access *pacc;
    struct pci_dev *dev;

    // get the action name
    property_get_action_name();

    // get <device> property value (a list of gpu id)
    device_all_selected = property_get_device(&error);
    if (error) {
        // log the error
        msg = action_name + " " + MODULE_NAME + " " + YAML_DEVICE_PROPERTY_ERROR;
        log(msg.c_str(), rvs::logerror);

        // log the module's result (FALSE) and abort PCIe qualification check (this parameter is mandatory)
        msg = action_name + " " + MODULE_NAME + " FALSE";
        log(msg.c_str(), rvs::logresults);

        return 1;  // PCIe qualification check cannot continue
    }

    // get the <deviceid> property value
    int devid = property_get_deviceid(&error);
    if (!error) {
        if (devid != -1) {
            deviceid = static_cast<unsigned short int>(devid);
            device_id_filtering = true;
        }
    } else {
        // log the error
        msg = action_name + " " + MODULE_NAME + " " + YAML_DEVICEID_PROPERTY_ERROR;
        log(msg.c_str(), rvs::logerror);

        // continue with the PCIe qualification check (this parameter is optional)
    }

    // get all gpu_id for all AMD compatible GPUs that are registered into the system via kfd query
    gpu_get_all_gpu_id(gpus_id);

    if (gpus_id.empty()) {
        // no need to query the PCI bus as there is no AMD compatible GPU
        msg = action_name + " " + MODULE_NAME + " FALSE";
        log(msg.c_str(), rvs::logresults);
        return 0;
    }

    // get all GPU location_id (all values are unique and point to the sysfs)
    // this list is "synced" (in regards to the elements position) with gpus_id
    gpu_get_all_location_id(gpus_location_id);

    if (gpus_location_id.empty()) {
        // basically, if we got to this point then the gpus_location_id
        // list cannot be empty unless there was some kind of an error
        // while querying the kfd

        // log the error
        msg = action_name + " " + MODULE_NAME + " " + KFD_QUERYING_ERROR;
        log(msg.c_str(), rvs::logerror);

        // log the module's result (FALSE)
        msg = action_name + " " + MODULE_NAME + " FALSE";
        log(msg.c_str(), rvs::logresults);
        return 1;  // PCIe qualification check cannot continue
    }

    // get the pci_access structure
    pacc = pci_alloc();

    if (pacc == NULL) {
        // log the error
        msg = action_name + " " + MODULE_NAME + " " + PCI_ALLOC_ERROR;
        log(msg.c_str(), rvs::logerror);

        // log the module's result (FALSE)
        msg = action_name + " " + MODULE_NAME + " FALSE";
        log(msg.c_str(), rvs::logresults);
        return 1;  // PCIe qualification check cannot continue
    }

    // initialize the PCI library
    pci_init(pacc);
    // get the list of devices
    pci_scan_bus(pacc);

    // iterate over devices
    for (dev = pacc->devices; dev; dev = dev->next) {
        // fill in the info
        pci_fill_info(dev, PCI_FILL_IDENT | PCI_FILL_BASES | PCI_FILL_CLASS | PCI_FILL_EXT_CAPS | PCI_FILL_CAPS | PCI_FILL_PHYS_SLOT);

        // computes the actual dev's location_id (sysfs entry)
        unsigned short int dev_location_id = ((((unsigned short int) (dev->bus)) << 8) | (dev->func));

        // check if this pci_dev corresponds to one of the AMD GPUs
        vector<unsigned short int>::iterator it_gpu = find( gpus_location_id.begin(), gpus_location_id.end(), dev_location_id);

        if (it_gpu != gpus_location_id.end()) {
            // that should be an AMD GPU

            // check for deviceid filtering
            if (!device_id_filtering || (device_id_filtering && dev->device_id == deviceid)) {
                // check if the GPU is part of the PCIe check
                // (either device: all or the gpu_id is the device: <gpu id> list

                bool cur_gpu_selected = false;

                if (device_all_selected) {
                    cur_gpu_selected = true;
                } else {
                    // search for this gpu in the list provided under the <device> property

                    // get the actual position in the location_id list
                    unsigned short int index_loc_id = std::distance(gpus_location_id.begin(), it_gpu);
                    // get the gpu_id for the same position
                    unsigned short int gpu_id = gpus_id.at(index_loc_id);

                    // check if this gpu_id is in the list provided within the <device> property
                    vector<string>::iterator it_gpu_id = find(device_prop_gpu_id_list.begin(), device_prop_gpu_id_list.end(), std::to_string(gpu_id));

                    if (it_gpu_id != device_prop_gpu_id_list.end())
                        cur_gpu_selected = true;
                }

                if (cur_gpu_selected) {
                    amd_gpus_found = true;
                    if(!get_gpu_all_pcie_capabilities(dev))
                        pci_infra_qual_result = false;
                }
            }
        }
    }

    pci_cleanup(pacc);

    if (!amd_gpus_found)
        pci_infra_qual_result = false;

    msg = action_name + " " + MODULE_NAME + " " + (pci_infra_qual_result ? "TRUE" : "FALSE");
    log(msg.c_str(), rvs::logresults);

    return 0;
}
