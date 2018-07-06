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
#include "action.h"

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
#include "rvs_module.h"
#include "rvsloglp.h"

#define CHAR_BUFF_MAX_SIZE              1024
#define PCI_DEV_NUM_CAPABILITIES        16
#define PCI_ALLOC_ERROR                 "pci_alloc() error"

#define JSON_CAPS_NODE_NAME             "capabilities"
#define JSON_CREATE_NODE_ERROR          "JSON cannot create node"

#define PEQT_RESULT_PASS_MESSAGE        "TRUE"
#define PEQT_RESULT_FAIL_MESSAGE        "FALSE"

#define MODULE_NAME                     "peqt"

using namespace std;

// collection of allowed PCIe capabilities
const char* pcie_cap_names[] =
        { "link_cap_max_speed", "link_cap_max_width", "link_stat_cur_speed",
                "link_stat_neg_width", "slot_pwr_limit_value",
                "slot_physical_num", "device_id", "vendor_id", "kernel_driver",
                "dev_serial_num", "pwr_base_pwr", "pwr_rail_type",
                "atomic_op_requester",  "atomic_32_bit_op_completer",
		"atomic_64_bit_op_completer",
		"atomic_128_bit_cas_op_completer"
        };

// array of pointer to function corresponding to each capability
void (*arr_prop_pfunc_names[])(struct pci_dev *dev, char *) = {
    get_link_cap_max_speed, get_link_cap_max_width,
    get_link_stat_cur_speed, get_link_stat_neg_width,
    get_slot_pwr_limit_value, get_slot_physical_num,
    get_device_id, get_vendor_id, get_kernel_driver,
    get_dev_serial_num, get_pwr_base_pwr,
    get_pwr_rail_type, get_atomic_op_requester,
	get_atomic_32_bit_op_completer, get_atomic_64_bit_op_completer,
	get_atomic_128_bit_cas_op_completer
};

using std::vector;
using std::string;
using std::map;

/**
 * default class constructor
 */
action::action() {
    bjson = false;
    json_root_node = NULL;
}

/**
 * class destructor
 */
action::~action() {
    property.clear();
}


/**
 * gets all PCIe capabilities for a given AMD compatible GPU and
 * checks the values against the given set of regular expressions
 * @param dev pointer to pci_dev corresponding to the current GPU
 * @param gpu_id unique gpu id
 * @return false if regex check failed, true otherwise
 */
bool action::get_gpu_all_pcie_capabilities(struct pci_dev *dev,
        uint16_t gpu_id) {
    char buff[CHAR_BUFF_MAX_SIZE];
    string prop_name, msg;
    bool pci_infra_qual_result = true;
    map<string, string>::iterator it;  // module's properties map iterator
    void *json_pcaps_node = NULL;

    if (bjson) {
        if (json_root_node != NULL) {
            json_pcaps_node = rvs::lp::CreateNode(json_root_node,
                    JSON_CAPS_NODE_NAME);
            if (json_pcaps_node == NULL) {
                // log the error
                msg = action_name + " " + MODULE_NAME + " "
                        + JSON_CREATE_NODE_ERROR;
                log(msg.c_str(), rvs::logerror);
            }
        }
    }

    if (bjson && json_pcaps_node != NULL) {
        rvs::lp::AddString(json_pcaps_node, RVS_JSON_LOG_GPU_ID_KEY,
                std::to_string(gpu_id));
    }

    for (it = property.begin(); it != property.end(); ++it) {
        // skip the "capability."
        string prop_name = it->first.substr(it->first.find_last_of(".") + 1);
        for (unsigned char i = 0; i < PCI_DEV_NUM_CAPABILITIES; i++)
            if (prop_name == pcie_cap_names[i]) {
                // call the capability's corresponding function
                (*arr_prop_pfunc_names[i])(dev, buff);

                // log the capability's value
                msg = action_name + " " + MODULE_NAME + " " + pcie_cap_names[i]
                        + " " + buff;
                log(msg.c_str(), rvs::loginfo);

                if (bjson && json_pcaps_node != NULL) {
                    rvs::lp::AddString(json_pcaps_node, pcie_cap_names[i],
                            buff);
                }

                // check for regex match
                if (it->second != "") {
                    try {
                        regex prop_regex(it->second);
                        if (!regex_match(buff, prop_regex)) {
                            pci_infra_qual_result = false;
                        }
                    } catch (const std::regex_error& e) {
                        // log the regex error
                        msg = action_name + " " + MODULE_NAME + " "
                                + YAML_REGULAR_EXPRESSION_ERROR + " at '"
                                + it->second + "'";
                        log(msg.c_str(), rvs::logerror);
                    }
                }
            }
    }

    if (bjson && json_pcaps_node != NULL)
        rvs::lp::AddNode(json_root_node, json_pcaps_node);

    return pci_infra_qual_result;
}

/**
 * gets the action name from the module's properties collection
 */
void action::property_get_action_name(void) {
    action_name = "[]";
    map<string, string>::iterator it = property.find(RVS_CONF_NAME_KEY);
    if (it != property.end()) {
        action_name = it->second;
        property.erase(it);
    }
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
    uint16_t deviceid;
    vector<uint16_t> gpus_location_id;
    vector<uint16_t> gpus_id;

    struct pci_access *pacc;
    struct pci_dev *dev;

    // get the action name
    property_get_action_name();

    bjson = false;  // already initialized in the default constructor

    // check for -j flag (json logging)
    if (property.find("cli.-j") != property.end()) {
        unsigned int sec;
        unsigned int usec;
        rvs::lp::get_ticks(sec, usec);

        bjson = true;

        json_root_node = rvs::lp::LogRecordCreate(MODULE_NAME,
                action_name.c_str(), rvs::loginfo, sec, usec);
        if (json_root_node == NULL) {
            // log the error
            msg =
                    action_name + " " + MODULE_NAME + " "
                            + JSON_CREATE_NODE_ERROR;
            log(msg.c_str(), rvs::logerror);
        }
    }

    // get <device> property value (a list of gpu id)
    device_all_selected = property_get_device(&error);
    if (error) {
        // log the error
        msg =
                action_name + " " + MODULE_NAME + " "
                        + YAML_DEVICE_PROPERTY_ERROR;
        log(msg.c_str(), rvs::logerror);

        // log the module's result (FALSE) and abort PCIe qualification check
        // (<device> parameter is mandatory)
        msg = action_name + " " + MODULE_NAME + " " + PEQT_RESULT_FAIL_MESSAGE;
        log(msg.c_str(), rvs::logresults);

        return 1;  // PCIe qualification check cannot continue
    }

    // get the <deviceid> property value
    int devid = property_get_deviceid(&error);
    if (!error) {
        if (devid != -1) {
            deviceid = static_cast<uint16_t>(devid);
            device_id_filtering = true;
        }
    } else {
        // log the error
        msg = action_name + " " + MODULE_NAME + " "
                + YAML_DEVICEID_PROPERTY_ERROR;
        log(msg.c_str(), rvs::logerror);

        // continue with PCIe qualification check (this parameter is optional)
    }

    // get all gpu_id for all AMD compatible GPUs that are registered
    // into the system, via kfd querying
    gpu_get_all_gpu_id(gpus_id);

    if (gpus_id.empty()) {
        // no need to query the PCI bus as there is no AMD compatible GPU
        msg = action_name + " " + MODULE_NAME + " " + PEQT_RESULT_FAIL_MESSAGE;
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
        msg = action_name + " " + MODULE_NAME + " " + PEQT_RESULT_FAIL_MESSAGE;
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
        msg = action_name + " " + MODULE_NAME + " " + PEQT_RESULT_FAIL_MESSAGE;
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
        pci_fill_info(dev,
                PCI_FILL_IDENT | PCI_FILL_BASES | PCI_FILL_CLASS
                | PCI_FILL_EXT_CAPS | PCI_FILL_CAPS | PCI_FILL_PHYS_SLOT);

        // computes the actual dev's location_id (sysfs entry)
        uint16_t dev_location_id = ((((uint16_t) (dev->bus)) << 8)
                | (dev->func));

        // check if this pci_dev corresponds to one of the AMD GPUs
        vector<uint16_t>::iterator it_gpu = find(
                gpus_location_id.begin(), gpus_location_id.end(),
                dev_location_id);

        if (it_gpu != gpus_location_id.end()) {
            // that should be an AMD GPU

            // check for deviceid filtering
            if (!device_id_filtering
                    || (device_id_filtering && dev->device_id == deviceid)) {
                // check if the GPU is part of the PCIe check
                // (either device: all or the gpu_id is in
                // the device: <gpu id> list

                bool cur_gpu_selected = false;

                // get the actual position in the location_id list
                uint16_t index_loc_id = std::distance(
                        gpus_location_id.begin(), it_gpu);
                // get the gpu_id for the same position
                uint16_t gpu_id = gpus_id.at(index_loc_id);
                if (device_all_selected) {
                    cur_gpu_selected = true;
                } else {
                    // search for this gpu in the list
                    // provided under the <device> property
                    vector<string>::iterator it_gpu_id = find(
                            device_prop_gpu_id_list.begin(),
                            device_prop_gpu_id_list.end(),
                            std::to_string(gpu_id));

                    if (it_gpu_id != device_prop_gpu_id_list.end())
                        cur_gpu_selected = true;
                }

                if (cur_gpu_selected) {
                    amd_gpus_found = true;
                    if (!get_gpu_all_pcie_capabilities(dev, gpu_id))
                        pci_infra_qual_result = false;
                }
            }
        }
    }

    pci_cleanup(pacc);

    if (!amd_gpus_found)
        pci_infra_qual_result = false;

    msg = action_name + " " + MODULE_NAME + " "
            + (pci_infra_qual_result ?
                    PEQT_RESULT_PASS_MESSAGE : PEQT_RESULT_FAIL_MESSAGE);
    log(msg.c_str(), rvs::logresults);

    if (bjson && json_root_node != NULL) {
        rvs::lp::AddString(json_root_node, "RESULT",
                (pci_infra_qual_result ?
                        PEQT_RESULT_PASS_MESSAGE : PEQT_RESULT_FAIL_MESSAGE));
        rvs::lp::LogRecordFlush(json_root_node);
    }

    return 0;
}