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
#include <utility>
#include <iostream>

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
#define PCI_DEV_NUM_CAPABILITIES        14
#define PCI_ALLOC_ERROR                 "pci_alloc() error"

#define JSON_CAPS_NODE_NAME             "capabilities"
#define JSON_CREATE_NODE_ERROR          "JSON cannot create node"

#define PEQT_RESULT_PASS_MESSAGE        "TRUE"
#define PEQT_RESULT_FAIL_MESSAGE        "FALSE"

#define MODULE_NAME                     "peqt"

#define YAML_CAPABILITY_TAG             "capability"
#define PB_OP_COND_DYN_DELIMITER        "_"

#define PB_NUM_OP_STATES                4
#define PB_NUM_OP_TYPES                 5
#define PN_NUM_OP_POWER_RAILS           4

using std::string;
using std::regex;
using std::vector;
using std::map;
using std::cerr;

// collection of allowed PCIe capabilities
const char* pcie_cap_names[] =
        {   "link_cap_max_speed", "link_cap_max_width", "link_stat_cur_speed",
            "link_stat_neg_width", "slot_pwr_limit_value",
            "slot_physical_num", "device_id", "vendor_id", "kernel_driver",
            "dev_serial_num", "atomic_op_routing",  "atomic_op_32_completer",
            "atomic_op_64_completer", "atomic_op_128_CAS_completer"
        };

// array of pointer to function corresponding to each capability
void (*arr_prop_pfunc_names[])(struct pci_dev *dev, char *) = {
    get_link_cap_max_speed, get_link_cap_max_width,
    get_link_stat_cur_speed, get_link_stat_neg_width,
    get_slot_pwr_limit_value, get_slot_physical_num,
    get_device_id, get_vendor_id, get_kernel_driver,
    get_dev_serial_num, get_atomic_op_routing,
    get_atomic_op_32_completer, get_atomic_op_64_completer,
    get_atomic_op_128_CAS_completer
};

const char * pb_op_pm_states_list[] = {"D0", "D1", "D2", "D3"};
const char * pb_op_types_list[] = {"PMEAux", "Auxiliary", "Idle",
                                    "Sustained", "Maximum"};
const char * pb_op_power_rails_list[] = {"Power_12V", "Power_3_3V",
                                        "Power_1_5V_1_8V", "Thermal"};


const uint8_t pb_op_pm_states_encoding[] = {0, 1, 2, 3};
const uint8_t pb_op_types_encoding[] = {0, 1, 2, 3, 7};
const uint8_t pb_op_power_rails_encoding[] = {0, 1, 2, 7};

/**
 * @brief default class constructor
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
 * @brief gets all PCIe capabilities for a given AMD compatible GPU and
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
    uint8_t i;

    if (bjson) {
        if (json_root_node != NULL) {
            json_pcaps_node = rvs::lp::CreateNode(json_root_node,
                    JSON_CAPS_NODE_NAME);
            if (json_pcaps_node == NULL) {
                // log the error
                msg = action_name + " " + MODULE_NAME + " "
                        + JSON_CREATE_NODE_ERROR;
                cerr << "RVS-PEQT: " << msg;
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
        bool prop_found = false;
        for (i = 0; i < PCI_DEV_NUM_CAPABILITIES; i++) {
            if (prop_name == pcie_cap_names[i]) {
                prop_found = true;
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
                        cerr << "RVS-PEQT: " << msg;
                    }
                }
                break;
            }
        }

        if (!prop_found &&
                it->first.find(YAML_CAPABILITY_TAG) != string::npos) {
            // the property was not found among those that
            // have fixed/constant name => check whether it's
            // a dynamic Power Budgeting capability
            if (regex_match(prop_name, pb_dynamic_regex)) {
                // no additional checks are needed (itetator != .end() && npos)
                // because the prop_name already matched the regular expression

                std::size_t pos_pb_pm_state =
                            prop_name.find_first_of(PB_OP_COND_DYN_DELIMITER);
                map<string, uint8_t>::iterator it_pb_pm_state =
                                pb_op_pm_states_encodings_map.find
                                    (prop_name.substr(0, pos_pb_pm_state));
                uint8_t pb_op_pm_state = it_pb_pm_state->second;

                std::size_t pos_pb_type =
                            prop_name.find(PB_OP_COND_DYN_DELIMITER,
                                                    pos_pb_pm_state + 1);
                map<string, uint8_t>::iterator it_pb_type =
                                pb_op_pm_types_encodings_map.find
                                    (prop_name.substr(pos_pb_pm_state + 1,
                                        pos_pb_type - pos_pb_pm_state - 1));
                uint8_t pb_op_pm_type = it_pb_type->second;

                map<string, uint8_t>::iterator it_pb_power_rail =
                                pb_op_pm_power_rails_encodings_map.find
                                        (prop_name.substr(pos_pb_type + 1));
                uint8_t pb_op_power_rail = it_pb_power_rail->second;
                // query for power budgeting capabilities
                get_pwr_budgeting(dev, pb_op_pm_state, pb_op_pm_type,
                                                    pb_op_power_rail, buff);

                // log the capability's value
                msg = action_name + " " + MODULE_NAME + " " + prop_name
                        + " " + buff;
                log(msg.c_str(), rvs::loginfo);

                if (bjson && json_pcaps_node != NULL) {
                    rvs::lp::AddString(json_pcaps_node, prop_name,
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
                        cerr << "RVS-PEQT: " << msg;
                    }
                }
            }
        }
    }

    if (bjson && json_pcaps_node != NULL)
        rvs::lp::AddNode(json_root_node, json_pcaps_node);

    return pci_infra_qual_result;
}


/**
 * @brief runs the whole PEQT logic
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
    uint8_t i;

    struct pci_access *pacc;
    struct pci_dev *dev;

    // get the action name
    rvs::actionbase::property_get_action_name(&error);
    if (error == 2) {
      msg = "action field is missing in gst module";
      cerr << "RVS-PEQT: " << msg;
      return -1;
    }

    bjson = false;  // already initialized in the default constructor

    // check for -j flag (json logging)
    if (property.find("cli.-j") != property.end()) {
        unsigned int sec;
        unsigned int usec;
        rvs::lp::get_ticks(&sec, &usec);

        bjson = true;

        json_root_node = rvs::lp::LogRecordCreate(MODULE_NAME,
                action_name.c_str(), rvs::loginfo, sec, usec);
        if (json_root_node == NULL) {
            // log the error
            msg =
                    action_name + " " + MODULE_NAME + " "
                            + JSON_CREATE_NODE_ERROR;
            cerr << "RVS-PEQT: " << msg;
        }
    }

    // get <device> property value (a list of gpu id)
    device_all_selected = property_get_device(&error);
    if (error) {
        // log the error
        msg =
                action_name + " " + MODULE_NAME + " "
                        + YAML_DEVICE_PROPERTY_ERROR;
        cerr << "RVS-PEQT: " << msg;

        // log the module's result (FALSE) and abort PCIe qualification check
        // (<device> parameter is mandatory)
        msg = action_name + " " + MODULE_NAME + " " + PEQT_RESULT_FAIL_MESSAGE;
        cerr << "RVS-PEQT: " << msg;

        return -1;  // PCIe qualification check cannot continue
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
        cerr << "RVS-PEQT: " << msg;

        // log the module's result (FALSE) and abort PCIe qualification check
        // (<device> parameter is mandatory)
        msg = action_name + " " + MODULE_NAME + " " + PEQT_RESULT_FAIL_MESSAGE;
        cerr << "RVS-PEQT: " << msg;

        return -1;  // PCIe qualification check cannot continue
    }

    // get the pci_access structure
    pacc = pci_alloc();

    if (pacc == NULL) {
        // log the error
        msg = action_name + " " + MODULE_NAME + " " + PCI_ALLOC_ERROR;
        cerr << "RVS-PEQT: " << msg;

        // log the module's result (FALSE)
        msg = action_name + " " + MODULE_NAME + " " + PEQT_RESULT_FAIL_MESSAGE;
        cerr << "RVS-PEQT: " << msg;
        return 1;  // PCIe qualification check cannot continue
    }

    // compose Power Budgeting dynamic regex
    string dyn_pb_regex_str = "^(";
    for (i = 0; i < PB_NUM_OP_STATES; i++) {
        dyn_pb_regex_str += pb_op_pm_states_list[i];
        pb_op_pm_states_encodings_map.insert(std::pair<string, uint8_t>
                    (pb_op_pm_states_list[i], pb_op_pm_states_encoding[i]));
        if (i < PB_NUM_OP_STATES - 1)
            dyn_pb_regex_str += "|";
    }
    dyn_pb_regex_str += ")_(";
    for (i = 0; i < PB_NUM_OP_TYPES; i++) {
        dyn_pb_regex_str += pb_op_types_list[i];
        pb_op_pm_types_encodings_map.insert(std::pair<string, uint8_t>
                    (pb_op_types_list[i], pb_op_types_encoding[i]));
        if (i < PB_NUM_OP_TYPES - 1)
            dyn_pb_regex_str += "|";
    }
    dyn_pb_regex_str += ")_(";
    for (i = 0; i < PN_NUM_OP_POWER_RAILS; i++) {
        dyn_pb_regex_str += pb_op_power_rails_list[i];
        pb_op_pm_power_rails_encodings_map.insert(std::pair<string, uint8_t>
                    (pb_op_power_rails_list[i], pb_op_power_rails_encoding[i]));
        if (i < PN_NUM_OP_POWER_RAILS - 1)
            dyn_pb_regex_str += "|";
    }

    dyn_pb_regex_str += ")$";
    pb_dynamic_regex.assign(dyn_pb_regex_str);

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
        int32_t gpu_id = rvs::gpulist::GetGpuId(dev_location_id);

        if (-1 != gpu_id) {
            // that should be an AMD GPU

            // check for deviceid filtering
            if (!device_id_filtering
                    || (device_id_filtering && dev->device_id == deviceid)) {
                // check if the GPU is part of the PCIe check
                // (either device: all or the gpu_id is in
                // the device: <gpu id> list

                bool cur_gpu_selected = false;

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
                    if (!get_gpu_all_pcie_capabilities(dev,
                        static_cast<uint16_t>(gpu_id)))
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
