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
#include <iostream>
#include <fstream>
#include <regex>
#include <utility>
#include <algorithm>
#include <memory>
#include <map>

#ifdef __cplusplus
extern "C" {
#endif
#include <pci/pci.h>
#ifdef __cplusplus
}
#endif
#include <dirent.h>

#define __HIP_PLATFORM_HCC__
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"

#include "iet_worker.h"
#include "gpu_util.h"
#include "rvs_util.h"
#include "rvs_module.h"
#include "rvsloglp.h"
#include "rocm_smi/rocm_smi.h"

using std::string;
using std::vector;
using std::map;
using std::regex;
using std::cerr;
using std::fstream;

#define RVS_CONF_TARGET_POWER_KEY       "target_power"
#define RVS_CONF_RAMP_INTERVAL_KEY      "ramp_interval"
#define RVS_CONF_TOLERANCE_KEY          "tolerance"
#define RVS_CONF_MAX_VIOLATIONS_KEY     "max_violations"
#define RVS_CONF_SAMPLE_INTERVAL_KEY    "sample_interval"
#define RVS_CONF_LOG_INTERVAL_KEY       "log_interval"
#define RVS_CONF_MATRIX_SIZE_KEY        "matrix_size"

#define MODULE_NAME                     "iet"

#define IET_DEFAULT_RAMP_INTERVAL       5000
#define IET_DEFAULT_LOG_INTERVAL        1000
#define IET_DEFAULT_MAX_VIOLATIONS      0
#define IET_DEFAULT_TOLERANCE           0.1
#define IET_DEFAULT_SAMPLE_INTERVAL     100

#define IET_DEFAULT_MATRIX_SIZE         5760

#define RVS_DEFAULT_PARALLEL            false
#define RVS_DEFAULT_COUNT               1
#define RVS_DEFAULT_WAIT                0
#define RVS_DEFAULT_DURATION            0

#define IET_NO_COMPATIBLE_GPUS          "No AMD compatible GPU found!"
#define PCI_ALLOC_ERROR                 "pci_alloc() error"

#define FLOATING_POINT_REGEX            "^[0-9]*\\.?[0-9]+$"

#define JSON_CREATE_NODE_ERROR          "JSON cannot create node"
#define IRQ_PATH_MAX_LENGTH             256
#define SMI_DEVICE_FOLDER_BASE_NAME     "device"
#define HWMON_FOLDER_BASE_NAME          "hwmon"
#define GPU_POWER_DATA_FILE             "power1_average"

/**
 * @brief call-back function to append to a vector of Devices
 * @param d represent device
 * @param p pointer
 * @return true if dev connected to monitor, false otherwise
 */
static bool smi_get_gpu_devices_list(
                const std::shared_ptr<amd::smi::Device> &d, void *p) {
  std::string val_str;
  assert(p != nullptr);

  std::vector<std::shared_ptr<amd::smi::Device>> *device_list =
    reinterpret_cast<std::vector<std::shared_ptr<amd::smi::Device>> *>(p);

  if (d->monitor() != nullptr) {
    device_list->push_back(d);
  }
  return false;
}

/**
 * @brief call-back function to append to a vector of Devices
 * @param d represent device
 * @param p pointer
 * @return true if dev connected to monitor, false otherwise
 */
static string get_hwmon_entry(const std::string &path) {
    DIR *dirp;
    struct dirent *dir;
    dirp = opendir(path.c_str());
    if (dirp) {
        while ((dir = readdir(dirp)) != 0) {
            if ((strcmp(dir->d_name, ".") == 0) ||
            (strcmp(dir->d_name, "..") == 0))
                continue;
            if (strstr(dir->d_name, HWMON_FOLDER_BASE_NAME) != NULL) {
                closedir(dirp);
                return path + "/" + dir->d_name + "/" + GPU_POWER_DATA_FILE;
            }
        }
        closedir(dirp);
        return "";
    } else {
        return "";
    }
}

/**
 * @brief default class constructor
 */
action::action() {
}

/**
 * @brief class destructor
 */
action::~action() {
    property.clear();
}

/**
 * @brief reads the EDPp test's ramp-up time from the module's properties collection
 * @param error pointer to a memory location where the error code will be stored
 */
void action::property_get_iet_ramp_interval(int *error) {
    *error = 0;
    iet_ramp_interval = IET_DEFAULT_RAMP_INTERVAL;
    map<string, string>::iterator it =
                            property.find(RVS_CONF_RAMP_INTERVAL_KEY);
    if (it != property.end()) {
        if (is_positive_integer(it->second))
            iet_ramp_interval = std::stoul(it->second);
        else
            *error = 1;
        property.erase(it);
    }
}

/**
 * @brief reads the log interval from the module's properties collection
 * @param error pointer to a memory location where the error code will be stored
 */
void action::property_get_iet_log_interval(int *error) {
    *error = 0;
    iet_log_interval = IET_DEFAULT_LOG_INTERVAL;
    map<string, string>::iterator it =
                            property.find(RVS_CONF_LOG_INTERVAL_KEY);
    if (it != property.end()) {
        if (is_positive_integer(it->second)) {
            iet_log_interval = std::stoul(it->second);
            if (iet_log_interval == 0)
                iet_log_interval = IET_DEFAULT_LOG_INTERVAL;
        } else {
            *error = 1;
        }
        property.erase(it);
    }
}

/**
 * @brief reads the sample interval from the module's properties collection
 * @param error pointer to a memory location where the error code will be stored
 */
void action::property_get_iet_sample_interval(int *error) {
    *error = 0;
    iet_sample_interval = IET_DEFAULT_SAMPLE_INTERVAL;
    map<string, string>::iterator it =
                            property.find(RVS_CONF_SAMPLE_INTERVAL_KEY);
    if (it != property.end()) {
        if (is_positive_integer(it->second)) {
            iet_sample_interval = std::stoul(it->second);
            if (iet_sample_interval == 0)
                iet_sample_interval = IET_DEFAULT_SAMPLE_INTERVAL;
        } else {
            *error = 1;
        }
        property.erase(it);
    }
}

/**
 * @brief reads the max_violations (maximum allowed number of target_power violations)
 * from the module's properties collection
 * @param error pointer to a memory location where the error code will be stored
 */
void action::property_get_iet_max_violations(int *error) {
    *error = 0;
    iet_max_violations = IET_DEFAULT_MAX_VIOLATIONS;
    map<string, string>::iterator it =
                            property.find(RVS_CONF_MAX_VIOLATIONS_KEY);
    if (it != property.end()) {
        if (is_positive_integer(it->second))
            iet_max_violations = std::stoi(it->second);
        else
            *error = 1;
        property.erase(it);
    }
}

/**
 * @brief reads the target power level from the module's properties collection
 * @param error pointer to a memory location where the error code will be stored
 */
void action::property_get_iet_target_power(int *error) {
    *error = 0;  // init with 'no error'
    map<string, string>::iterator it =
                            property.find(RVS_CONF_TARGET_POWER_KEY);
    if (it != property.end()) {
        try {
            regex float_number_regex(FLOATING_POINT_REGEX);
            if (!regex_match(it->second, float_number_regex)) {
                *error = 1;  // not a floating point number
            } else {
                iet_target_power = std::stof(it->second);
            }
        } catch (const std::regex_error& e) {
            *error = 1;  // something went wrong with the regex
        }
    } else {
        *error = 2;
    }
}

/**
 * @brief reads the power tolerance from the module's properties collection
 * @param error pointer to a memory location where the error code will be stored
 */
void action::property_get_iet_tolerance(int *error) {
    *error = 0;
    iet_tolerance = IET_DEFAULT_TOLERANCE;
    map<string, string>::iterator it = property.find(RVS_CONF_TOLERANCE_KEY);
    if (it != property.end()) {
        try {
            regex float_number_regex(FLOATING_POINT_REGEX);
            if (regex_match(it->second, float_number_regex)) {
                iet_tolerance = std::stof(it->second);
            } else {
                *error = 1;  // not a floating point number
            }
        } catch (const std::regex_error& e) {
            *error = 1;
        }
        property.erase(it);
    }
}

/**
 * @brief reads matrix size from the module's properties collection
 * @param error pointer to a memory location where the error code will be stored
 */
void action::property_get_iet_matrix_size(int *error) {
    *error = 0;
    iet_matrix_size = IET_DEFAULT_MATRIX_SIZE;
    map<string, string>::iterator it =
                            property.find(RVS_CONF_MATRIX_SIZE_KEY);
    if (it != property.end()) {
        if (is_positive_integer(it->second))
            iet_matrix_size = std::stoul(it->second);
        else
            *error = 1;
        property.erase(it);
    }
}

/**
 * @brief reads all IET related configuration keys from
 * the module's properties collection
 * @return true if no fatal error occured, false otherwise
 */
bool action::get_all_iet_config_keys(void) {
    int error;
    string msg, ststress;

    if (has_property(RVS_CONF_TARGET_POWER_KEY, ststress)) {
        property_get_iet_target_power(&error);
        if (error) {  // <target_power> is mandatory => IET cannot continue
            cerr << "RVS-IET: action: " << action_name <<
                "  invalid '" << RVS_CONF_TARGET_POWER_KEY <<
                "' key value " << ststress << std::endl;
            return false;
        }
    } else {
        cerr << "RVS-IET: action: " << action_name <<
            "  key '" << RVS_CONF_TARGET_POWER_KEY <<
            "' was not found" << std::endl;
        return false;
    }

    property_get_iet_ramp_interval(&error);
    if (error) {
        cerr << "RVS-IET: action: " << action_name <<
            "  invalid '" << RVS_CONF_RAMP_INTERVAL_KEY << "'" << std::endl;
        return false;
    }

    property_get_iet_log_interval(&error);
    if (error) {
        cerr << "RVS-IET: action: " << action_name <<
            "  invalid '" << RVS_CONF_LOG_INTERVAL_KEY << "'" << std::endl;
        return false;
    }

    property_get_iet_sample_interval(&error);
    if (error) {
        cerr << "RVS-IET: action: " << action_name <<
            "  invalid '" << RVS_CONF_SAMPLE_INTERVAL_KEY << "'" << std::endl;
        return false;
    }


    property_get_iet_max_violations(&error);
    if (error) {
        cerr << "RVS-IET: action: " << action_name <<
            "  invalid '" << RVS_CONF_MAX_VIOLATIONS_KEY << "'" << std::endl;
        return false;
    }

    property_get_iet_tolerance(&error);
    if (error) {
        cerr << "RVS-IET: action: " << action_name <<
            "  invalid '" << RVS_CONF_TOLERANCE_KEY << "'" << std::endl;
        return false;
    }

    property_get_iet_matrix_size(&error);
    if (error) {
        cerr << "RVS-IET: action: " << action_name <<
            "  invalid '" << RVS_CONF_MATRIX_SIZE_KEY << "'" << std::endl;
        return false;
    }
    return true;
}

/**
 * @brief reads all common configuration keys from
 * the module's properties collection
 * @return true if no fatal error occured, false otherwise
 */
bool action::get_all_common_config_keys(void) {
    string msg, sdevid, sdev;
    int error;

    // get <device> property value (a list of gpu id)
    if (has_property("device", sdev)) {
        device_all_selected = property_get_device(&error);
        if (error) {  // log the error & abort IET
            cerr << "RVS-IET: action: " << action_name <<
                "  invalid 'device' key value " << sdev << std::endl;
            return false;
        }
    } else {
        cerr << "RVS-IET: action: " << action_name <<
            "  key 'device' was not found" << std::endl;
        return false;
    }

    // get the <deviceid> property value
    if (has_property("deviceid", sdevid)) {
        int devid = property_get_deviceid(&error);
        if (!error) {
            if (devid != -1) {
                deviceid = static_cast<uint16_t>(devid);
                device_id_filtering = true;
            }
        } else {
            cerr << "RVS-IET: action: " << action_name <<
                "  invalid 'deviceid' key value " << sdevid << std::endl;
            return false;
        }
    }

    // get the other action/IET related properties
    rvs::actionbase::property_get_run_parallel(&error);
    if (error == 1) {
        cerr << "RVS-IET: action: " << action_name <<
            "  invalid '" << RVS_CONF_PARALLEL_KEY <<
            "' key value" << std::endl;
        return false;
    }

    rvs::actionbase::property_get_run_count(&error);
    if (error == 1) {
        cerr << "RVS-IET: action: " << action_name <<
            "  invalid '" << RVS_CONF_COUNT_KEY << "' key value" << std::endl;
        return false;
    }

    rvs::actionbase::property_get_run_wait(&error);
    if (error == 1) {
        cerr << "RVS-IET: action: " << action_name <<
            "  invalid '" << RVS_CONF_WAIT_KEY << "' key value" << std::endl;
        return false;
    }

    rvs::actionbase::property_get_run_duration(&error);
    if (error == 1) {
        cerr << "RVS-IET: action: " << action_name <<
            "  invalid '" << RVS_CONF_DURATION_KEY <<
            "' key value" << std::endl;
        return false;
    }

    return true;
}

/**
 * @brief runs the edp test
 * @return true if no error occured, false otherwise
 */
bool action::do_edp_test(void) {
    size_t k = 0;
    while (1) {
        unsigned int i = 0;
        if (gst_run_wait_ms != 0)  // delay iet execution
            sleep(gst_run_wait_ms);

        vector<IETWorker> workers(edpp_gpus.size());

        vector<gpu_hwmon_info>::iterator it;

        // all worker instances have the same json settings
        IETWorker::set_use_json(bjson);

        for (it = edpp_gpus.begin(); it != edpp_gpus.end(); ++it) {
            // set worker thread params
            workers[i].set_name(action_name);
            workers[i].set_gpu_id((*it).gpu_id);
            workers[i].set_gpu_device_index((*it).hip_gpu_deviceid);
            workers[i].set_gpu_hwmon_entry((*it).gpu_hwmon_power_entry);
            workers[i].set_run_wait_ms(gst_run_wait_ms);
            workers[i].set_run_duration_ms(gst_run_duration_ms);
            workers[i].set_ramp_interval(iet_ramp_interval);
            workers[i].set_log_interval(iet_log_interval);
            workers[i].set_sample_interval(iet_sample_interval);
            workers[i].set_max_violations(iet_max_violations);
            workers[i].set_target_power(iet_target_power);
            workers[i].set_tolerance(iet_tolerance);
            workers[i].set_matrix_size(iet_matrix_size);
            i++;
        }

        if (gst_runs_parallel) {
            for (i = 0; i < edpp_gpus.size(); i++)
                workers[i].start();

            // join threads
            for (i = 0; i < edpp_gpus.size(); i++)
                workers[i].join();
        } else {
            for (i = 0; i < edpp_gpus.size(); i++) {
                workers[i].start();
                workers[i].join();
            }
        }

        if (gst_run_count != 0) {
            k++;
            if (k == gst_run_count)
                break;
        }
    }
    return true;
}

/**
 * @brief gets irq value for device
 * @param dev_path represents path of device
 * @return irq value
 */
const std::string action::get_irq(const std::string dev_path) {
    std::ifstream f_id;
    char path[IRQ_PATH_MAX_LENGTH];
    string irq = "";

    snprintf(path, IRQ_PATH_MAX_LENGTH, "%s/device/irq", dev_path.c_str());
    f_id.open(path);

    if (f_id.is_open()) {
        f_id >> irq;
        f_id.close();
    }
    return irq;
}

/**
 * @brief gets the number of ROCm compatible AMD GPUs
 * @return run number of GPUs
 */
int action::get_num_amd_gpu_devices(void) {
    int hip_num_gpu_devices;
    string msg;
    hipGetDeviceCount(&hip_num_gpu_devices);
    if (hip_num_gpu_devices == 0) {  // no AMD compatible GPU
        msg = action_name + " " + MODULE_NAME + " " + IET_NO_COMPATIBLE_GPUS;
        log(msg.c_str(), rvs::logerror);

        if (bjson) {
            unsigned int sec;
            unsigned int usec;
            rvs::lp::get_ticks(sec, usec);
            void *json_root_node = rvs::lp::LogRecordCreate(MODULE_NAME,
                            action_name.c_str(), rvs::loginfo, sec, usec);
            if (!json_root_node) {
                // log the error
                string msg = action_name + " " + MODULE_NAME + " "
                                            + JSON_CREATE_NODE_ERROR;
                log(msg.c_str(), rvs::logerror);
                return 0;
            }

            rvs::lp::AddString(json_root_node, "ERROR", IET_NO_COMPATIBLE_GPUS);
            rvs::lp::LogRecordFlush(json_root_node);
        }
        return 0;
    }
    return hip_num_gpu_devices;
}

/**
 * @brief retrieves the hwmon path (and some other info) for the given GPU
 * and adds it to the list of those that will run the EDPp test
 * @param dev_location_id GPU device location ID
 * @param gpu_irq GPU's IRQ
 * @param gpu_id GPU's ID as exported by KFD
 * @param hip_num_gpu_devices number of GPU devices (as reported by HIP API)
 * @return true if all info could be retrieved and the gpu was successfully to
 * the EDPp test list, false otherwise
 */
bool action::add_gpu_to_edpp_list(uint16_t dev_location_id, uint16_t gpu_irq,
                                  int32_t gpu_id, int hip_num_gpu_devices) {
    bool dev_index_found = false;

    for (int i = 0; i < hip_num_gpu_devices; i++) {
        // get GPU device properties
        hipDeviceProp_t props;
        hipGetDeviceProperties(&props, i);

        // compute device location_id (needed to match this device
        // with one of those found while querying the pci bus
        uint16_t hip_dev_location_id =
                ((((uint16_t) (props.pciBusID)) << 8) | (props.pciDeviceID));

        if (hip_dev_location_id == dev_location_id) {
            // now try finding the hwmon entry which is
            // needed for the power related data
            for (auto cgpu_dev : monitor_devices) {
                // get irq of device
                string irq = get_irq(cgpu_dev ->path());
                if (irq != "" && gpu_irq == std::stoi(irq)) {
                    // found the device
                    string base_folder_hwmon = cgpu_dev->path() +
                            "/" + SMI_DEVICE_FOLDER_BASE_NAME +
                            "/" + HWMON_FOLDER_BASE_NAME;
                    string cgpu_hwmon_power_entry =
                        get_hwmon_entry(base_folder_hwmon);
                    if (cgpu_hwmon_power_entry != "") {
                        gpu_hwmon_info cgpu_info;
                        cgpu_info.hip_gpu_deviceid = i;
                        cgpu_info.gpu_id = gpu_id;
                        cgpu_info.gpu_hwmon_power_entry =
                            cgpu_hwmon_power_entry;
                        edpp_gpus.push_back(cgpu_info);

                        return true;
                    }
                    break;
                }
            }
            if (dev_index_found)
                break;
        }
    }

    return false;
}


/**
 * @brief gets all selected GPUs and starts the worker threads
 * @return run result
 */
int action::get_all_selected_gpus(void) {
    string msg;
    bool amd_gpus_found = false;
    int hip_num_gpu_devices;
    struct pci_access *pacc;
    struct pci_dev *pci_cdev;
    amd::smi::RocmSMI hw;

    hip_num_gpu_devices = get_num_amd_gpu_devices();
    if (hip_num_gpu_devices == 0)
        return 0;  // no AMD compatible GPU found!

    // get GPU devices via smi lib
    hw.DiscoverDevices();
    hw.IterateSMIDevices(smi_get_gpu_devices_list,
      reinterpret_cast<void *>(&monitor_devices));

    // get the pci_access structure
    pacc = pci_alloc();

    if (pacc == NULL) {
        // log the error
        msg = action_name + " " + MODULE_NAME + " " + PCI_ALLOC_ERROR;
        log(msg.c_str(), rvs::logerror);

        return -1;  // EDPp test cannot continue
    }

    // initialize the PCI library
    pci_init(pacc);
    // get the list of devices
    pci_scan_bus(pacc);

    // iterate over devices
    for (pci_cdev = pacc->devices; pci_cdev; pci_cdev = pci_cdev->next) {
        // fill in the info
        pci_fill_info(pci_cdev,
                PCI_FILL_IDENT | PCI_FILL_BASES | PCI_FILL_CLASS
                | PCI_FILL_EXT_CAPS | PCI_FILL_CAPS | PCI_FILL_PHYS_SLOT);

        // computes the actual dev's location_id (sysfs entry)
        uint16_t dev_location_id = ((((uint16_t) (pci_cdev->bus)) << 8)
                | (pci_cdev->func));

        // check if this pci_dev corresponds to one of the AMD GPUs
        int32_t gpu_id = rvs::gpulist::GetGpuId(dev_location_id);
        if (gpu_id == -1)
            continue;

        // that should be an AMD GPU
        // check for deviceid filtering
        if (!device_id_filtering || (device_id_filtering &&
                        pci_cdev->device_id == deviceid)) {
            // check if the GPU is part of the EDPp test  (either <device>: all
            // or the gpu_id is in the device: <gpu id> list)

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

            if (cur_gpu_selected)
                if (add_gpu_to_edpp_list(dev_location_id, pci_cdev->irq,
                    gpu_id, hip_num_gpu_devices))
                    amd_gpus_found = true;
        }
    }

    pci_cleanup(pacc);

    if (amd_gpus_found) {
        if (do_edp_test())
            return 0;
        return -1;
    }

    return 0;
}

/**
 * @brief runs the whole IET logic
 * @return run result
 */
int action::run(void) {
    string msg;
    int error;

    // get the action name
    rvs::actionbase::property_get_action_name(&error);
    if (error == 2) {
      msg = "action name field is missing in iet module";
      log(msg.c_str(), rvs::logerror);
      return -1;
    }

    device_all_selected = false;
    device_id_filtering = false;

    // check for -j flag (json logging)
    if (property.find("cli.-j") != property.end())
        bjson = true;

    if (!get_all_common_config_keys())
        return -1;
    if (!get_all_iet_config_keys())
        return -1;

    if (gst_run_duration_ms > 0 && (gst_run_duration_ms < iet_ramp_interval)) {
        cerr << "RVS-IET: action: " << action_name <<
            "  'duration' cannot be less than 'ramp_interval'" << std::endl;
        return -1;
    }

    return get_all_selected_gpus();
}
