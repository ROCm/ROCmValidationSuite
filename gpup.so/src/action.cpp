/*******************************************************************************
 *
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

#include <string>
#include <vector>
#include <fstream>
#include <regex>
#include <map>
#include <iostream>
#include <sstream>
#include "action.h"
#include "rvs_module.h"
#include "gpu_util.h"
#include "rvs_util.h"
#include "rvsloglp.h"


#define KFD_SYS_PATH_NODES "/sys/class/kfd/kfd/topology/nodes"

#define RVS_CONF_NAME_KEY               "name"
#define RVS_CONF_DEVICE_KEY             "device"
#define RVS_CONF_DEVICE_ID_KEY          "device_id"
#define RVS_JSON_LOG_GPU_ID_KEY         "gpu_id"

#define JSON_PROP_NODE_NAME             "properties"
#define JSON_IO_LINK_PROP_NODE_NAME     "io_links-properties"
#define JSON_CREATE_NODE_ERROR          "JSON cannot create node"

#define CHAR_MAX_BUFF_SIZE              256

#define MODULE_NAME                     "gpup"

using std::string;
using std::regex;
using std::vector;
using std::map;
using std::cerr;
// collection of allowed GPU properties
const char* gpu_prop_names[] =
        { "cpu_cores_count", "simd_count", "mem_banks_count", "caches_count",
"io_links_count", "cpu_core_id_base", "simd_id_base", "max_waves_per_simd",
"lds_size_in_kb", "gds_size_in_kb", "wave_front_size", "array_count",
"simd_arrays_per_engine", "cu_per_simd_array", "simd_per_cu",
"max_slots_scratch_cu", "vendor_id", "device_id", "location_id",
"drm_render_minor", "max_engine_clk_ccompute", "local_mem_size", "fw_version",
"capability", "max_engine_clk_ccompute"
        };
// collection of allowed io links properties
const char* gpu_io_link_prop_names[] =
        { "count", "type", "version_major", "version_minor", "node_from",
"node_to", "weight", "min_latency", "max_latency", "min_bandwidth",
"max_bandwidth", "recommended_transfer_size", "flags"
        };

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
 * gets the gpu_id list from the module's properties collection
 * @param error pointer to a memory location where the error code will be stored
 * @return true if "all" is selected, false otherwise
 */
bool action::property_get_device(int *error, int num_nodes) {
    map<string, string>::iterator it;
    std::ifstream f_id;
    char path[CHAR_MAX_BUFF_SIZE];
    string gpu_id;
    *error = 0;  // init with 'no error'
    it = property.find(RVS_CONF_DEVICE_KEY);
    if (it != property.end()) {
        if (it->second == "all") {
          for (int node_id = 0; node_id < num_nodes; node_id++) {
                snprintf(path, CHAR_MAX_BUFF_SIZE, "%s/%d/gpu_id",
                KFD_SYS_PATH_NODES, node_id);
                f_id.open(path);
                f_id >> gpu_id;
                if (gpu_id != "0") {
                    gpus_id.push_back(gpu_id);
                }
                f_id.close();
        }
            property.erase(it);
            return true;
        } else {
            // split the list of gpu_id
            gpus_id = str_split(it->second, YAML_DEVICE_PROP_DELIMITER);
            property.erase(it);
            if (gpus_id.empty()) {
                *error = 1;  // list of gpu_id cannot be empty
            } else {
                for (vector<string>::iterator it_gpu_id =
                        gpus_id.begin();
                        it_gpu_id != gpus_id.end(); ++it_gpu_id) {
                    if (!is_positive_integer(*it_gpu_id)) {
                        *error = 1;
                        break;
                    }}
            }
            return false;
        }
    } else {
        *error = 1;
        return false;
    }
}

/**
 * checks if device id is correct
 * @param node_id represents node folder
 * @param dev_id unique device id
 * @return true if dev_id is correct, false otherwise
 */            
bool action::device_id_correct(int node_id, int dev_id) {
    std::ifstream f_prop;
    bool dev_id_corr = true;
    string s;
    char path[CHAR_MAX_BUFF_SIZE];

    snprintf(path, CHAR_MAX_BUFF_SIZE, "%s/%d/properties", KFD_SYS_PATH_NODES,
    node_id);
    f_prop.open(path);

    if (dev_id != -1) {
        while (f_prop >> s) {
            if (s == RVS_CONF_DEVICE_ID_KEY) {
                f_prop >> s;
                if (std::to_string(dev_id) != s)  // skip this node
                    dev_id_corr = false;
            }
            f_prop>> s;
        }
        f_prop.close();
    }
    return dev_id_corr;
}

/**
 * gets the gpu_id from node
 * @param node_id represents node folder
 * @return gpu_id value
 */
string action::property_get_gpuid(int node_id) {
    std::ifstream f_id;
    string gpu_id;
    char path[CHAR_MAX_BUFF_SIZE];

    snprintf(path, CHAR_MAX_BUFF_SIZE, "%s/%d/gpu_id", KFD_SYS_PATH_NODES,
    node_id);
    f_id.open(path);

    f_id >> gpu_id;
    return gpu_id;
}

/**
 * extract properties/io_links properties names
 * @param props JSON_PROP_NODE_NAME or JSON_IO_LINK_PROP_NODE_NAME
 * @return true if success, false otherwise
 */
bool action::property_split(string props) {
    map<string, string>::iterator it;
    string s;
    auto prop_length = std::end(gpu_prop_names) - std::begin(gpu_prop_names);
    auto io_prop_length = std::end(gpu_io_link_prop_names) -
    std::begin(gpu_io_link_prop_names);

     for (it = property.begin(); it != property.end(); ++it) {
         s = it->first;
        if (s.find(".") != std::string::npos && s.substr(0, s.find(".")) ==
        props) {
            if (!(s.substr(s.find(".")+1) == "all")) {
                if (props == JSON_PROP_NODE_NAME)
                    property_name.push_back(s.substr(s.find(".")+1));
                else
                    io_link_property_name.push_back(s.substr(s.find(".")+1));
            } else {
                if (props == JSON_PROP_NODE_NAME) {
                    for (int i = 0; i < prop_length; i++)
                        property_name.push_back(gpu_prop_names[i]);
                } else {
                    for (int i = 0; i < io_prop_length; i++)
                    io_link_property_name.push_back(gpu_io_link_prop_names[i]);
                }
                return 1;
            }
        }
    }
    return 0;
}

/**
 * gets properties values
 * @param gpu_id value of gpu_id of device
 * @param node_id represents node folder
 */
void action::property_get_value(string gpu_id, int node_id) {
    char path[CHAR_MAX_BUFF_SIZE];
    void *json_gpuprop_node = NULL;
    string prop_name, prop_val, msg;
    std::ifstream f_prop;

            snprintf(path, CHAR_MAX_BUFF_SIZE, "%s/%d/properties",
            KFD_SYS_PATH_NODES, node_id);
            f_prop.open(path);

            if (bjson) {
                if (json_root_node != NULL) {
                    json_gpuprop_node = rvs::lp::CreateNode(json_root_node,
                    JSON_PROP_NODE_NAME);
                    if (json_gpuprop_node == NULL) {
                        // log the error
                        msg = action_name + " " + MODULE_NAME + " " +
                        JSON_CREATE_NODE_ERROR;
                        cerr << "RVS-GPUP: " << msg;
                    }
                }
            }

            if (bjson && json_gpuprop_node != NULL) {  // json logging stuff
                rvs::lp::AddString(json_gpuprop_node, RVS_JSON_LOG_GPU_ID_KEY,
                gpu_id);
            }

            // properties
            for (vector<string>::iterator it_prop_name=property_name.begin();
            it_prop_name != property_name.end(); ++it_prop_name) {
                while (f_prop >> prop_name) {
                    if (prop_name == *it_prop_name) {
                        f_prop >> prop_val;
                        msg = "["+action_name + "] " + MODULE_NAME +
                        " " + gpu_id +
                        " " + prop_name + " " + prop_val;
                        log(msg.c_str(), rvs::logresults);
                    if (bjson && json_gpuprop_node != NULL) {
                        rvs::lp::AddString(json_gpuprop_node, prop_name,
                        prop_val);
                    }
                        break;
                    }
                    f_prop >> prop_val;
                }
                f_prop.clear();
                f_prop.seekg(0, std::ios::beg);
            }
        f_prop.close();

        if (bjson && json_gpuprop_node != NULL)  // json logging stuff
        rvs::lp::AddNode(json_root_node, json_gpuprop_node);
}

/**
 * get io links properties values
 * @param gpu_id unique gpu_id
 * @param node_id represents node folder
 */
void action::property_io_links_get_value(string gpu_id, int node_id) {
    void *json_gpuprop_node = NULL;
    char path[CHAR_MAX_BUFF_SIZE];
    string prop_name, prop_val, msg;
    std::ifstream f_link_prop;

            if (bjson) {
                if (json_root_node != NULL) {
                    json_gpuprop_node = rvs::lp::CreateNode(json_root_node,
                    JSON_IO_LINK_PROP_NODE_NAME);
                    if (json_gpuprop_node == NULL) {
                        // log the error
                        msg = action_name + " " + MODULE_NAME + " " +
                        JSON_CREATE_NODE_ERROR;
                        cerr << "RVS-GPUP: " << msg;
                    }
                }
            }

            if (bjson && json_gpuprop_node != NULL) {  // json logging stuff
                rvs::lp::AddString(json_gpuprop_node, RVS_JSON_LOG_GPU_ID_KEY,
                gpu_id);
            }

            snprintf(path, CHAR_MAX_BUFF_SIZE, "%s/%d/io_links",
            KFD_SYS_PATH_NODES, node_id);
            int num_links = gpu_num_subdirs(const_cast<char*>(path),
            const_cast<char*>(""));

            for (int link_id = 0; link_id < num_links; link_id++) {
                snprintf(path, CHAR_MAX_BUFF_SIZE,
                "%s/%d/io_links/%d/properties", KFD_SYS_PATH_NODES, node_id,
                link_id);
                f_link_prop.open(path);

                for (vector<string>::iterator it_io_prop_name =
                io_link_property_name.begin(); it_io_prop_name !=
                io_link_property_name.end();
                ++it_io_prop_name) {
                    // file doesn't contain property name "count" and its value
                    if (*it_io_prop_name == "count") {
                        msg = "["+action_name + "] " + MODULE_NAME +
                        " " + gpu_id +
                        " " + std::to_string(link_id) + " "+ "count" + " " +
                        std::to_string(num_links);
                        log(msg.c_str(), rvs::logresults);
                        if (bjson && json_gpuprop_node != NULL)
                            rvs::lp::AddString(json_gpuprop_node, "count" ,
                            std::to_string(num_links));
                    }
                    while (f_link_prop >> prop_name) {
                        if (prop_name == *it_io_prop_name) {
                            f_link_prop >> prop_val;
                            msg = "["+action_name + "] " + MODULE_NAME + " " +
                            gpu_id + " " + std::to_string(link_id) + " "+
                            prop_name + " " + prop_val;
                            log(msg.c_str(), rvs::logresults);
                            if (bjson && json_gpuprop_node != NULL) {
                                rvs::lp::AddString(json_gpuprop_node, prop_name
                                , prop_val);
                            }
                                break;
                        }
                        f_link_prop >> prop_val;
                    }
                    f_link_prop.clear();
                    f_link_prop.seekg(0, std::ios::beg);
                }
            }

        f_link_prop.close();

        if (bjson && json_gpuprop_node != NULL)  // json logging stuff
        rvs::lp::AddNode(json_root_node, json_gpuprop_node);
}

/**
 * runs the whole GPUP logic
 * @return run result
 */
int action::run(void) {
    string gpu_id, msg;
    int num_nodes;
    bool dev_id_corr;
    int error = 0;

    // discover the number of nodes: Inside nodes folder there are only folders
    // that represent the node number
    num_nodes = gpu_num_subdirs(const_cast<char*>(KFD_SYS_PATH_NODES),
    const_cast<char*>(""));

    // get the action name
    rvs::actionbase::property_get_action_name(&error);
    if (error == 2) {
      msg = "action field is missing in gpup module";
      cerr << "RVS-GPUP: " << msg;
      return -1;
    }

    // get <device> property value (a list of gpu id)
    property_get_device(&error, num_nodes);

    // get the <deviceid> property value if provided
    int dev_id = property_get_deviceid(&error);
    if (error != 0) {
      cerr << "RVS-GPUP: action: " << property["name"] <<
                  "  invalid 'deviceid' key value: " <<
                  dev_id << std::endl;
      return -1;
    }
    // extract properties and io_links properties names
    property_split(JSON_PROP_NODE_NAME);
    property_split(JSON_IO_LINK_PROP_NODE_NAME);

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
            msg = action_name + " " + MODULE_NAME + " " +
            JSON_CREATE_NODE_ERROR;
        }
    }

    for (vector<string>::iterator it_gpu_id = gpus_id.begin(); it_gpu_id !=
    gpus_id.end(); ++it_gpu_id) {
        for (int node_id = 0; node_id < num_nodes; node_id++) {
            dev_id_corr = device_id_correct(node_id, dev_id);

            gpu_id = property_get_gpuid(node_id);

            if (gpu_id == *it_gpu_id) {
              if (dev_id_corr) {
                // properties values
                property_get_value(gpu_id, node_id);
                // io_links properties
                property_io_links_get_value(gpu_id, node_id);
              } else {
                  cerr << "RVS-GPUP: action: " << property["name"] <<
                  "  invalid 'deviceid' key value: " <<
                  dev_id << std::endl;
                  return -1;
             }
            }
          }
      }

    if (bjson && json_root_node != NULL) {  // json logging stuff
        rvs::lp::LogRecordFlush(json_root_node);
    }

    return 0;
}
