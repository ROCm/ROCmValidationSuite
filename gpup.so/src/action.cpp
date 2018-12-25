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

#include "include/action.h"

#include <string>
#include <vector>
#include <fstream>
#include <regex>
#include <map>
#include <iostream>
#include <sstream>

#include "include/rvs_key_def.h"
#include "include/rvs_module.h"
#include "include/gpu_util.h"
#include "include/rvs_util.h"
#include "include/rvsloglp.h"


#define KFD_QUERYING_ERROR              "An error occurred while querying "\
                                        "the GPU properties"

#define KFD_SYS_PATH_NODES "/sys/class/kfd/kfd/topology/nodes"

#define JSON_PROP_NODE_NAME             "properties"
#define JSON_IO_LINK_PROP_NODE_NAME     "io_links-properties"
#define JSON_CREATE_NODE_ERROR          "JSON cannot create node"

#define CHAR_MAX_BUFF_SIZE              256

#define MODULE_NAME                     "gpup"
#define MODULE_NAME_CAPS                "GPUP"

using std::string;
using std::regex;
using std::vector;
using std::map;


/**
 * default class constructor
 */
gpup_action::gpup_action() {
    bjson = false;
    json_root_node = NULL;
}

/**
 * class destructor
 */
gpup_action::~gpup_action() {
    property.clear();
}

/**
 * extract properties/io_links properties names
 * @param props JSON_PROP_NODE_NAME or JSON_IO_LINK_PROP_NODE_NAME
 * @return true if success, false otherwise
 */
bool gpup_action::property_split(string props) {
  string s;
//   auto prop_length = std::end(gpu_prop_names) - std::begin(gpu_prop_names);
//   auto io_prop_length = std::end(gpu_io_link_prop_names) -
//  std::begin(gpu_io_link_prop_names);
  std::string prop_name_;

  RVSTRACE_
  for (auto it = property.begin(); it != property.end(); ++it) {
    RVSTRACE_
    s = it->first;
    if (s.find(".") != std::string::npos && s.substr(0, s.find(".")) ==
      props) {
      RVSTRACE_
      prop_name_ = s.substr(s.find(".")+1);
      if (prop_name_ == "all") {
        RVSTRACE_
        if (props == JSON_PROP_NODE_NAME) {
          RVSTRACE_
          property_name.clear();
        } else {
          RVSTRACE_
          io_link_property_name.clear();
        }
        RVSTRACE_
        return true;
      } else {
        RVSTRACE_
        if (props == JSON_PROP_NODE_NAME) {
          RVSTRACE_
          RVSDEBUG("property", prop_name_);
          property_name.push_back(prop_name_);
        } else if (props == JSON_IO_LINK_PROP_NODE_NAME) {
          RVSTRACE_
          RVSDEBUG("io_link_property", prop_name_);
          io_link_property_name.push_back(prop_name_);
        }
        RVSTRACE_
      }
      RVSTRACE_
    }
    RVSTRACE_
  }
  RVSTRACE_
  return false;
}

/**
 * Remove all accurances of 'name' in vector property_name_validate
 * @param name string to look for
 * @return 0 all the time
 */
int gpup_action::validate_property_name(const std::string& name) {
  auto it = std::find(property_name_validate.begin(),
                      property_name_validate.end(), name);
  while (it != property_name_validate.end()) {
    property_name_validate.erase(it);
    it = std::find(property_name_validate.begin(), property_name_validate.end()
                   , name);
  }
  return 0;
}

/**
 * gets properties values
 * @param gpu_id value of gpu_id of device
 */
int gpup_action::property_get_value(uint16_t gpu_id) {
  uint16_t node_id;
  char path[CHAR_MAX_BUFF_SIZE];
  void *json_gpuprop_node = NULL;
  string prop_name, prop_val, msg;
  std::ifstream f_prop;

  RVSTRACE_
  if (rvs::gpulist::gpu2node(gpu_id, &node_id)) {
    RVSTRACE_
    return -1;
  }

  // cache property names to validate for existance
  property_name_validate = property_name;

  snprintf(path, CHAR_MAX_BUFF_SIZE, "%s/%d/properties",
           KFD_SYS_PATH_NODES, node_id);

  if (bjson) {
    RVSTRACE_
    if (json_root_node == NULL) {
      RVSTRACE_
      return -1;
    }
    json_gpuprop_node = rvs::lp::CreateNode(json_root_node,
                                            JSON_PROP_NODE_NAME);
    if (json_gpuprop_node == NULL) {
      RVSTRACE_
      // log the error
      msg = std::string(JSON_CREATE_NODE_ERROR);
      rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
      return -1;
    }
    rvs::lp::AddNode(json_root_node, json_gpuprop_node);
  }

  RVSTRACE_
  f_prop.open(path);
  while (f_prop >> prop_name) {
    RVSTRACE_
    f_prop >> prop_val;

    validate_property_name(prop_name);
    // check if filtering by property is needed
    if (io_link_property_name.size() > 0) {
      auto it = std::find(property_name.begin(),
                          property_name.end(),
                          prop_name);
      // not found - skip to next property
      if (it == property_name.end()) {
        continue;
      }
    }
    msg = "["+action_name + "] " + MODULE_NAME +
    " " + std::to_string(gpu_id) +
    " " + prop_name + " " + prop_val;
    rvs::lp::Log(msg, rvs::logresults);
    if (bjson && json_gpuprop_node != NULL) {
      rvs::lp::AddString(json_gpuprop_node, prop_name, prop_val);
    }
  }
  RVSTRACE_
  f_prop.close();

  if (property_name_validate.size() > 0) {
    RVSTRACE_
    msg = "Properties not found for GPU " + std::to_string(gpu_id) + ":";
    for (auto it = property_name_validate.begin();
         it != property_name_validate.end(); it++) {
      msg += " " + *it;
    }
    rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
    return -1;
  }

  RVSTRACE_
  return 0;
}

/**
 * get io links properties values
 * @param gpu_id unique gpu_id
 */
int gpup_action::property_io_links_get_value(uint16_t gpu_id) {
  void* json_iolinks_node = nullptr;
  char path[CHAR_MAX_BUFF_SIZE];
  string prop_name, prop_val, msg;
  std::ifstream f_prop;
  uint16_t node_id;

  RVSTRACE_
  if (rvs::gpulist::gpu2node(gpu_id, &node_id)) {
    RVSTRACE_
    return -1;
  }

  snprintf(path, CHAR_MAX_BUFF_SIZE, "%s/%d/io_links",
           KFD_SYS_PATH_NODES, node_id);
  int num_links = gpu_num_subdirs(const_cast<char*>(path),
                                  const_cast<char*>(""));

  // construct node for IO links collection
  if (bjson) {
    RVSTRACE_
    json_iolinks_node = rvs::lp::CreateNode(json_root_node,
                                            JSON_IO_LINK_PROP_NODE_NAME);
    if (json_iolinks_node == NULL) {
      RVSTRACE_
      // log the error
      msg = std::string(JSON_CREATE_NODE_ERROR);
      rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
      return -1;
    }
    rvs::lp::AddNode(json_root_node, json_iolinks_node);
  }
  RVSTRACE_

  // for all links
  for (int link_id = 0; link_id < num_links; link_id++) {
    void* json_link_ptr_ = nullptr;

    snprintf(path, CHAR_MAX_BUFF_SIZE,
             "%s/%d/io_links/%d/properties",
             KFD_SYS_PATH_NODES, node_id, link_id);

    if (bjson) {
      RVSTRACE_
      json_link_ptr_ = rvs::lp::CreateNode(json_iolinks_node,
                                           std::to_string(link_id).c_str());
      if (json_link_ptr_ == NULL) {
        // log the error
        msg = std::string(JSON_CREATE_NODE_ERROR);
        rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
        return -1;
      }
      rvs::lp::AddNode(json_iolinks_node, json_link_ptr_);
    }

    RVSTRACE_
    f_prop.open(path);
    while (f_prop >> prop_name) {
      RVSTRACE_
      f_prop >> prop_val;

      // filter by property name if needed
      if (io_link_property_name.size() > 0) {
        auto it = std::find(io_link_property_name.begin(),
                            io_link_property_name.end(),
                            prop_name);
        if (it == io_link_property_name.end()) {
          continue;
        }
      }
      msg = "["+action_name + "] " + MODULE_NAME +
      " " + std::to_string(gpu_id) +
      " " + std::to_string(link_id) +
      " " + prop_name + " " + prop_val;
      rvs::lp::Log(msg, rvs::logresults);
      if (bjson && json_link_ptr_ != NULL) {
        rvs::lp::AddString(json_link_ptr_, prop_name, prop_val);
      }
    }
    RVSTRACE_
    f_prop.close();
  }
  return 0;
}

/**
 * runs the whole GPUP logic
 * @return run result
 */
int gpup_action::run(void) {
    std::string msg;
    int sts = 0;

    // get the action name
    if (property_get(RVS_CONF_NAME_KEY, &action_name)) {
      rvs::lp::Err("Action name missing", MODULE_NAME_CAPS);
      return -1;
    }

    // get <device> property value (a list of gpu id)
    if (int sts = property_get_device()) {
      switch (sts) {
      case 1:
        msg = "Invalid 'device' key value.";
        break;
      case 2:
        msg = "Missing 'device' key.";
        break;
      }
      rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
      return -1;
    }

    // get the <deviceid> property value if provided
    if (property_get_int<uint16_t>(RVS_CONF_DEVICEID_KEY,
                                  &property_device_id, 0u)) {
      msg = "Invalid 'deviceid' key value.";
      rvs::lp::Err(msg, MODULE_NAME_CAPS, action_name);
      return -1;
    }

    // extract properties and io_links properties names
    property_split(JSON_PROP_NODE_NAME);
    property_split(JSON_IO_LINK_PROP_NODE_NAME);

    bjson = false;  // already initialized in the default constructor

    // check for -j flag (json logging)
    if (has_property("cli.-j")) {
        bjson = true;
    }

    // get all AMD GPUs
    vector<uint16_t> gpu;
    gpu_get_all_gpu_id(&gpu);
    bool b_gpu_found = false;

    // iterate over AMD GPUs
    for (auto it = gpu.begin(); it !=gpu.end(); ++it) {
      // filter by gpu_id if needed
      if (property_device_id > 0) {
        uint16_t dev_id;
        if (!rvs::gpulist::gpu2device(*it, &dev_id)) {
          if (dev_id != property_device_id) {
            continue;
          }
        } else {
          msg = "Device ID not found for GPU " + std::to_string(*it);
          rvs::lp::Err(msg, MODULE_NAME, action_name);
          return -1;
        }
      }

      // filter by device if needed
      if (!property_device_all) {
        if (std::find(property_device.begin(), property_device.end(), *it) ==
          property_device.end()) {
            continue;
        }
      }

      b_gpu_found = true;

      // if JSON required
      if (bjson) {
        unsigned int sec;
        unsigned int usec;
        rvs::lp::get_ticks(&sec, &usec);

        json_root_node = rvs::lp::LogRecordCreate(MODULE_NAME,
        action_name.c_str(), rvs::logresults, sec, usec);
        if (json_root_node == nullptr) {
          // log the error
          msg = JSON_CREATE_NODE_ERROR;
          rvs::lp::Err(msg, MODULE_NAME, action_name);
          return -1;
        }

        // Add GPU ID
        rvs::lp::AddInt(json_root_node, RVS_JSON_LOG_GPU_ID_KEY, *it);
      }

      // properties values
      sts = property_get_value(*it);

      // so far so good?
      if (sts == 0) {
        RVSTRACE_
        // do io_links properties
        sts = property_io_links_get_value(*it);
      }

      if (bjson) {  // json logging stuff
        RVSTRACE_
        rvs::lp::LogRecordFlush(json_root_node);
        json_root_node = nullptr;
      }

      if (sts) {
        RVSTRACE_
        break;
      }
    }  // for all gpu_id

    if (!b_gpu_found) {
      msg = "No device matches criteria from configuration. ";
      rvs::lp::Err(msg, MODULE_NAME, action_name);
      return -1;
    }
    return sts;
}
