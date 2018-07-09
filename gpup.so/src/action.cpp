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

#include <iostream>
#include <stdlib.h>
#include <dirent.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <regex>
#include <map>
#include "rvs_module.h"
#include "gpu_util.h"
#include "rvs_util.h"
#include "rvsloglp.h"


#define KFD_SYS_PATH_NODES "/sys/class/kfd/kfd/topology/nodes"

#define JSON_PROP_NODE_NAME             "properties"
#define JSON_CREATE_NODE_ERROR          "JSON cannot create node"

#define YAML_DEVICE_PROP_DELIMITER      ","

#define MODULE_NAME                     "gpup"

using namespace std;
using std::string;

/**
 * default class constructor
 */
action::action()
{
    bjson = false;
    json_root_node = NULL;
}

/**
 * class destructor
 */
action::~action()
{
	property.clear();
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


int action::run(void)
{    
    ifstream f_id,f_prop,f_link_prop;
    char path[256];
    string prop_name, prop_val, gpu_id, devices, msg, s;
    int num_nodes, num_links;
    bool device_all_selected = false, dev_id_corr;
    int error = 0;
    void *json_gpuprop_node = NULL;
    std::map<string,string>::iterator it;
        
    // discover the number of nodes: Inside nodes folder there are only folders that represent the node number
    num_nodes = gpu_num_subdirs((char*)KFD_SYS_PATH_NODES, (char*)"");
        
    // get the action name
    property_get_action_name();

    // get <device> property value (a list of gpu id)
    device_all_selected = property_get_device(&error);
    
    // get the <deviceid> property value
    int dev_id = property_get_deviceid(&error);
    
    
    bjson = false;  // already initialized in the default constructor

    // check for -j flag (json logging)
    if( property.find("cli.-j") != property.end())
    {
        unsigned int sec;
        unsigned int usec;
        rvs::lp::get_ticks(sec, usec);

        bjson = true;

        json_root_node = rvs::lp::LogRecordCreate(MODULE_NAME, action_name.c_str(), rvs::loginfo, sec, usec);
        if (json_root_node == NULL) {
            // log the error
            msg = action_name + " " + MODULE_NAME + " " + JSON_CREATE_NODE_ERROR;
            log(msg.c_str(), rvs::logerror);
        }
    } 
    

    for (vector<string>::iterator it_gpu_id = gpus_id.begin(); it_gpu_id != gpus_id.end(); ++it_gpu_id)
    {
        for(int node_id=0; node_id<num_nodes; node_id++){
            snprintf(path, 256, "%s/%d/gpu_id", KFD_SYS_PATH_NODES, node_id);
            f_id.open(path);
            snprintf(path, 256, "%s/%d/properties", KFD_SYS_PATH_NODES, node_id);
            f_prop.open(path);
            
            dev_id_corr = true;

            if(dev_id != -1){
                while( f_prop >> s ){
                    if( s == "device_id" ){
                        f_prop >> s;
                        if ( std::to_string(dev_id) != s)//skip this node
                            dev_id_corr = false;
                    }
                    f_prop>> s;
                }
                f_prop.clear();
                f_prop.seekg( 0, std::ios::beg );
            }
                
            f_id >> gpu_id;
                
            if ( gpu_id == *it_gpu_id  && dev_id_corr){
                
            json_gpuprop_node = NULL;

            if ( bjson ){
                if (json_root_node != NULL) {
                    json_gpuprop_node = rvs::lp::CreateNode(json_root_node, JSON_PROP_NODE_NAME);
                    if (json_gpuprop_node == NULL) {
                    // log the error
                        msg = action_name + " " + MODULE_NAME + " " + JSON_CREATE_NODE_ERROR;
                        log(msg.c_str(), rvs::logerror);
                    }
                }
            }

            if (bjson && json_gpuprop_node != NULL) { // json logging stuff
                rvs::lp::AddString(json_gpuprop_node, RVS_JSON_LOG_GPU_ID_KEY, gpu_id);
            }
                for ( it=property.begin(); it!=property.end(); ++it ){
                    snprintf(path, 256, "%s/%d/io_links", KFD_SYS_PATH_NODES, node_id);
                    num_links = gpu_num_subdirs((char*)path, (char*)"");
                    
                    s = it->first;

                    if( s.find(".")!= std::string::npos && s.substr(0,s.find(".")) == "properties"){
                        if( s.substr(s.find(".")+1) == "all" ){
                            while(f_prop >> prop_name){
                                f_prop >> prop_val;
                                msg = action_name + " " + MODULE_NAME  + " " + gpu_id + " "+ prop_name + " " + prop_val;
                                log( msg.c_str(), rvs::logresults);
                                if (bjson && json_gpuprop_node != NULL) // json logging stuff
                                    rvs::lp::AddString(json_gpuprop_node, prop_name, prop_val);                                
                            }
                        }
                        else{
                            while( f_prop >> prop_name ){
                                if (prop_name == s.substr(s.find(".")+1)){
                                    f_prop >> prop_val;
                                    msg = action_name + " " + MODULE_NAME  + " " + gpu_id + " "+ prop_name + " " + prop_val;
                                    log( msg.c_str(), rvs::logresults);
                                    if (bjson && json_gpuprop_node != NULL)  // json logging stuff
                                        rvs::lp::AddString(json_gpuprop_node, prop_name, prop_val);                                
                                    break;
                                }
                                f_prop >> prop_val;
                            }
                        }
                    }
                    f_prop.clear();
                    f_prop.seekg( 0, std::ios::beg );
                    
                    if( s.find(".")!= std::string::npos && s.substr(0,s.find(".")) == "io_links-properties"){                 
                        for(int link_id=0; link_id<num_links; link_id++){
                            snprintf(path, 256, "%s/%d/io_links/%d/properties", KFD_SYS_PATH_NODES, node_id, link_id);
                            f_link_prop.open(path);
                            if( s.substr(s.find(".")+1) == "all" ){
                                msg = action_name + " " + MODULE_NAME+ " " + gpu_id + " " + std::to_string(link_id) + " "+ "count" + " " + std::to_string(num_links);
                                log( msg.c_str(), rvs::logresults);
                                if (bjson && json_gpuprop_node != NULL)  // json logging stuff
                                    rvs::lp::AddString(json_gpuprop_node, "count" , std::to_string(num_links));
                                while(f_link_prop >> prop_name){
                                    f_link_prop >> prop_val;
                                    msg = action_name + " " + MODULE_NAME + " " + gpu_id + " " + std::to_string(link_id) + " "+ prop_name + " " + prop_val;
                                    log( msg.c_str(), rvs::logresults);
                                    if (bjson && json_gpuprop_node != NULL)  // json logging stuff
                                        rvs::lp::AddString(json_gpuprop_node, prop_name , prop_val);
                                }
                            }
                            else{
                                while(f_link_prop >> prop_name){
                                    if (prop_name == s.substr(s.find(".")+1)){
                                        f_link_prop >> prop_val;
                                        msg = action_name + " " + MODULE_NAME + " " + gpu_id + " " + std::to_string(link_id) + " "+ prop_name + " " + prop_val;
                                        log( msg.c_str(), rvs::logresults);
                                        if (bjson && json_gpuprop_node != NULL)  // json logging stuff
                                            rvs::lp::AddString(json_gpuprop_node, prop_name , prop_val);
                                        break;
                                    }
                                f_prop >> prop_val;
                                }
                            }
                            f_link_prop.close();
                        }
                    }
                }
            }
            f_id.close();
            f_prop.close();
        }
            if (bjson && json_gpuprop_node != NULL)  // json logging stuff
            rvs::lp::AddNode(json_root_node, json_gpuprop_node);
    }


    if (bjson && json_root_node != NULL) {  // json logging stuff
        rvs::lp::LogRecordFlush(json_root_node);
    }
    
    return 0;
}