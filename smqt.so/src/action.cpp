
#include "action.h"

#include <stdlib.h>
#include <stdio.h>      //log
#include <math.h>       //log
#include <iostream>
#include <stdlib.h>
#include <dirent.h>
#include <string.h>
#include <fstream>
#include <chrono>
#include <map>
#include <string>
#include <algorithm>
#include <cmath>
#ifdef __cplusplus
extern "C" {
  #endif
  #include <pci/pci.h>
  #include <linux/pci.h>
  #ifdef __cplusplus
}
#endif

#include "rvs_module.h"
#include "pci_caps.h"
#include "gpu_util.h"
#include "rvsloglp.h"

using namespace std;

//config
unsigned long bar1_req_size, bar1_base_addr_min, bar1_base_addr_max, bar2_req_size, bar2_base_addr_min, bar2_base_addr_max, bar4_req_size, bar4_base_addr_min, bar4_base_addr_max, bar5_req_size;
//output
unsigned long bar1_size, bar1_base_addr, bar2_size, bar2_base_addr, bar4_size, bar4_base_addr, bar5_size;
bool pass;

action::action()
{  

}

action::~action()
{
	property.clear();
}

int action::run(void)
{ 
  std::string action_name;
  std::vector<unsigned short int> gpus_location_id;
  struct pci_access *pacc;
 
  gpu_get_all_location_id(gpus_location_id);  
  //get the pci_access structure
  pacc = pci_alloc();
  //initialize the PCI library
  pci_init(pacc);
  //get the list of devices
  pci_scan_bus(pacc);
  
  struct pci_dev *dev;  
  dev = pacc->devices; 
  
  //get requsted values 
  for (auto it=property.begin(); it!=property.end(); ++it){
    if(it->first == "name")
      action_name = it->second;
    if (it->first=="bar1_req_size") 
      bar1_req_size=std::atoi (it->second.c_str());        
    if (it->first=="bar2_req_size") 
      bar2_req_size=std::atoi (it->second.c_str());
    if (it->first=="bar4_req_size") 
      bar4_req_size=std::atoi (it->second.c_str());
    if (it->first=="bar5_req_size") 
      bar5_req_size=std::atoi (it->second.c_str());
    
    if (it->first=="bar1_base_addr_min") 
      bar1_base_addr_min=std::atoi (it->second.c_str());
    if (it->first=="bar2_base_addr_min") 
      bar2_base_addr_min=std::atoi (it->second.c_str());
    if (it->first=="bar4_base_addr_min") 
      bar4_base_addr_min=std::atoi (it->second.c_str());
    
    if (it->first=="bar1_base_addr_max") 
      bar1_base_addr_max=std::atoi (it->second.c_str());
    if (it->first=="bar2_base_addr_max") 
      bar2_base_addr_max=std::atoi (it->second.c_str());
    if (it->first=="bar4_base_addr_max") 
      bar4_base_addr_max=std::atoi (it->second.c_str());        
  }
  
  //iterate over devices
  for (dev = pacc->devices; dev; dev = dev->next) {
    pci_fill_info(dev, PCI_FILL_IDENT | PCI_FILL_BASES | PCI_FILL_CLASS | PCI_FILL_EXT_CAPS | PCI_FILL_CAPS | PCI_FILL_PHYS_SLOT); //fil in the info
    
    //computes the actual dev's location_id (sysfs entry)
    unsigned short int dev_location_id = ((((unsigned short int)(dev->bus)) << 8) | (dev->func));
    
    //check if this pci_dev corresponds to one of AMD GPUs
    auto it_gpu = find(gpus_location_id.begin(), gpus_location_id.end(), dev_location_id);
    
    if (it_gpu == gpus_location_id.end())
      continue;
        
    //get actual values
    bar1_base_addr = dev->base_addr[0];
    bar1_size = dev->size[0];   
    bar2_base_addr = dev->base_addr[2];
    bar2_size = dev->size[2];
    bar4_base_addr = dev->base_addr[5];
    bar4_size = dev->size[5]; 
    bar5_size = dev->rom_size;

      
    //check if values are as expected  
    pass=true;
    
    if ((bar1_base_addr<bar1_base_addr_min)||(bar1_base_addr>bar1_base_addr_max))
      pass=false;
    if ((bar2_base_addr<bar1_base_addr_min)||(bar2_base_addr>bar2_base_addr_max))
      pass=false;
    if ((bar4_base_addr<bar4_base_addr_min)||(bar4_base_addr>bar4_base_addr_max))
      pass=false;
    
    if ((bar1_req_size>bar1_size)||(bar2_req_size<bar2_size)||(bar4_req_size<bar4_size)||(bar5_req_size<bar5_size))
      pass=false;

    //loginfo
    unsigned int sec;
    unsigned int usec;
    rvs::lp::get_ticks(sec, usec);
    string msgs1,msgs2,msgs4,msgs5,msga1,msga2,msga4,pmsg,str;
    char hex_value[30];
    if (remainder(bar1_size,pow(2,30))==0)
      str=std::to_string((int)round(bar1_size/pow(2,30)))+" GB";
    else if (remainder(bar1_size,pow(2,20))==0)
      str=std::to_string((int)round(bar1_size/pow(2,20)))+" MB";
    else if (remainder(bar1_size,pow(2,10))==0)
      str=std::to_string((int)round(bar1_size/pow(2,10)))+" KB";
    msgs1 = "[" + action_name + "] " + " smqt bar1_size      " + std::to_string(bar1_size) + " (" + str+ ")";
    
    if (remainder(bar2_size,pow(2,30))==0)
      str=std::to_string((int)round(bar2_size/pow(2,30)))+" GB";
    else if (remainder(bar2_size,pow(2,20))==0)
      str=std::to_string((int)round(bar2_size/pow(2,20)))+" MB";
    else if (remainder(bar2_size,pow(2,10))==0)
      str=std::to_string((int)round(bar2_size/pow(2,10)))+" KB";
    msgs2 = "[" + action_name + "] " + " smqt bar2_size      " + std::to_string(bar2_size) + " (" + str+ ")"; 
    
    if (remainder(bar4_size,pow(2,30))==0)
      str=std::to_string((int)round(bar4_size/pow(2,30)))+" GB";
    else if (remainder(bar4_size,pow(2,20))==0)
      str=std::to_string((int)round(bar4_size/pow(2,20)))+" MB";
    else if (remainder(bar4_size,pow(2,10))==0)
      str=std::to_string((int)round(bar4_size/pow(2,10)))+" KB";
    
    msgs4 = "[" + action_name + "] " + " smqt bar4_size      " + std::to_string(bar4_size) + " (" + str+ ")";  
    
    if (remainder(bar5_size,pow(2,30))==0)
      str=std::to_string((int)round(bar5_size/pow(2,30)))+" GB";
    else if (remainder(bar5_size,pow(2,20))==0)
      str=std::to_string((int)round(bar5_size/pow(2,20)))+" MB";
    else if (remainder(bar5_size,pow(2,10))==0)
      str=std::to_string((int)round(bar5_size/pow(2,10)))+" KB";
    msgs5 = "[" + action_name + "] " + " smqt bar5_size      " + std::to_string(bar5_size) + " (" + str+ ")";
    sprintf(hex_value, "%lX", bar1_base_addr);  
    msga1 = "[" + action_name + "] " + " smqt bar1_base_addr " + hex_value;
    sprintf(hex_value, "%lX", bar2_base_addr);   
    msga2 = "[" + action_name + "] " + " smqt bar2_base_addr " + hex_value;
    sprintf(hex_value, "%lX", bar4_base_addr); 
    msga4 = "[" + action_name + "] " + " smqt bar4_base_addr " + hex_value;
    pmsg = "[" + action_name + "] " + " smqt " + std::to_string(pass);
    void* r = rvs::lp::LogRecordCreate("SMQT", action_name.c_str(), rvs::loginfo, sec, usec);
    rvs::lp::Log( msgs1.c_str(), rvs::loginfo, sec, usec);
    rvs::lp::Log( msga1.c_str(), rvs::loginfo, sec, usec);
    rvs::lp::Log( msgs2.c_str(), rvs::loginfo, sec, usec);
    rvs::lp::Log( msga2.c_str(), rvs::loginfo, sec, usec);
    rvs::lp::Log( msgs4.c_str(), rvs::loginfo, sec, usec);
    rvs::lp::Log( msga4.c_str(), rvs::loginfo, sec, usec);
    rvs::lp::Log( msgs5.c_str(), rvs::loginfo, sec, usec);
    rvs::lp::Log( pmsg.c_str(), rvs::logresults);    
    rvs::lp::AddString(r, "bar1_size", std::to_string(bar1_size));
    rvs::lp::AddString(r, "bar1_base_addr", std::to_string(bar1_base_addr));
    rvs::lp::AddString(r, "bar2_size", std::to_string(bar2_size));
    rvs::lp::AddString(r, "bar2_base_addr", std::to_string(bar2_base_addr));
    rvs::lp::AddString(r, "bar4_size", std::to_string(bar4_size));
    rvs::lp::AddString(r, "bar4_base_addr", std::to_string(bar4_base_addr));
    rvs::lp::AddString(r, "bar5_size", std::to_string(bar4_size));
    rvs::lp::AddString(r, "pass", std::to_string(pass));
    rvs::lp::LogRecordFlush(r);
    
  } 
    
  return pass;
    
}