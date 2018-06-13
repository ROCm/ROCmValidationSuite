
#include "worker.h"

#include <chrono>
#include <map>
#include <string>
#include <algorithm>
#include <iostream>

#ifdef __cplusplus
extern "C" {
#endif
#include <pci/pci.h>
#include <linux/pci.h>
#ifdef __cplusplus
}
#endif

#include "rvsliblogger.h"
#include "rvs_module.h"
#include "pci_caps.h"
#include "gpu_util.h"
#include "rvsloglp.h"


Worker::Worker() {}
Worker::~Worker() {}

void Worker::run() {
	
	brun = true;
    char buff[1024];

//     void (*arr_prop_pfunc_names[]) (struct pci_dev *dev, char *) = {get_link_cap_max_speed, get_link_cap_max_width, get_link_stat_cur_speed, get_link_stat_neg_width, get_slot_pwr_limit_value, get_slot_physical_num, get_device_id, get_vendor_id, get_kernel_driver, get_dev_serial_num, get_pwr_base_pwr, get_pwr_rail_type, get_atomic_op_completer};

	std::map<string,string>::iterator it;
 	std::vector<unsigned short int> gpus_location_id;
	std::map<unsigned short int, string> old_val;

    struct pci_access *pacc;
    struct pci_dev *dev;

	
	// worker thread has started
	while (brun) {
		log("[PESM] worker thread is running...", rvs::logdebug);
		
		//get all GPU location_id (Note: we're not using device_id as the unique identifier of the GPU because multiple GPUs can have the same ID ... this is also true for the case of the machine where we're working)
		//therefore, what we're using is the location_id which is unique and points to the sysfs
		gpu_get_all_location_id(gpus_location_id);

		//get the pci_access structure
		pacc = pci_alloc();
		//initialize the PCI library
		pci_init(pacc);
		//get the list of devices
		pci_scan_bus(pacc);

		//iterate over devices
		for (dev = pacc->devices; dev; dev = dev->next) {
			pci_fill_info(dev, PCI_FILL_IDENT | PCI_FILL_BASES | PCI_FILL_CLASS | PCI_FILL_EXT_CAPS | PCI_FILL_CAPS | PCI_FILL_PHYS_SLOT); //fil in the info

			//computes the actual dev's location_id (sysfs entry)
			unsigned short int dev_location_id = ((((unsigned short int)(dev->bus)) << 8) | (dev->func));

			//check if this pci_dev corresponds to one of AMD GPUs
			auto it_gpu = find(gpus_location_id.begin(), gpus_location_id.end(), dev_location_id);

			if (it_gpu == gpus_location_id.end())
				continue;
			
			// get current speed for the link
			get_link_stat_cur_speed(dev, buff);
			string new_val(buff);
			
			// same as the previously measured? do nothing just continue
			if (old_val[dev_location_id] == new_val)
				continue;
			
			// new value is different, so store it;
			old_val[dev_location_id] = new_val;
			
			string msg("[" + action_name + "] " + "[PESM] " + std::to_string(dev_location_id) + " link speed change " + new_val);
			log( msg.c_str(), rvs::loginfo);

      void* r = rvs::lp::LogRecordCreate("PESM", action_name);
      rvs::lp::AddString(r, "msg", "link speed change");
      rvs::lp::AddString(r, "val", new_val);
      rvs::lp::LogRecordFlush(r);

		}

		pci_cleanup(pacc);
		
		sleep(1);
		
	}
	
	log("[PESM] worker thread has finished", rvs::logdebug);
	
}

void Worker::stop() {
	
	log("[PESM] in Worker::stop()", rvs::logdebug);
	// reset "run" flag
	brun = false;
	
	// wait a bit to make sure thread has exited
	try {
		if (t.joinable()) 
			t.join();
	}
	catch(...) {
	}
}