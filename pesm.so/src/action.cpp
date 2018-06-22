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

#include "rvs_module.h"
#include "worker.h"

extern "C"
{
#include <pci/pci.h>
#include <linux/pci.h>
}
#include <iostream>
#include <algorithm>

#include "pci_caps.h"
#include "gpu_util.h"
#include "rvs_module.h"
#include "rvsloglp.h"


using namespace std;

extern const char* pcie_cap_names[];

static Worker* pworker;

action::action() {
}

action::~action() {
  property.clear();
}

int action::run(void) {
  log("[PESM] in run()", rvs::logdebug);

  if (has_property("do_gpu_list")) {
    return do_gpu_list();
  }

  // handle "wait" property
  if (property["wait"] != "") {
    sleep(atoi(property["wait"].c_str()));
  }

  if(property["monitor"] == "true") {
    log("property[\"monitor\"] == \"true\"", rvs::logdebug);

    if (!pworker) {
    log("creating Worker", rvs::logdebug);
      pworker = new Worker();
      pworker->set_name(property["name"]);

      // pas -j flag
      if( property.find("cli.-j") != property.end())
      {
        pworker->json(true);
      }
    }
    log("starting Worker", rvs::logdebug);
    pworker->start();
//     log("detaching Worker", rvs::logdebug);
//     pworker->detach();

    log("[PESM] Monitoring started", rvs::logdebug);
  }
  else {
    log("property[\"monitor\"] != \"true\"", rvs::logdebug);
    if (pworker) {
      pworker->set_stop_name(property["name"]);
      pworker->stop();
      delete pworker;
      pworker = nullptr;
    }
    log("[PESM] Monitoring stopped", rvs::logdebug);
  }
  return 0;
}

int action::do_gpu_list() {
  log("[PESM] in do_gpu_list()", rvs::logdebug);

  std::map<string,string>::iterator it;
  std::vector<unsigned short int>   gpus_location_id;

  struct pci_access* pacc;
  struct pci_dev*    dev;
  char buff[1024];
  char devname[1024];

  //get all GPU location_id (Note: we're not using device_id as the unique identifier
  // because multiple GPUs can have the same ID. We use location_id which is unique and points to the sysfs
  gpu_get_all_location_id(gpus_location_id);

  //get the pci_access structure
  pacc = pci_alloc();
  //initialize the PCI library
  pci_init(pacc);
  //get the list of devices
  pci_scan_bus(pacc);

  bool bheader_printed = false;
  int  ix = 0;
  //iterate over devices
  for (dev = pacc->devices; dev; dev = dev->next) {
    pci_fill_info(dev, PCI_FILL_IDENT | PCI_FILL_BASES | PCI_FILL_CLASS | PCI_FILL_EXT_CAPS | PCI_FILL_CAPS | PCI_FILL_PHYS_SLOT); //fil in the info

    //computes the actual dev's location_id (sysfs entry)
    unsigned short int dev_location_id = ((((unsigned short int)(dev->bus)) << 8) | (dev->func));

    //check if this pci_dev corresponds to one of AMD GPUs
    auto it_gpu = find(gpus_location_id.begin(), gpus_location_id.end(), dev_location_id);

    if (it_gpu == gpus_location_id.end())
      continue;

    if (!bheader_printed) {
      bheader_printed = true;
      cout << "Supported GPUs available:" << endl;
    }

    sprintf(buff, "%02X:%02X.%d", dev->bus, dev->dev, dev->func);

    string name;
    name = pci_lookup_name(pacc, devname, sizeof(devname), PCI_LOOKUP_DEVICE, dev->vendor_id, dev->device_id);

    cout << buff  << " - GPU[" << ix << "] " << name << hex << " (Device " << dev->device_id << ")"<< endl;
    ix++;
  }

  pci_cleanup(pacc);

  if (!bheader_printed) {
    cout << endl << "No supported GPUs available." << endl;
  }

  return 0;
}
