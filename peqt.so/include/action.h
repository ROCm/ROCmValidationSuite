// Copyright [year] <Copyright Owner> ... goes here
#ifndef PEQT_SO_INCLUDE_ACTION_H_
#define PEQT_SO_INCLUDE_ACTION_H_

#ifdef __cplusplus
extern "C" {
#endif
#include <pci/pci.h>
#ifdef __cplusplus
}
#endif

#include <vector>
#include <string>

#include "rvslib.h"

using std::vector;
using std::string;

class action: public rvs::lib::actionbase {
 public:
    action();
    virtual ~action();

    virtual int property_set(const char*, const char*);
    virtual int run(void);

 private:
    vector<string> device_prop_gpu_id_list;  // the list of all gpu_id
                                             // in the <device> property

    string action_name;

    // PCIe capabilities stuff
    bool get_gpu_all_pcie_capabilities(struct pci_dev *dev);

    // configuration properties getters
    bool property_get_device(int *error); // gets the device property value (list of gpu_id)
    //from the module's properties collection
    void property_get_action_name(void);  // gets the action name
    int property_get_deviceid(int *error);  // gets the deviceid

 protected:
};

#endif  // PEQT_SO_INCLUDE_ACTION_H_
