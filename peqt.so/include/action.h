#ifndef ACTION_H_
#define ACTION_H_

#include "rvslib.h"

extern "C"
{
unsigned char pci_dev_find_cap_offset(struct pci_dev *dev, unsigned char cap);
void get_link_cap_max_speed(struct pci_dev *dev, char *buf);
void get_link_cap_max_width(struct pci_dev *dev, char *buff);
void get_link_stat_cur_speed(struct pci_dev *dev, char *buff);
void get_link_stat_neg_width(struct pci_dev *dev, char *buff);
void get_slot_pwr_limit_value(struct pci_dev *dev, char *buff);
void get_slot_physical_num(struct pci_dev *dev, char *buff);
void get_device_id(struct pci_dev *dev, char *buff);
void get_vendor_id(struct pci_dev *dev, char *buff);
void get_kernel_driver(struct pci_dev *dev, char *buff);
}


class action : public rvs::lib::actionbase
{
public:
	action();
	virtual ~action();
	
	virtual int property_set(const char*, const char*);
	virtual int run(void);
	
protected:
	
};

#endif /* ACTION_H_ */
