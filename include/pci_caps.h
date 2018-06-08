#ifndef _PCI_CAP_H_
#define _PCI_CAP_H_

#ifdef __cplusplus
extern "C" {
#endif

unsigned int pci_dev_find_cap_offset(struct pci_dev *dev, unsigned char cap);
void get_link_cap_max_speed(struct pci_dev *dev, char *buf);
void get_link_cap_max_width(struct pci_dev *dev, char *buff);
void get_link_stat_cur_speed(struct pci_dev *dev, char *buff);
void get_link_stat_neg_width(struct pci_dev *dev, char *buff);
void get_slot_pwr_limit_value(struct pci_dev *dev, char *buff);
void get_slot_physical_num(struct pci_dev *dev, char *buff);
void get_device_id(struct pci_dev *dev, char *buff);
void get_dev_serial_num(struct pci_dev *dev, char *buff);
void get_vendor_id(struct pci_dev *dev, char *buff);
void get_kernel_driver(struct pci_dev *dev, char *buff);
void get_pwr_base_pwr(struct pci_dev *dev, char *buff);
void get_pwr_rail_type(struct pci_dev *dev, char *buff);
void get_atomic_op_completer(struct pci_dev *dev, char *buff);

#ifdef __cplusplus
}
#endif


#endif // _PCI_CAP_H_