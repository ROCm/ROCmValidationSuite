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
#ifndef INCLUDE_PCI_CAPS_H_
#define INCLUDE_PCI_CAPS_H_

#ifdef __cplusplus
extern "C" {
#endif

unsigned int pci_dev_find_cap_offset(struct pci_dev *dev,
                                     unsigned char cap, unsigned char type);
void get_link_cap_max_speed(struct pci_dev *dev, char *buf);
void get_link_cap_max_width(struct pci_dev *dev, char *buff);
void get_link_stat_cur_speed(struct pci_dev *dev, char *buff);
void get_link_stat_neg_width(struct pci_dev *dev, char *buff);
void get_slot_pwr_limit_value(struct pci_dev *dev, char *buff);
void get_slot_physical_num(struct pci_dev *dev, char *buff);
void get_pci_bus_id(struct pci_dev *dev, char *buff);
void get_device_id(struct pci_dev *dev, char *buff);
void get_dev_serial_num(struct pci_dev *dev, char *buff);
void get_vendor_id(struct pci_dev *dev, char *buff);
void get_kernel_driver(struct pci_dev *dev, char *buff);
void get_pwr_budgeting(struct pci_dev *dev, uint8_t pb_pm_state,
                       uint8_t pb_type, uint8_t pb_power_rail, char *buff);
void get_pwr_curr_state(struct pci_dev *dev, char *buff);
void get_atomic_op_routing(struct pci_dev *dev, char *buff);
void get_atomic_op_32_completer(struct pci_dev *dev, char *buff);
void get_atomic_op_64_completer(struct pci_dev *dev, char *buff);
void get_atomic_op_128_CAS_completer(struct pci_dev *dev, char *buff);
int64_t get_atomic_op_register_value(struct pci_dev *dev);

#ifdef __cplusplus
}
#endif


#endif  // INCLUDE_PCI_CAPS_H_
