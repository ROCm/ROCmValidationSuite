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

#ifdef __cplusplus
extern "C" {
#endif
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <pci/pci.h>
#include <linux/pci.h>
#include <math.h>
#include <stdint.h>
#ifdef __cplusplus
}
#endif

#define PCI_CAP_DATA_MAX_BUF_SIZE 1024
#define PCI_CAP_NOT_SUPPORTED "NOT SUPPORTED"
#define MEM_BAR_MAX_INDEX 5

#ifdef RVS_UNIT_TEST
  #include "include/rvs_unit_testing_defs.h"
  #define pci_read_long rvs_pci_read_long
  #define pci_read_word rvs_pci_read_word
  #define pci_get_param rvs_pci_get_param
  #define readlink rvs_readlink
  #define pci_write_byte rvs_pci_write_byte
  using rvs::rvs_pci_read_long;
  using rvs::rvs_pci_read_word;
  using rvs::rvs_pci_get_param;
  using rvs::rvs_readlink;
  using rvs::rvs_pci_write_byte;
#endif

extern "C" {

/**
 * gets the offset (within the PCI related regs) of a given PCI capability (e.g.: PCI_CAP_ID_EXP)
 * @param dev a pci_dev structure containing the PCI device information
 * @param cap a PCI capability (e.g.: PCI_CAP_ID_EXP) All the capabilities are detailed in <pci_regs.h>
 * @param type capability type
 * @return capability offset
 */
unsigned int pci_dev_find_cap_offset(struct pci_dev *dev, unsigned char cap,
        unsigned char type) {
    struct pci_cap * pcap = dev->first_cap;
    while (pcap != NULL) {
        if (pcap->id == cap && pcap->type == type)
            return pcap->addr;
        pcap = pcap->next;
    }

    return 0;
}

/**
 * gets the max link speed
 * @param dev a pci_dev structure containing the PCI device information
 * @param buff pre-allocated char buffer
 * @return 
 */
void get_link_cap_max_speed(struct pci_dev *dev, char *buff) {
    const char *link_max_speed;

    // get pci dev capabilities offset
    unsigned int cap_offset = pci_dev_find_cap_offset(dev, PCI_CAP_ID_EXP,
    PCI_CAP_NORMAL);

    if (cap_offset != 0) {
        unsigned int pci_dev_lnk_cap = pci_read_long(dev,
                cap_offset + PCI_EXP_LNKCAP);

        // using 1,2,3 & 4 instead of the dedicated constants
        // (that can be found in pci_regs.h) is the ugly way of doing it
        // however, this is because in some linux versions the #define stops
        // at PCI_EXP_LNKCAP_SLS_5_0GB (no 8, 16)
        switch (pci_dev_lnk_cap & PCI_EXP_LNKCAP_SLS) {
        case 1:
            link_max_speed = "2.5 GT/s";
            break;
        case 2:
            link_max_speed = "5 GT/s";
            break;
        case 3:
            link_max_speed = "8 GT/s";
            break;
        case 4:
            link_max_speed = "16 GT/s";
            break;
        default:
            link_max_speed = "Unknown speed";
        }

        snprintf(buff, PCI_CAP_DATA_MAX_BUF_SIZE, "%s", link_max_speed);
    } else {
      snprintf(buff, PCI_CAP_DATA_MAX_BUF_SIZE, "%s", PCI_CAP_NOT_SUPPORTED);
    }
}

/**
 * gets the PCI dev max link width
 * @param dev a pci_dev structure containing the PCI device information
 * @param buff pre-allocated char buffer
 * @return 
 */
void get_link_cap_max_width(struct pci_dev *dev, char *buff) {
    // get pci dev capabilities offset
    unsigned int cap_offset = pci_dev_find_cap_offset(dev, PCI_CAP_ID_EXP,
    PCI_CAP_NORMAL);

    if (cap_offset != 0) {
        unsigned int pci_dev_lnk_cap = pci_read_long(dev,
                cap_offset + PCI_EXP_LNKCAP);
        snprintf(buff, PCI_CAP_DATA_MAX_BUF_SIZE, "x%d",
                ((pci_dev_lnk_cap & PCI_EXP_LNKCAP_MLW) >> 4));
    } else {
      snprintf(buff, PCI_CAP_DATA_MAX_BUF_SIZE, "%s", PCI_CAP_NOT_SUPPORTED);
    }
}

/**
 * gets the current link speed
 * @param dev a pci_dev structure containing the PCI device information
 * @param buff pre-allocated char buffer
 * @return 
 */
void get_link_stat_cur_speed(struct pci_dev *dev, char *buff) {
    const char *link_cur_speed;

    // get pci dev capabilities offset
    unsigned int cap_offset = pci_dev_find_cap_offset(dev, PCI_CAP_ID_EXP,
    PCI_CAP_NORMAL);

    if (cap_offset != 0) {
        u16 pci_dev_lnk_stat = pci_read_word(dev, cap_offset + PCI_EXP_LNKSTA);

        switch (pci_dev_lnk_stat & PCI_EXP_LNKSTA_CLS) {
        case PCI_EXP_LNKSTA_CLS_2_5GB:
            link_cur_speed = "2.5 GT/s";
            break;
        case PCI_EXP_LNKSTA_CLS_5_0GB:
            link_cur_speed = "5 GT/s";
            break;
        case PCI_EXP_LNKSTA_CLS_8_0GB:
            link_cur_speed = "8 GT/s";
            break;
#ifdef PCI_EXP_LNKSTA_CLS_16_0GB
            case PCI_EXP_LNKSTA_CLS_16_0GB:
            link_cur_speed = "16 GT/s";
            break;
#endif
        default:
            link_cur_speed = "Unknown speed";
        }

        snprintf(buff, PCI_CAP_DATA_MAX_BUF_SIZE, "%s", link_cur_speed);
    } else {
      snprintf(buff, PCI_CAP_DATA_MAX_BUF_SIZE, "%s", PCI_CAP_NOT_SUPPORTED);
    }
}

/**
 * gets the negotiated link width
 * @param dev a pci_dev structure containing the PCI device information
 * @param buff pre-allocated char buffer
 * @return 
 */
void get_link_stat_neg_width(struct pci_dev *dev, char *buff) {
    // get pci dev capabilities offset
    unsigned int cap_offset = pci_dev_find_cap_offset(dev, PCI_CAP_ID_EXP,
    PCI_CAP_NORMAL);

    if (cap_offset != 0) {
        u16 pci_dev_lnk_stat = pci_read_word(dev, cap_offset + PCI_EXP_LNKSTA);
        snprintf(buff, PCI_CAP_DATA_MAX_BUF_SIZE, "x%d",
                ((pci_dev_lnk_stat & PCI_EXP_LNKSTA_NLW)
                        >> PCI_EXP_LNKSTA_NLW_SHIFT));
    } else {
      snprintf(buff, PCI_CAP_DATA_MAX_BUF_SIZE, "%s", PCI_CAP_NOT_SUPPORTED);
    }
}

/**
 * gets the power limit value
 * @param dev a pci_dev structure containing the PCI device information
 * @param buff pre-allocated char buffer
 * @return 
 */
void get_slot_pwr_limit_value(struct pci_dev *dev, char *buff) {
    // get pci dev capabilities offset
    unsigned int cap_offset = pci_dev_find_cap_offset(dev, PCI_CAP_ID_EXP,
    PCI_CAP_NORMAL);
    float pwr;

    if (cap_offset != 0) {
        unsigned int slot_cap = pci_read_long(dev, cap_offset + PCI_EXP_SLTCAP);
        unsigned char slot_pwr_limit_scale = (slot_cap & PCI_EXP_SLTCAP_SPLS)
                >> 15;
        u16 slot_pwr_limit_value = (slot_cap & PCI_EXP_SLTCAP_SPLV) >> 7;

        if (slot_pwr_limit_value > 0xEF) {
            switch (slot_pwr_limit_value) {
            // according to the PCI Express Base Specification Revision 3.0
            case 0xF0:
                pwr = 250.0;
                break;
            case 0xF1:
                pwr = 270.0;
                break;
            case 0xF2:
                pwr = 300.0;
                break;
            default:
                pwr = -1.0;
                // (F3h to FFh = Reserved for Slot Power Limit
                // values above 300W)
            }
        } else {
            // according to the PCI Express Base Specification Revision 3.0
            pwr = static_cast<float>(slot_pwr_limit_value)
                    * pow(10, -slot_pwr_limit_scale);
        }

        snprintf(buff, PCI_CAP_DATA_MAX_BUF_SIZE, "%0.3fW", pwr);

    } else {
      snprintf(buff, PCI_CAP_DATA_MAX_BUF_SIZE, "%s", PCI_CAP_NOT_SUPPORTED);
    }
}

/**
 * gets PCI dev physical slot number
 * @param dev a pci_dev structure containing the PCI device information
 * @param buff pre-allocated char buffer 
 * @return 
 */
void get_slot_physical_num(struct pci_dev *dev, char *buff) {
    // get pci dev capabilities offset
    unsigned int cap_offset = pci_dev_find_cap_offset(dev, PCI_CAP_ID_EXP,
    PCI_CAP_NORMAL);

    if (cap_offset != 0) {
        unsigned int slot_cap = pci_read_long(dev, cap_offset + PCI_EXP_SLTCAP);
        snprintf(buff, PCI_CAP_DATA_MAX_BUF_SIZE, "#%u",
                ((slot_cap & PCI_EXP_SLTCAP_PSN) >> 19));
    } else {
        snprintf(buff, PCI_CAP_DATA_MAX_BUF_SIZE, "%s", PCI_CAP_NOT_SUPPORTED);
    }
}

/**
 * gets PCI dev bus id
 * @param dev a pci_dev structure containing the PCI device information
 * @param buff pre-allocated char buffer 
 * @return 
 */
void get_pci_bus_id(struct pci_dev *dev, char *buff) {
  snprintf(buff, PCI_CAP_DATA_MAX_BUF_SIZE, "%0X", dev->bus);
}

/**
 * gets PCI device id (it appears as a function just to keep compatibility with the array of pointer to function)
 * @param dev a pci_dev structure containing the PCI device information
 * @param buff pre-allocated char buffer 
 * @return 
 */
void get_device_id(struct pci_dev *dev, char *buff) {
    snprintf(buff, PCI_CAP_DATA_MAX_BUF_SIZE, "%u", dev->device_id);
}

/**
 * gets PCI device's vendor id (it appears as a function just to keep compatibility with the array of pointer to function)
 * @param dev a pci_dev structure containing the PCI device information
 * @param buff pre-allocated char buffer 
 * @return 
 */
void get_vendor_id(struct pci_dev *dev, char *buff) {
    snprintf(buff, PCI_CAP_DATA_MAX_BUF_SIZE, "%u", dev->vendor_id);
}

/**
 * gets the PCI dev driver name
 * @param dev a pci_dev structure containing the PCI device information
 * @param buf pre-allocated char buffer
 * @return 
 */
void get_kernel_driver(struct pci_dev *dev, char *buff) {
    char name[1024], *drv, *base;
    int n;

    buff[0] = '\0';

    if (dev->access->method != PCI_ACCESS_SYS_BUS_PCI) {
        return;
    }

    base = pci_get_param(dev->access, const_cast<char *>("sysfs.path"));
    if (!base || !base[0]) {
        return;
    }

    n = snprintf(name, sizeof(name), "%s/devices/%04x:%02x:%02x.%d/driver",
            base, dev->domain, dev->bus, dev->dev, dev->func);
    if (n < 0 || n >= static_cast<int>(sizeof(name))) {
        return;
    }

    n = readlink(name, buff, PCI_CAP_DATA_MAX_BUF_SIZE);
    if (n < 0) {
        return;
    }

    if (n >= PCI_CAP_DATA_MAX_BUF_SIZE) {
        return;
    }

    buff[n] = 0;

    if ((drv = strrchr(buff, '/')) != NULL)
        snprintf(buff, PCI_CAP_DATA_MAX_BUF_SIZE, "%s", drv + 1);
}

/**
 * gets the device serial number
 * @param dev a pci_dev structure containing the PCI device information
 * @param buf pre-allocated char buffer
 */
void get_dev_serial_num(struct pci_dev *dev, char *buff) {
    unsigned int cap_offset_dsn = pci_dev_find_cap_offset(dev,
    PCI_EXT_CAP_ID_DSN, PCI_CAP_EXTENDED);

    if (cap_offset_dsn != 0) {
        unsigned int t1, t2;
        t1 = pci_read_long(dev, cap_offset_dsn + 4);
        t2 = pci_read_long(dev, cap_offset_dsn + 8);
        snprintf(buff, PCI_CAP_DATA_MAX_BUF_SIZE,
                "%02x-%02x-%02x-%02x-%02x-%02x-%02x-%02x", t2 >> 24,
                (t2 >> 16) & 0xff, (t2 >> 8) & 0xff, t2 & 0xff, t1 >> 24,
                (t1 >> 16) & 0xff, (t1 >> 8) & 0xff, t1 & 0xff);
    } else {
      snprintf(buff, PCI_CAP_DATA_MAX_BUF_SIZE, "%s", PCI_CAP_NOT_SUPPORTED);
    }
}

/**
 * gets the device power budgeting capabilities
 * @param dev a pci_dev structure containing the PCI device information
 * @param pb_pm_state the PM State for the given operating condition
 * @param pb_type the type of the given operating condition
 * @param pb_power_rail thermal load or power rail for the given operating condition
 * @param buf pre-allocated char buffer
 */
void get_pwr_budgeting(struct pci_dev *dev, uint8_t pb_pm_state,
                       uint8_t pb_type, uint8_t pb_power_rail, char *buff) {
    u16 i, w = 0;
    u16 base, scale;
    uint8_t pb_act_pm_state, pb_act_type, pb_act_power_rail;

    unsigned int cap_offset_pwbgd = pci_dev_find_cap_offset(dev,
    PCI_EXT_CAP_ID_PWR, PCI_CAP_EXTENDED);

    snprintf(buff, PCI_CAP_DATA_MAX_BUF_SIZE, "%s", PCI_CAP_NOT_SUPPORTED);

    if (cap_offset_pwbgd != 0) {
        i = 0;

        do {
            pci_write_byte(dev, cap_offset_pwbgd + PCI_PWR_DSR, i);
            w = pci_read_word(dev, cap_offset_pwbgd + PCI_PWR_DATA);

            if (!w)
                return;

            pb_act_pm_state = PCI_PWR_DATA_PM_STATE(w);
            pb_act_type = PCI_PWR_DATA_TYPE(w);
            pb_act_power_rail = PCI_PWR_DATA_RAIL(w);

            if (pb_act_pm_state == pb_pm_state && pb_act_type == pb_type &&
                                        pb_act_power_rail == pb_power_rail) {
                base = PCI_PWR_DATA_BASE(w);
                scale = PCI_PWR_DATA_SCALE(w);
                snprintf(buff, PCI_CAP_DATA_MAX_BUF_SIZE, "%.3fW",
                        base * pow(10, -scale));
                return;
            }

            i++;
        } while (1);
    }
}

/**
 * Get current power state
 * @param dev a pci_dev structure containing the PCI device information
 * @param buf pre-allocated char buffer
 */
void get_pwr_curr_state(struct pci_dev *dev, char *buff) {
  u16 pmcsr;
  const char *type_s;

  // init output buffer with "not supported" message
  snprintf(buff, PCI_CAP_DATA_MAX_BUF_SIZE, "%s", PCI_CAP_NOT_SUPPORTED);

  // fetch capability offset
  unsigned int cap_offset = pci_dev_find_cap_offset(dev,
    PCI_CAP_ID_PM, PCI_CAP_NORMAL);

  if (cap_offset == 0)
    return;

  pmcsr = pci_read_word(dev, cap_offset + PCI_PM_CTRL);

  switch (pmcsr & PCI_PM_CTRL_STATE_MASK) {
  case 0:
      type_s = "D0";
      break;
  case 1:
      type_s = "D1";
      break;
  case 2:
      type_s = "D2";
      break;
  case 3:
      type_s = "D3";
      break;
  }

  snprintf(buff, PCI_CAP_DATA_MAX_BUF_SIZE, "%s", type_s);
}

/**
 * gets the device atomic requester capabilities
 * @param dev a pci_dev structure containing the PCI device information
 * @param buf pre-allocated char buffer
 */
void get_atomic_op_routing(struct pci_dev *dev, char *buff) {
    bool atomic_op_routing_enable = false;

    // get pci dev capabilities offset
    unsigned int cap_offset = pci_dev_find_cap_offset(dev, PCI_CAP_ID_EXP,
    PCI_CAP_NORMAL);

    if (cap_offset != 0) {
        // get Capability version via
        // PCI Express Capabilities Register (offset 02h)
        u16 cap_flags = pci_read_word(dev, cap_offset + 2);

        // check if it's capability version 2
        if (!((cap_flags & PCI_EXP_FLAGS_VERS) < 2)) {
            u16 dev_ctl2_reg_val = pci_read_word(dev,
                    cap_offset + PCI_EXP_DEVCTL2);

            // hardcoded 0x0040 because PCI_EXP_DEVCTL2_ATOMIC_REQ
            // is not present on all versions of pci_regs.h
            atomic_op_routing_enable = static_cast<bool>(dev_ctl2_reg_val
                    & 0x0040);

            snprintf(buff, PCI_CAP_DATA_MAX_BUF_SIZE, "%s",
                    atomic_op_routing_enable ? "TRUE" : "FALSE");
        } else {
          snprintf(buff, PCI_CAP_DATA_MAX_BUF_SIZE, "%s",
              PCI_CAP_NOT_SUPPORTED);
        }
    } else {
      snprintf(buff, PCI_CAP_DATA_MAX_BUF_SIZE, "%s", PCI_CAP_NOT_SUPPORTED);
    }
}

/**
 * gets the device atomic capabilities register value
 * @param dev a pci_dev structure containing the PCI device information
 */
int64_t get_atomic_op_register_value(struct pci_dev *dev) {
    unsigned char i, has_memory_bar = 0;

    // get pci dev capabilities offset
    unsigned int cap_offset = pci_dev_find_cap_offset(dev, PCI_CAP_ID_EXP,
    PCI_CAP_NORMAL);

    if (cap_offset != 0) {
        // get Capability version via
        // PCI Express Capabilities Register (offset 02h)
        u16 cap_flags = pci_read_word(dev, cap_offset + 2);

        // check if it's capability version 2
        if (!((cap_flags & PCI_EXP_FLAGS_VERS) < 2)) {
            // check if the device has memory space BAR
            // (basically it should have but let us be sure about it)
            for (i = 0; i < MEM_BAR_MAX_INDEX + 1; i++)
                if (dev->base_addr[i] && dev->size[i]) {
                    if (!(dev->base_addr[i] & PCI_BASE_ADDRESS_SPACE_IO)) {
                        has_memory_bar = 1;
                        break;
                    }
                }

            if (has_memory_bar) {
                return pci_read_long(dev, cap_offset + PCI_EXP_DEVCAP2);

            } else {
              return -1;
            }
        } else {
          return -1;
        }
    } else {
      return -1;
    }
}

/**
 * gets the device atomic 32-bit completer capabilities
 * @param dev a pci_dev structure containing the PCI device information
 * @param buf pre-allocated char buffer
 */
void get_atomic_op_32_completer(struct pci_dev *dev, char *buff) {
  int64_t atomic_op_completer_value;
  bool atomic_op_completer_supported_32_bit = false;

  atomic_op_completer_value = get_atomic_op_register_value(dev);

  if (atomic_op_completer_value != -1) {
        atomic_op_completer_supported_32_bit =
            static_cast<bool>(static_cast
                <unsigned int>(atomic_op_completer_value) & 0x0080);

        snprintf(buff, PCI_CAP_DATA_MAX_BUF_SIZE, "%s",
                atomic_op_completer_supported_32_bit ?
                    "TRUE" : "FALSE");
  } else {
    snprintf(buff, PCI_CAP_DATA_MAX_BUF_SIZE, "%s", PCI_CAP_NOT_SUPPORTED);
  }
}

/**
 * gets the device atomic 64-bit completer capabilities
 * @param dev a pci_dev structure containing the PCI device information
 * @param buf pre-allocated char buffer
 */
void get_atomic_op_64_completer(struct pci_dev *dev, char *buff) {
  int64_t atomic_op_completer_value;
  bool atomic_op_completer_supported_64_bit = false;

  atomic_op_completer_value = get_atomic_op_register_value(dev);

  if (atomic_op_completer_value != -1) {
        atomic_op_completer_supported_64_bit =
                static_cast<bool>(static_cast
                    <unsigned int>(atomic_op_completer_value) & 0x0100);

        snprintf(buff, PCI_CAP_DATA_MAX_BUF_SIZE, "%s",
                atomic_op_completer_supported_64_bit ?
                    "TRUE" : "FALSE");
  } else {
    snprintf(buff, PCI_CAP_DATA_MAX_BUF_SIZE, "%s", PCI_CAP_NOT_SUPPORTED);
  }
}

/**
 * gets the device atomic 128-bit CAS completer capabilities
 * @param dev a pci_dev structure containing the PCI device information
 * @param buf pre-allocated char buffer
 */
void get_atomic_op_128_CAS_completer(struct pci_dev *dev, char *buff) {
  int64_t atomic_op_completer_value;
  bool atomic_op_completer_supported_128_bit_CAS = false;

  atomic_op_completer_value = get_atomic_op_register_value(dev);

  if (atomic_op_completer_value != -1) {
    atomic_op_completer_supported_128_bit_CAS =
                static_cast<bool>(static_cast
                    <unsigned int>(atomic_op_completer_value) & 0x0200);

        snprintf(buff, PCI_CAP_DATA_MAX_BUF_SIZE, "%s",
            atomic_op_completer_supported_128_bit_CAS ?
                    "TRUE" : "FALSE");
  } else {
    snprintf(buff, PCI_CAP_DATA_MAX_BUF_SIZE, "%s", PCI_CAP_NOT_SUPPORTED);
  }
}

}
