#ifdef __cplusplus
extern "C" {
#endif
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <pci/pci.h>
#include <linux/pci.h>
#include <math.h>
#ifdef __cplusplus
}
#endif

#define DRIVER_BUF_SIZE 1024

extern "C"
{
/**
 * gets the offset (within the PCI related regs) of a given PCI capability (e.g.: PCI_CAP_ID_EXP)
 * @param dev a pci_dev structure containing the PCI device information
 * @param cap a PCI capability (e.g.: PCI_CAP_ID_EXP) All the capabilities are detailed in <pci_regs.h>
 * @return capability offset
 */
unsigned char pci_dev_find_cap_offset(struct pci_dev *dev, unsigned char cap)
{
    int id;
    int max_cap = 48;
    int pos = PCI_CAPABILITY_LIST;
    int status;

    status = pci_read_byte(dev, PCI_STATUS);
    if ( (status & PCI_STATUS_CAP_LIST) == 0 )
        return 0;

    while ( max_cap-- ){
        pos = pci_read_byte(dev, pos);
        if ( pos < 0x40 )
            break;

        pos &= ~3;
        id = pci_read_byte(dev, pos + PCI_CAP_LIST_ID);

        if ( id == 0xff )
            break;
        if ( id == cap )
            return pos;

        pos += PCI_CAP_LIST_NEXT;
    }

    return 0;
}

/**
 * gets the max link speed
 * @param dev a pci_dev structure containing the PCI device information
 * @param buff pre-allocated char buffer
 * @return 
 */
void get_link_cap_max_speed(struct pci_dev *dev, char *buff)
{
    //get pci dev capabilities offset
    unsigned char cap_offset = pci_dev_find_cap_offset(dev, PCI_CAP_ID_EXP);

    if(cap_offset != 0){
        unsigned int pci_dev_lnk_cap = pci_read_long(dev, cap_offset + PCI_EXP_LNKCAP);

        //using 1,2,3 & 4 instead of the dedicated constants (that can be found in pci_regs.h) is the ugly way of doing it
        //however, this is because in some linux versions the #define stops at PCI_EXP_LNKCAP_SLS_5_0GB (no 8, 16)        
        switch (pci_dev_lnk_cap & PCI_EXP_LNKCAP_SLS) {
            case 1:
                strcpy(buff, "2.5 GT/s");
                break;
            case 2:
                strcpy(buff, "5 GT/s");
                break;
            case 3:
                strcpy(buff, "8 GT/s");
                break;
            case 4:
                strcpy(buff, "16 GT/s");
                break;
            default:
                strcpy(buff, "Unknown speed");
        }
    }
    else        
        strcpy(buff,  "");
}

/**
 * gets the PCI dev max link width
 * @param dev a pci_dev structure containing the PCI device information
 * @param buff pre-allocated char buffer
 * @return 
 */
void get_link_cap_max_width(struct pci_dev *dev, char *buff)
{
    //get pci dev capabilities offset
    unsigned char cap_offset = pci_dev_find_cap_offset(dev, PCI_CAP_ID_EXP);

    if(cap_offset != 0){
        unsigned int pci_dev_lnk_cap = pci_read_long(dev, cap_offset + PCI_EXP_LNKCAP);
        sprintf(buff, "x%d", ((pci_dev_lnk_cap & PCI_EXP_LNKCAP_MLW) >> 4)); //TODO: check if "x" is needed here
    }
    else
        strcpy(buff, "x0"); //TODO: check if "x" is needed here
}

/**
 * gets the current link speed
 * @param dev a pci_dev structure containing the PCI device information
 * @param buff pre-allocated char buffer
 * @return 
 */
void get_link_stat_cur_speed(struct pci_dev *dev, char *buff)
{
    //get pci dev capabilities offset
    unsigned char cap_offset = pci_dev_find_cap_offset(dev, PCI_CAP_ID_EXP);

    if(cap_offset != 0){
        unsigned short int pci_dev_lnk_stat = pci_read_word(dev, cap_offset + PCI_EXP_LNKSTA); 

        switch (pci_dev_lnk_stat & PCI_EXP_LNKSTA_CLS) {
            case PCI_EXP_LNKSTA_CLS_2_5GB:
                strcpy(buff, "2.5 GT/s");
                break;
            case PCI_EXP_LNKSTA_CLS_5_0GB:
                strcpy(buff, "5 GT/s");
                break;
            case PCI_EXP_LNKSTA_CLS_8_0GB:
                strcpy(buff, "8 GT/s");
                break;
#ifdef PCI_EXP_LNKSTA_CLS_16_0GB
            case PCI_EXP_LNKSTA_CLS_16_0GB:
                strcpy(buff, "16 GT/s");
                break;
#endif
            default:
                strcpy(buff, "Unknown speed");
        }
    }
    else
        strcpy(buff, "");
}

/**
 * gets the negotiated link width
 * @param dev a pci_dev structure containing the PCI device information
 * @param buff pre-allocated char buffer
 * @return 
 */
void get_link_stat_neg_width(struct pci_dev *dev, char *buff)
{
    //get pci dev capabilities offset
    unsigned char cap_offset = pci_dev_find_cap_offset(dev, PCI_CAP_ID_EXP);

    if(cap_offset != 0){
        unsigned short int pci_dev_lnk_stat = pci_read_word(dev, cap_offset + PCI_EXP_LNKSTA);
        sprintf(buff, "x%d", ((pci_dev_lnk_stat & PCI_EXP_LNKSTA_NLW) >> PCI_EXP_LNKSTA_NLW_SHIFT)); //TODO: check if "x" is needed here
    }
    else
        strcpy(buff, "x0"); //TODO: check if "x" is needed here
}

/**
 * gets the power limit value
 * @param dev a pci_dev structure containing the PCI device information
 * @param buff pre-allocated char buffer
 * @return 
 */
void get_slot_pwr_limit_value(struct pci_dev *dev, char *buff)
{
    //get pci dev capabilities offset
    unsigned char cap_offset = pci_dev_find_cap_offset(dev, PCI_CAP_ID_EXP);
    float pwr;

    if(cap_offset != 0){
        unsigned int slot_cap = pci_read_long(dev, cap_offset + PCI_EXP_SLTCAP);
        unsigned char slot_pwr_limit_scale = (slot_cap & PCI_EXP_SLTCAP_SPLS) >> 15;
        unsigned short int slot_pwr_limit_value = (slot_cap & PCI_EXP_SLTCAP_SPLV) >> 7;

        if(slot_pwr_limit_value > 0xEF){
            switch (slot_pwr_limit_value){
            //according to the PCI Express Base Specification Revision 3.0
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
                pwr = -1.0; //TODO see about returning -1 (F3h to FFh = Reserved for Slot Power Limit values above 300W)
            }
        }
        else{
            //according to the PCI Express Base Specification Revision 3.0
            pwr = (float)(slot_pwr_limit_value) * pow(10, -slot_pwr_limit_scale);
        }
    }
    else
        pwr = -1.0;

    sprintf(buff, "%0.3fW", pwr); //TODO: check if "W" is needed here
}

/**
 * gets PCI dev physical slot number
 * @param dev a pci_dev structure containing the PCI device information
 * @param buff pre-allocated char buffer 
 * @return 
 */
void get_slot_physical_num(struct pci_dev *dev, char *buff)
{
    //get pci dev capabilities offset
    unsigned char cap_offset = pci_dev_find_cap_offset(dev, PCI_CAP_ID_EXP);

    if(cap_offset != 0){
        unsigned int slot_cap = pci_read_long(dev, cap_offset + PCI_EXP_SLTCAP);
        sprintf(buff, "#%u", ((slot_cap & PCI_EXP_SLTCAP_PSN) >> 19)); //TODO: check if "#" is needed here
    }
    else
        strcpy(buff, "#0"); //TODO: check if "#" is needed here
}

/**
 * gets PCI device id (it appears as a function just to keep compatibility with the array of pointer to function)
 * @param dev a pci_dev structure containing the PCI device information
 * @param buff pre-allocated char buffer 
 * @return 
 */
void get_device_id(struct pci_dev *dev, char *buff)
{
    sprintf(buff, "%u", dev->device_id);    
}

/**
 * gets PCI device's vendor id (it appears as a function just to keep compatibility with the array of pointer to function)
 * @param dev a pci_dev structure containing the PCI device information
 * @param buff pre-allocated char buffer 
 * @return 
 */
void get_vendor_id(struct pci_dev *dev, char *buff)
{
    sprintf(buff, "%u", dev->vendor_id);    
}

/**
 * gets the PCI dev driver name
 * @param dev a pci_dev structure containing the PCI device information
 * @param buf pre-allocated char buffer
 * @return 
 */
void get_kernel_driver(struct pci_dev *dev, char *buff)
{
    char name[1024], *drv, *base;
    int n;

    if (dev->access->method != PCI_ACCESS_SYS_BUS_PCI){
        strcpy(buff, "");
        return;
    }

    base = pci_get_param(dev->access, (char *)"sysfs.path");
    if (!base || !base[0]){
        strcpy(buff, "");
        return;
    }

    n = snprintf(name, sizeof(name), "%s/devices/%04x:%02x:%02x.%d/driver", base, dev->domain, dev->bus, dev->dev, dev->func);
    if (n < 0 || n >= (int)sizeof(name)){
        strcpy(buff, "");
        return;
    }

    n = readlink(name, buff, DRIVER_BUF_SIZE);
    if (n < 0){
        strcpy(buff, "");
        return;
    }

    if (n >= DRIVER_BUF_SIZE){
        strcpy(buff, "");
        return;        
    }
    
    buff[n] = 0;

    if (drv = strrchr(buff, '/'))
        strcpy(buff, drv + 1);
}

}


