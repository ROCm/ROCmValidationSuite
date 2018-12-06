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

#ifndef INCLUDE_RVS_UNIT_TESTING_DEFS_H_
#define INCLUDE_RVS_UNIT_TESTING_DEFS_H_

#include <pci/pci.h>
#include <linux/pci.h>
#include <queue>

namespace rvs {

// return value for pci_read_long
extern std::queue<u32> rvs_pci_read_long_return_value;

// return value for pci_read_word
extern std::queue<u16> rvs_pci_read_word_return_value;

// return value for pci_get_param
extern char* rvs_pci_get_param_return_value;

// return value for pci_read_word
extern ssize_t rvs_readlink_return_value;

// return value for pci_get_param
extern char* rvs_readlink_buff_return_value;

u16 rvs_pci_read_word(struct pci_dev *, int pos) PCI_ABI;
u32 rvs_pci_read_long(struct pci_dev *, int pos) PCI_ABI;
char* rvs_pci_get_param(struct pci_access *acc, char *param) PCI_ABI;
ssize_t rvs_readlink(char* path, char* buf, size_t bufsize);
int rvs_pci_write_byte(struct pci_dev *, int pos, u8 data) PCI_ABI;

}  // namespace rvs

#endif  // INCLUDE_RVS_UNIT_TESTING_DEFS_H_
