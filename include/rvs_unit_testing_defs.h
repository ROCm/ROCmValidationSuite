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

namespace rvs {

// return value for pci_read_long
extern u32 rvs_pci_read_long_return_value;

// return value for pci_read_word
extern u16 rvs_pci_read_word_return_value;

u16 rvs_pci_read_word(struct pci_dev *, int pos) PCI_ABI;
u32 rvs_pci_read_long(struct pci_dev *, int pos) PCI_ABI;

}

#endif  // INCLUDE_RVS_UNIT_TESTING_DEFS_H_