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

#include <queue>

#include "include/rvs_unit_testing_defs.h"

namespace rvs {

// return value for pci_read_long
std::queue<u32> rvs_pci_read_long_return_value;

// return value for pci_read_word
std::queue<u16> rvs_pci_read_word_return_value;

// return value for pci_get_param
char* rvs_pci_get_param_return_value;

// return value for pci_read_word
ssize_t rvs_readlink_return_value = 0;

// return value for pci_get_param
char* rvs_readlink_buff_return_value;

// override function to return rvs_pci_read_long_return_value
u32 rvs_pci_read_long(struct pci_dev *, int pos) PCI_ABI {
  u32 result;
  // fix unused variable warning
  pos = pos;
  if (rvs_pci_read_long_return_value.size() == 1) {
    return rvs_pci_read_long_return_value.front();
  } else {
    result = rvs_pci_read_long_return_value.front();
    rvs_pci_read_long_return_value.pop();
    return result;
  }
}

// override function to return rvs_pci_read_word_return_value
u16 rvs_pci_read_word(struct pci_dev *, int pos) PCI_ABI {
  u16 result;
  // fix unused variable warning
  pos = pos;
  if (rvs_pci_read_word_return_value.size() == 1) {
    return rvs_pci_read_word_return_value.front();
  } else {
    result = rvs_pci_read_word_return_value.front();
    rvs_pci_read_word_return_value.pop();
    return result;
  }
}

// override function to return rvs_pci_get_param_return_value
char* rvs_pci_get_param(struct pci_access *acc, char *param) PCI_ABI {
  // fix unused variable warning
  param = param;
  acc = acc;
  return rvs_pci_get_param_return_value;
}

// override function to return rvs_pci_get_param_return_value
ssize_t rvs_readlink(char* path, char* buf, size_t bufsize) {
  // fix unused variable warning
  path = path;
  bufsize = bufsize;
  for (unsigned int i = 0u; i < sizeof(rvs_readlink_buff_return_value); i++) {
    *(buf + i) = *(rvs_readlink_buff_return_value + i);
  }
  return rvs_readlink_return_value;
}

// override function
int rvs_pci_write_byte(struct pci_dev *, int pos, u8 data) PCI_ABI {
  // fix unused variable warning
  pos = pos;
  data = data;
  return 0;
}

}  // namespace rvs
