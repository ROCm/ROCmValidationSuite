/********************************************************************************
 *
 * Copyright (c) 2018-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include "include/rsmi_util.h"

#include <cassert>

namespace rvs {
std::map<uint64_t, amdsmi_processor_handle> smipci_to_hdl_map;
amdsmi_status_t smi_pci_hdl_mapping(){
  amdsmi_status_t ret;
   uint64_t _bdfid = 0;
  uint32_t socket_count = 0;
  ret = amdsmi_get_socket_handles(&socket_count, nullptr);
  std::vector<amdsmi_socket_handle> sockets(socket_count);
  ret = amdsmi_get_socket_handles(&socket_count, &sockets[0]);
  for(auto socket : sockets){
    uint32_t device_count = 0;// # of devices in this socket
    ret = amdsmi_get_processor_handles(socket, &device_count, nullptr);
    std::vector<amdsmi_processor_handle> processor_handles(device_count);
    ret = amdsmi_get_processor_handles(socket,
              &device_count, &processor_handles[0]);
    for(auto dev: processor_handles){
      if(AMDSMI_STATUS_SUCCESS == amdsmi_get_gpu_bdf_id(dev, &_bdfid)){
        smipci_to_hdl_map.insert({_bdfid, dev});
      }
    }
  }
  return AMDSMI_STATUS_SUCCESS;
}

	
amdsmi_status_t  rsmi_dev_ind_get(uint64_t bdfid, amdsmi_processor_handle* pdv_hdl) {
  assert(pdv_hdl != nullptr);
  uint64_t _bdfid = 0;
  amdsmi_status_t ret;
  *pdv_hdl = 0;
  smi_pci_hdl_mapping();
  for(auto itr = smipci_to_hdl_map.begin(); itr!=smipci_to_hdl_map.end();++itr){
    if(itr->first == bdfid)
	*pdv_hdl = itr->second;
        return AMDSMI_STATUS_SUCCESS;
  }
   return AMDSMI_STATUS_INVAL;
}

std::map<uint64_t, amdsmi_processor_handle> get_smi_pci_map(){
  return smipci_to_hdl_map;
}

}  // namespace rvs

