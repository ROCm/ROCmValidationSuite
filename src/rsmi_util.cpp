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
#include "include/rsmi_util.h"

#include <cassert>

namespace rvs {


rsmi_status_t  rsmi_dev_ind_get(uint64_t bdfid, uint32_t* pdv_ind) {
  assert(pdv_ind != nullptr);

  uint32_t ix = 0;
  uint64_t _bdfid = 0;
  uint32_t num_devices = 0;
  rsmi_status_t sts;

  //Initialize
  *pdv_ind = 0;

  if (RSMI_STATUS_SUCCESS != (sts = rsmi_num_monitor_devices(&num_devices))) {
    return sts;
  }

  for (ix = 0; ix < num_devices; ix++) {
    if (RSMI_STATUS_SUCCESS == rsmi_dev_pci_id_get(ix, &_bdfid)) {
      if (_bdfid == bdfid) {
        *pdv_ind = ix;
        return RSMI_STATUS_SUCCESS;
      }
    }
  }
  return RSMI_STATUS_INVALID_ARGS;
}

}  // namespace rvs


