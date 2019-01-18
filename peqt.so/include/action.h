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
#ifndef PEQT_SO_INCLUDE_ACTION_H_
#define PEQT_SO_INCLUDE_ACTION_H_

#ifdef __cplusplus
extern "C" {
#endif
#include <pci/pci.h>
#ifdef __cplusplus
}
#endif

#include <vector>
#include <string>
#include <regex>
#include <map>

#include "include/rvsactionbase.h"

using std::vector;
using std::string;
using std::regex;
using std::map;

/**
 * @class peqt_action
 * @ingroup PEQT
 *
 * @brief PEQT action implementation class
 *
 * Derives from rvs::actionbase and implements actual action functionality
 * in its run() method.
 *
 */
class peqt_action: public rvs::actionbase {
 public:
    peqt_action();
    virtual ~peqt_action();

    virtual int run(void);

 protected:
  bool get_all_common_config_keys(void);

 private:
    //! TRUE if JSON output is required
    bool bjson;
    //! JSON root node
    void* json_root_node;

    //! PCI PB (Power Budgeting) <<PM State, encoding>> pairs (according to
    //! PCI_Express_Base_Specification_Revision_3.0)
    map <string, uint8_t> pb_op_pm_states_encodings_map;
    //! PCI PB (Power Budgeting) <<Type, encoding>> pairs
    map <string, uint8_t> pb_op_pm_types_encodings_map;
    //! PCI PB (Power Budgeting) <<Power Rail, encoding>> pairs
    map <string, uint8_t> pb_op_pm_power_rails_encodings_map;
    //! regex for dynamic PB capabilities
    regex pb_dynamic_regex;

    bool get_gpu_all_pcie_capabilities(struct pci_dev *dev, uint16_t gpu_id);


 protected:
};

#endif  // PEQT_SO_INCLUDE_ACTION_H_
