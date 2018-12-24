/*******************************************************************************
 *
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
#ifndef GPUP_SO_INCLUDE_ACTION_H_
#define GPUP_SO_INCLUDE_ACTION_H_

#include <vector>
#include <string>

#include "include/rvsactionbase.h"

using std::vector;
using std::string;
/**
 * @class gpup_action
 * @ingroup GPUP
 *
 * @brief GPUP action implementation class
 *
 * Derives from rvs::actionbase and implements actual action functionality
 * in its run() method.
 *
 */

class gpup_action : public rvs::actionbase {
 public:
    gpup_action();
    virtual ~gpup_action();

    virtual int run(void);

 protected:
    //! the list of all gpu_id in the 'device' property
    vector<string> gpus_id;
    //! the list of properties that are in query
    vector<string> property_name;
    //! the list of io_links properties that are in query
    vector<string> io_link_property_name;
    //! property names to validate
    vector<string> property_name_validate;

    //! 'true' for JSON logging
    bool bjson;
    //! ptr to JSON root node
    void* json_root_node;

    // split properties and io_links properties
    bool property_split(string prop);
    // get properties values
    int property_get_value(uint16_t gpu_id);
    // get io links properties values
    int property_io_links_get_value(uint16_t gpu_id);
    // validate property names
    int validate_property_name(const std::string& name);
};

#endif  // GPUP_SO_INCLUDE_ACTION_H_
