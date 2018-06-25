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
#include <string>
#include <vector>
#include <map>

#include "gpu_util.h"
#include "rvs_util.h"
#include "rvsliblogger.h"
#include "rvs_module.h"
#include "rvsloglp.h"
#include "action.h"

using std::vector;
using std::string;
using std::map;

/**
 * default class constructor
 */
action::action() {
}

/**
 * class destructor
 */
action::~action() {
    property.clear();
}

/**
 * adds a (key, value) pair to the module's properties collection
 * @param Key one of the keys specified in the RVS SRS
 * @param Val key's value
 * @return add result
 */
int action::property_set(const char* Key, const char* Val) {
    return rvs::lib::actionbase::property_set(Key, Val);
}

/**
 * runs the whole GST logic
 * @return run result
 */
int action::run(void) {
    string msg;
    map<string, string>::iterator it;  // module's properties map iterator

    msg =  "GST in action!";
    log(msg.c_str(), rvs::loginfo);

    return 0;
}