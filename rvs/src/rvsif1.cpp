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
#include "include/rvsif1.h"

#include <string>
#include "include/rvsliblogger.h"

using std::string;

//! Default constructor
rvs::if1::if1()
:
rvs_module_action_property_set(nullptr),
rvs_module_action_run(nullptr) {
}

//! Default descrutor
rvs::if1::~if1() {
}


/**
 * @brief Copy constructor
 *
 * @param rhs reference to RHS instance
 *
 */
rvs::if1::if1(const if1& rhs) : ifbase(rhs) {
  *this = rhs;
}

/**
 * @brief Assignment operator
 *
 * @param rhs reference to RHS instance
 * @return reference to LHS instance
 *
 */
rvs::if1& rvs::if1::operator=(const rvs::if1& rhs) {
  // self-assignment check
  if (this != &rhs) {
    ifbase::operator=(rhs);
    rvs_module_action_property_set = rhs.rvs_module_action_property_set;
    rvs_module_action_run = rhs.rvs_module_action_run;
  }

  return *this;
}

/**
 * @brief Clone instance
 *
 * @return pointer to newly created instance
 *
 */
rvs::ifbase* rvs::if1::clone(void) {
  return new rvs::if1(*this);
}

/**
 * @brief Sets action property
 *
 * @param pKey Property key
 * @param pVal Property value
 * @return 0 - success, non-zero otherwise
 *
 */
int rvs::if1::property_set(const char* pKey, const char* pVal ) {
  rvs::logger::log("property: [" + string(pKey) + "]   val:[" +
                  string(pVal)+"]", rvs::logdebug);
  return (*rvs_module_action_property_set)(plibaction, pKey, pVal);
}

/**
 * @brief Sets action property
 *
 * @param Key Property key
 * @param Val Property value
 * @return 0 - success. non-zero otherwise
 *
 */
int rvs::if1::property_set(const std::string& Key, const std::string& Val) {
  return property_set( Key.c_str(), Val.c_str());
}

/**
 * @brief Execute action
 *
 * @return 0 - success, non-zero otherwise
 *
 */
int rvs::if1::run(void) {
  return (*rvs_module_action_run)(plibaction);
}
