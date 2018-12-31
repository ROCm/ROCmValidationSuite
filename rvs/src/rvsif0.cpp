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
#include "include/rvsif0.h"

//! Default constructor
rvs::if0::if0()
:
rvs_module_get_description(nullptr),
rvs_module_get_config(nullptr),
rvs_module_get_output(nullptr) {
}

//! Default destructor
rvs::if0::~if0() {
}

/**
 * @brief Copy constructor
 *
 * @param rhs reference to RHS instance
 *
 */
rvs::if0::if0(const if0& rhs) : ifbase(rhs) {
  *this = rhs;
}

/**
 * @brief Assignment operator
 *
 * @param rhs reference to RHS instance
 * @return reference to LHS instance
 *
 */
rvs::if0& rvs::if0::operator=(const rvs::if0& rhs) {
  // self-assignment check
  if (this != &rhs) {
    ifbase::operator=(rhs);
    rvs_module_get_description  = rhs.rvs_module_get_description;
    rvs_module_get_config    = rhs.rvs_module_get_config;
    rvs_module_get_output    = rhs.rvs_module_get_output;
  }

  return *this;
}

/**
 * @brief Clone instance
 *
 * @return pointer to newly created instance
 *
 */
rvs::ifbase* rvs::if0::clone(void) {
  return new rvs::if0(*this);
}

/**
 * @brief Get module description
 *
 * @return pointer to C string holding module description
 *
 */
const char* rvs::if0::get_description(void) {
  return (*rvs_module_get_description)();
}

/**
 * @brief Returns short info on what configuaration properties (keys) are suppored
 *
 * @return pointer to C string holding configuration info
 *
 */
const char* rvs::if0::get_config() {
  return (*rvs_module_get_config)();
}

/**
 * @brief Returns short info on what output properties (keys) are returned
 *
 * @return pointer to C string holding output info
 *
 */
const char* rvs::if0::get_output() {
  return (*rvs_module_get_output)();
}



