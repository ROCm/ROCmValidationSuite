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

#include "include/rvsaction.h"



/**
 * @section Test
 * @brief Constructor
 *
 * Returns value if propety is set.
 *
 * @param pName acion name as specified in .conf file
 * @param pLibAction pointer to action instance in an RVS module
 *
 * */
rvs::action::action(const char* pName, void* pLibAction) {
  name       = pName;
  plibaction = pLibAction;
}

//! Default destructor
rvs::action::~action() {
  ifmap.clear();
}

/**
 * @brief Returns pointer to RVS interface
 *
 * Given RVS interface ID, returns interface pointer as pointer to common
 * interface base class. If action does not support particular interface,
 * NULL is returned.
 *
 * @param iID Interface ID
 *
 * */
rvs::ifbase* rvs::action::get_interface(int iID) {
  auto it = ifmap.find(iID);
  if (it != ifmap.end())
    return &(*(it->second));

  return nullptr;
}
