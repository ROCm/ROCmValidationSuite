/********************************************************************************
 *
 * Copyright (c) 2018 OCm Developer Tools
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

#include "include/rvsoptions.h"

#include <map>
#include <string>

std::map<std::string, std::string> rvs::options::opt;

/**
 * @brief Check and retrieve option.
 *
 * @param Option option to look for
 * @param pval option value. Unchanged if option does not exist.
 * @return 'true' if Option exists, 'false' otherwise
 *
 */
bool  rvs::options::has_option(const std::string& Option, std::string* pval) {
  auto it = opt.find(std::string(Option));
  if (it == opt.end())
    return false;

  *pval = it->second;
  return true;
}

/**
 * @brief Check option.
 *
 * @param Option option to look for
 * @return 'true' if Option exists, 'false' otherwise
 *
 */
bool  rvs::options::has_option(const std::string& Option) {
  auto it = opt.find(std::string(Option));
  if (it == opt.end())
    return false;

  return true;
}

/**
 * @brief Get options.
 *
 * @return collection of options
 *
 */
const std::map<std::string, std::string>& rvs::options::get(void) {
  return opt;
}


