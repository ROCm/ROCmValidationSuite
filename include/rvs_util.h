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
#ifndef INCLUDE_RVS_UTIL_H_
#define INCLUDE_RVS_UTIL_H_

#include <vector>
#include <string>

using std::vector;
using std::string;

extern vector<string> str_split(const string& str_val,
        const string& delimiter);

extern int rvs_util_strarr_to_intarr(const std::vector<string>& sArr,
                                     std::vector<int>* piArr);

extern int rvs_util_strarr_to_uintarr(const std::vector<string>& sArr,
                                     std::vector<uint16_t>* piArr);

extern int rvs_util_strarr_to_uintarr(const std::vector<string>& sArr,
                                     std::vector<uint32_t>* piArr);

bool is_positive_integer(const std::string& str_val);

#endif  // INCLUDE_RVS_UTIL_H_
