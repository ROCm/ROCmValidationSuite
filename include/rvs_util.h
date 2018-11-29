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
#include <iostream>

extern bool is_positive_integer(const std::string& str_val);

extern std::vector<std::string> str_split(const std::string& str_val,
        const std::string& delimiter);

/**
 * Convert array of strings into array of signed integers of type T
 * @param sArr input string
 * @param iArr tokens' delimiter
 * @return -1 if error, number of elements in the array otherwise
 */
template <typename T>
int rvs_util_strarr_to_intarr(const std::vector<std::string>& sArr,
                              std::vector<T>* piArr) {
  piArr->clear();

  for (auto it = sArr.begin(); it != sArr.end(); ++it) {
    try {
      if (is_positive_integer(*it)) {
        piArr->push_back(std::stoi(*it));
      }
    }
    catch(...) {
    }
  }

  if (sArr.size() != piArr->size())
    return -1;

  return piArr->size();
}


/**
 * Convert array of strings into array of unsigned integers of type T
 * @param sArr input string
 * @param iArr tokens' delimiter
 * @return -1 if error, number of elements in the array otherwise
 */
template <typename T>
int rvs_util_strarr_to_uintarr(const std::vector<std::string>& sArr,
                              std::vector<T>* piArr) {
  piArr->clear();

  for (auto it = sArr.begin(); it != sArr.end(); ++it) {
    try {
      if (is_positive_integer(*it)) {
        piArr->push_back(std::stoul(*it));
      }
    }
    catch(...) {
    }
  }

  if (sArr.size() != piArr->size())
    return -1;

  return piArr->size();
}


extern int rvs_util_parse(const std::string& buff, bool* pval);

/**
 * @brief turns string value into right type of integer, else returns error
 */

template <typename T>
int rvs_util_parse(const std::string& buff,
                                    T* pval) {
  int error;
  if (buff.empty()) {  // method empty
    error = 2;
  } else {
    if (is_positive_integer(buff)) {
      try {
        *pval = std::stoul(buff);
        error = 0;
      } catch(...) {
        error = 1;  // we have an empty string
      }
    } else {
      error = 1;
    }
  }
  return error;
}

#endif  // INCLUDE_RVS_UTIL_H_
