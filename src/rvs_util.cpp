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
#include "include/rvs_util.h"

#include <vector>
#include <string>
#include <regex>


/**
 * splits a std::string based on a given delimiter
 * @param str_val input std::string
 * @param delimiter tokens' delimiter
 * @return std::vector containing all tokens
 */
std::vector<std::string> str_split(const std::string& str_val,
                                   const std::string& delimiter) {
    std::vector<std::string> str_tokens;
    size_t prev_pos = 0, cur_pos = 0;
    do {
        cur_pos = str_val.find(delimiter, prev_pos);
        if (cur_pos == std::string::npos)
            cur_pos = str_val.length();
        std::string token = str_val.substr(prev_pos, cur_pos - prev_pos);
        if (!token.empty())
            str_tokens.push_back(token);
        prev_pos = cur_pos + delimiter.length();
    } while (cur_pos < str_val.length() && prev_pos < str_val.length());
    return str_tokens;
}


/**
 * checks if input std::string is a positive integer number
 * @param str_val the input std::string
 * @return true if std::string is a positive integer number, false otherwise
 */
bool is_positive_integer(const std::string& str_val) {
    return !str_val.empty()
            && std::find_if(str_val.begin(), str_val.end(),
                    [](char c) {return !std::isdigit(c);}) == str_val.end();
}

int rvs_util_parse(const std::string& buff, bool* pval) {
  if (buff.empty()) {  // method empty
    return 2;  // not found
  }

  if (buff == "true") {
    *pval = true;
    return 0;  // OK - true
  }

  if (buff == "false") {
    *pval = false;
    return 0;  // OK - false
  }

  return 1;  // syntax error
}
