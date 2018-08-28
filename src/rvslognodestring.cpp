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

#include "rvslognodestring.h"

using std::string;

/**
 * @brief Constructor
 *
 * @param Name Node name
 * @param Val Node value
 * @param Parent Pointer to parent node
 *
 */
rvs::LogNodeString::LogNodeString(const std::string& Name,
                                  const std::string& Val,
                                  const LogNodeBase* Parent)
:
LogNodeBase(Name, Parent),
Value(Val) {
  Type = eLN::String;
}

/**
 * @brief Constructor
 *
 * @param Name Node name
 * @param Val Node value
 * @param Parent Pointer to parent node
 *
 */
rvs::LogNodeString::LogNodeString(const char* Name, const char* Val,
                                  const LogNodeBase* Parent)
:
LogNodeBase(Name, Parent),
Value(Val) {
  Type = eLN::String;
}

//! Destructor
rvs::LogNodeString::~LogNodeString() {
}

/**
 * @brief Provides JSON representation of Node
 *
 * Traverses list of child nodes and converts them into proper string representation.
 * Also ensures proper indentation and line breaks for formatted output.
 *
 * @param Lead String of blanks " " representing current indentation
 * @return Node as JSON string
 *
 */
std::string rvs::LogNodeString::ToJson(const std::string& Lead) {
  string result(RVSENDL);
  result += Lead + "\"" + Name + "\"" + " : " + "\"" + Value + "\"";

  return result;
}
