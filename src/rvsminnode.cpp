/********************************************************************************
 *
 * Copyright (c) 2018-2022 Advanced Micro Devices, Inc.
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
#include <iostream>

#include "include/rvsminnode.h"
#include "include/rvstrace.h"

using std::string;

/**
 * @brief Constructor
 *
 * @param Name Node name
 * @param Parent Pointer to parent node
 *
 */
rvs::MinNode::MinNode(const char* Name, int LogLevel, const rvs::LogNodeBase* Parent)
:
LogNode(Name, Parent),
Level(LogLevel){
  Type = eLN::Record;
}

//! Destructor
rvs::MinNode::~MinNode() {
  for (auto it = Child.begin(); it != Child.end(); ++it) {
    delete (*it);
  }
}

/**
 * @brief Get logging level
 *
 * @return Current logging level
 *
 */
int rvs::MinNode::LogLevel() {
  return Level;
}

/**
 * @brief Add previously constructed node to the list of child nodes
 *
 * @param pChild Pointer to child node
 *
 */
void rvs::MinNode::Add(rvs::LogNodeBase* pChild) {
  Child.push_back(pChild);
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
std::string rvs::MinNode::ToJson(const std::string& Lead) {
  DTRACE_
  string result(RVSENDL);
  result += "{";

  int  size = Child.size();
  for (int i = 0; i < size; i++) {
    result += Child[i]->ToJson(Lead + RVSINDENT);
    if (i+ 1 < size) {
      result += ",";
    }
  }
  result += RVSENDL + Lead + "}";

  return result;
}
