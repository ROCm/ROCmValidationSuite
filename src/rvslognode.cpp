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
#include "rvslognode.h"

using namespace std;

rvs::LogNode::LogNode( const std::string& Name, const LogNodeBase* Parent)
:
LogNodeBase(Name, Parent) {
  Type = eLN::List;
}

rvs::LogNode::LogNode( const char* Name, const LogNodeBase* Parent)
:
LogNodeBase(Name, Parent) {
  Type = eLN::List;
}

rvs::LogNode::~LogNode() {
  for(auto it = Child.begin(); it != Child.end(); ++it) {
    delete (*it);
  }

}

void rvs::LogNode::Add(LogNodeBase* pChild) {
  Child.push_back(pChild);
}

std::string rvs::LogNode::ToJson(const std::string& Lead) {

  string result(RVSENDL);
  result += Lead + "\"" + Name + "\"" + " : {";

  int  size = Child.size();
  for(int i = 0; i < size; i++) {
    result += Child[i]->ToJson(Lead + RVSINDENT);
    if( i+ 1 < size) {
      result += ",";
    }
  }
  result += Lead + "}";

  return result;
}