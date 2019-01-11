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
#ifndef INCLUDE_RVSLOGNODE_H_
#define INCLUDE_RVSLOGNODE_H_

#include <vector>
#include <memory>
#include <string>

#include "include/rvslognodebase.h"

namespace rvs {

/**
 * @class LogNode
 * @ingroup Launcher
 *
 * @brief Logger Node class
 *
 * Used to construct structure log record for JSON output
 *
 */
class LogNode : public LogNodeBase {
 public:
  explicit LogNode(const char* Name, const LogNodeBase* Parent = nullptr);
  virtual ~LogNode();

  virtual std::string ToJson(const std::string& Lead = "");

 public:
  void Add(LogNodeBase* spChild);

 public:
  //! list of child nodes
  std::vector<LogNodeBase*> Child;
};

}  // namespace rvs

#endif  // INCLUDE_RVSLOGNODE_H_
