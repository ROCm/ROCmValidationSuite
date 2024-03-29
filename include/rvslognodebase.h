/********************************************************************************
 *
 * Copyright (c) 2018-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef INCLUDE_RVSLOGNODEBASE_H_
#define INCLUDE_RVSLOGNODEBASE_H_

#include <string>

#define RVSENDL "\n"
#define RVSINDENT "  "

namespace rvs {

typedef enum eLN {
  Unknown = 0,
  List    = 1,
  String  = 2,
  Integer = 3,
  Record  = 4
} T_LNTYPE;

/**
 * @class LogNodeBase
 * @ingroup Launcher
 *
 * @brief Base class for all logger nodes
 *
 */
class LogNodeBase {
 public:
  virtual ~LogNodeBase();

/**
 * @brief Provides JSON representation of Node
 *
 * Converts node into proper string representation.
 * This method has to be implemented in every derived class.
 *
 * @param Lead String of blanks " " representing current indentation
 * @return Node as JSON string
 *
 */
  virtual std::string ToJson(const std::string& Lead = "") = 0;

 protected:
  explicit LogNodeBase(const char* rName,
                       const LogNodeBase* pParent = nullptr);

 protected:
  //! Node name
  std::string     Name;
  //! Parent node
  const LogNodeBase*   Parent;
  //! Node type
  T_LNTYPE       Type;
};




}  // namespace rvs

#endif  // INCLUDE_RVSLOGNODEBASE_H_
