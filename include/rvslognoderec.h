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
#ifndef INCLUDE_RVSLOGNODEREC_H_
#define INCLUDE_RVSLOGNODEREC_H_

#include <vector>
#include <string>

#include "include/rvslognode.h"

namespace rvs {
/**
 * @class LogNodeRec
 * @ingroup Launcher
 *
 * @brief Logger Node class
 *
 * Used to construct structure log record for JSON output
 *
 */
class LogNodeRec : public LogNode {
 public:
  LogNodeRec(const char* Name, int LogLevel, unsigned Sec,
             unsigned uSec, const LogNodeBase* Parent = nullptr);
  virtual ~LogNodeRec();

  virtual std::string ToJson(const std::string& Lead = "");

 public:
  int LogLevel();

 protected:
  //! Logging Level
  int Level;
  //! Timestamp - seconds from system start
  const int sec;
  //! Timestamp - microseconds in current second
  const int usec;
};

}  // namespace rvs

#endif  // INCLUDE_RVSLOGNODEREC_H_
