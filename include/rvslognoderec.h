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
#ifndef _LOGNODEREC_H
#define _LOGNODEREC_H

#include "rvslognode.h"

#include <vector>


namespace rvs 
{

/**
 * @class LogNodeRec
 * @ingroup Launcher
 *
 * @brief Logger Node class
 *
 * Used to construct structure log record for JSON output
 *
 */
class LogNodeRec : public LogNode
{

public:

  LogNodeRec( const std::string& Name, const int LogLevel, const unsigned int Sec, const unsigned int uSec, const LogNodeBase* Parent = nullptr);
  LogNodeRec( const char* Name, const int LogLevel, const unsigned int Sec, const unsigned int uSec, const LogNodeBase* Parent = nullptr);
  virtual ~LogNodeRec();

  virtual std::string ToJson(const std::string& Lead = "");

public:
  const int LogLevel();

protected:
  //! Logging Level
  int Level;
  //! Timestamp - seconds from system start
  const int sec;
  //! Timestamp - microseconds in current second
  const int usec;
};

}  // namespace rvs

#endif // _LOGNODEREC_H