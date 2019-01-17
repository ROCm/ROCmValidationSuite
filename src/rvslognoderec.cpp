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

#include "include/rvslognoderec.h"

#include <string>
#include "include/rvstrace.h"

/**
 * @brief Constructor
 *
 * @param Name Node name
 * @param LoggingLevel Logging level
 * @param Sec secconds since system start
 * @param uSec microseconds in current second
 * @param Parent Pointer to parent node
 *
 */
rvs::LogNodeRec::LogNodeRec(const char* Name, int LoggingLevel,
  const unsigned Sec, const unsigned uSec, const LogNodeBase* Parent)
:
LogNode(Name, Parent),
Level(LoggingLevel),
sec(Sec),
usec(uSec) {
  Type = eLN::Record;
}

//! Destructor
rvs::LogNodeRec::~LogNodeRec() {
}

/**
 * @brief Get logging level
 *
 * @return Current logging level
 *
 */
int rvs::LogNodeRec::LogLevel() {
  return Level;
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
std::string rvs::LogNodeRec::ToJson(const std::string& Lead) {
  DTRACE_
  std::string result(RVSENDL);
  result += Lead + "{";

  result += RVSENDL;
  result += Lead + RVSINDENT;
  result += std::string("\"") + "loglevel" + "\"" + " : " +
            std::to_string(Level) + ",";

  char  buff[64];
  snprintf(buff, sizeof(buff), "%6d.%-6d", sec, usec);
  result += RVSENDL;
  result += Lead + RVSINDENT;
  result += std::string("\"") + "time" + "\"" + " : " +
            std::string("\"") + buff + std::string("\"")  + ",";

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
