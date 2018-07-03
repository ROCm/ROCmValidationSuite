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
#ifndef RVSLIBLOGGER_H_
#define RVSLIBLOGGER_H_

#include <string>
#include "rvsliblog.h"


namespace rvs
{


/**
 * @class logger
 * @ingroup Launcher
 *
 * @brief Message logging class
 *
 */
class logger
{
protected:

public:
  static  void  log_level(const int level);
  static  int   log_level();

  static  void  to_json(const bool flag);
  static  bool  to_json();

  static  void  append(const bool flag);
  static  bool  append();

  static  void   logfile(const std::string& filename);
  static  const  std::string& logfile();

  static  bool   get_ticks(uint32_t& secs, uint32_t& usecs);

  static  int    initialize();
  static  int    terminate();

  static  int    log(const std::string& Message, const int level = 1);
  static  int    Log(const char* Message, const int level);
  static  int    LogExt(const char* Message, const int LogLevel, const unsigned int Sec, const unsigned int uSec);
  static  void*  LogRecordCreate( const char* Module, const char* Action, const int LogLevel, const unsigned int Sec, const unsigned int uSec);
  static  int    LogRecordFlush( void* pLogRecord);
  static  void*  CreateNode(void* Parent, const char* Name);
  static  void   AddString(void* Parent, const char* Key, const char* Val);
  static  void   AddInt(void* Parent, const char* Key, const int Val);
  static  void   AddNode(void* Parent, void* Child);
  static  int    ToFile(const std::string& Row);
  static  int    JsonPatchAppend(void);

protected:
  //! Current logging level (0..5)
  static  int    loglevel_m;
  //! 'true' if JSON output is requested
  static  bool   tojson_m;
  //! 'true' if append to existing log file is requested
  static  bool   append_m;
  //! 'true' if the incoming record is the first record in this rvs invocation
  static  bool   isfirstrecord_m;
  //! Name of the log file
  static  std::string logfile_m;
  //! Array of C std::strings representing logging level names
  static  const char*   loglevelname[6];
};

}  // namespace rvs






#endif // RVSLIBLOGGER_H_
