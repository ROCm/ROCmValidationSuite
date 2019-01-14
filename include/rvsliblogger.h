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
#ifndef INCLUDE_RVSLIBLOGGER_H_
#define INCLUDE_RVSLIBLOGGER_H_

#include <string>
#include <mutex>
#include "include/rvsliblog.h"


namespace rvs {


/**
 * @class logger
 * @ingroup Launcher
 *
 * @brief Message logging class
 *
 */
class logger {
 public:
  static  void  log_level(const int level);

  static  void  to_json(const bool flag);
  static  bool  to_json();

  static  void  append(const bool flag);
  static  bool  append();

  //! set quiet mode
  static  void  quiet() { b_quiet = true; }
  //! set logging file
  static  void  set_log_file(const std::string& fname);

  static  bool   get_ticks(uint32_t* psecs, uint32_t* pusecs);

  static  int    init_log_file();
  static  int    terminate();

  static  int    log(const std::string& Message, const int level = 1);
  static  int    Log(const char* Message, const int level);
  static  int    LogExt(const char* Message, const int LogLevel,
                        const unsigned int Sec, const unsigned int uSec);
  static  void*  LogRecordCreate(const char* Module, const char* Action,
                                  const int LogLevel, const unsigned int Sec,
                                  const unsigned int uSec);
  static  int    LogRecordFlush(void* pLogRecord);
  static  void*  CreateNode(void* Parent, const char* Name);
  static  void   AddString(void* Parent, const char* Key, const char* Val);
  static  void   AddInt(void* Parent, const char* Key, const int Val);
  static  void   AddNode(void* Parent, void* Child);
  static  int    JsonPatchAppend(int*);
  static  void   Stop(uint16_t flags);
  static  bool   Stopping(void);
  static  int    Err(const char *Message,
                   const char *Module = nullptr, const char *Action = nullptr);

 protected:
  static  int    ToFile(const std::string& Row);

  //! Current logging level (0..5)
  static  int    loglevel_m;
  //! 'true' if JSON output is requested
  static  bool   tojson_m;
  //! 'true' if append to existing log file is requested
  static  bool   append_m;
  //! 'true' if the incoming record is the first record in this rvs invocation
  static  bool   isfirstrecord_m;
  //! Array of C std::strings representing logging level names
  static  const char*   loglevelname[6];
  //! Mutex to synchronize cout output
  static std::mutex cout_mutex;
  //! Mutex to synchronize log file output
  static std::mutex log_mutex;
  //! flag indicating stop loging was requested
  static bool bStop;
  //! stop flags
  static uint16_t stop_flags;
  //! logging file
  static char log_file[1024];
  //! quiet mode
  static bool b_quiet;
};

}  // namespace rvs

#endif  // INCLUDE_RVSLIBLOGGER_H_
