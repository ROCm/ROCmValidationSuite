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
#include "include/rvsloglp.h"

#include <chrono>
#include <string>

#include "include/rvsliblogger.h"


using std::string;

T_MODULE_INIT rvs::lp::mi;

/**
 * @brief Initialize logger proxy class
 *
 * @param pMi Pointer to module initialization structure
 * @return 0 - success, non-zero otherwise
 *
 */
int   rvs::lp::Initialize(const T_MODULE_INIT* pMi) {
  mi.cbLog             = pMi->cbLog;
  mi.cbLogExt          = pMi->cbLogExt;
  mi.cbLogRecordCreate = pMi->cbLogRecordCreate;
  mi.cbLogRecordFlush  = pMi->cbLogRecordFlush;
  mi.cbCreateNode      = pMi->cbCreateNode;
  mi.cbAddString       = pMi->cbAddString;
  mi.cbAddInt          = pMi->cbAddInt;
  mi.cbAddNode         = pMi->cbAddNode;
  mi.cbStop            = pMi->cbStop;
  mi.cbStopping        = pMi->cbStopping;
  mi.cbErr             = pMi->cbErr;

  return 0;
}

/**
 * @brief Output log message
 *
 * @param pMsg Message to log
 * @param level Logging level
 * @return 0 - success, non-zero otherwise
 *
 */
int rvs::lp::Log(const char* pMsg, const int level) {
  return rvs::logger::Log(pMsg, level);
}

/**
 * @brief Output log message
 *
 * @param Msg Message to log
 * @param level Logging level
 * @return 0 - success, non-zero otherwise
 *
 */
int rvs::lp::Log(const std::string& Msg, const int level) {
  return rvs::logger::Log(Msg.c_str(), level);
}

/**
 * @brief Output log message
 *
 * @param Msg Message to log
 * @param LogLevel Logging level
 * @param Sec seconds from system start
 * @param uSec microseconds within current second
 * @return 0 - success, non-zero otherwise
 *
 */
int rvs::lp::Log(const std::string& Msg, const int LogLevel,
                 const unsigned int Sec, const unsigned int uSec) {
  return rvs::logger::LogExt(Msg.c_str(), LogLevel, Sec, uSec);
}

/**
 * @brief Create log record
 *
 * Note: this API is used to construct JSON output. Use LogExt() to perform
 * unstructured output.
 *
 * @param Module Module from which record is originating
 * @param Action Action from which record is originating
 * @param LogLevel Logging level
 * @param Sec seconds from system start
 * @param uSec microseconds within current second
 * @return 0 - success, non-zero otherwise
 *
 */
void* rvs::lp::LogRecordCreate(const char* Module, const char* Action,
                               const int LogLevel, const unsigned int Sec,
                               const unsigned int uSec) {
  return rvs::logger::LogRecordCreate(Module,  Action,  LogLevel, Sec, uSec);
}

/**
 * @brief Output log record
 *
 * Sends out record previously created using LogRecordCreate()
 *
 * @param pLogRecord Pointer to previously created log record
 * @return 0 - success, non-zero otherwise
 *
 */
int   rvs::lp::LogRecordFlush(void* pLogRecord) {
  return rvs::logger::LogRecordFlush(pLogRecord);
}

/**
 * @brief Create loggin output node
 *
 * Note: this API is used to construct JSON output.
 *
 * @param Parent Parent node
 * @param Name Node name
 * @return Pointer to newly created node
 *
 */
void* rvs::lp::CreateNode(void* Parent, const char* Name) {
  return rvs::logger::CreateNode(Parent, Name);
}

/**
 * @brief Create and add child node of type string to the given parent node
 *
 * Note: this API is used to construct JSON output.
 *
 * @param Parent Parent node
 * @param Key Key (node name)
 * @param Val Node value
 *
 */
void  rvs::lp::AddString(void* Parent, const std::string& Key,
                         const std::string& Val) {
  rvs::logger::AddString(Parent, Key.c_str(), Val.c_str());
}

/**
 * @brief Create and add child node of type int to the given parent node
 *
 * Note: this API is used to construct JSON output.
 *
 * @param Parent Parent node
 * @param Key Key (node name)
 * @param Val Node value
 *
 */
void  rvs::lp::AddString(void* Parent, const char* Key, const char* Val) {
  rvs::logger::AddString(Parent, Key, Val);
}

/**
 * @brief Create and add child node of type int to the given parent node
 *
 * Note: this API is used to construct JSON output.
 *
 * @param Parent Parent node
 * @param Key Key as C string
 * @param Val Value as integer
 *
 */
void  rvs::lp::AddInt(void* Parent, const char* Key, const int Val) {
  rvs::logger::AddInt(Parent, Key, Val);
}

/**
 * @brief Add child node to parent
 *
 * Takes node previously created using CreateNode() and
 * adds it to the parent node.
 *
 * Note: this API is used to construct JSON output.
 *
 * @param Parent Parent node
 * @param Child Child node
 *
 */
void  rvs::lp::AddNode(void* Parent, void* Child) {
  rvs::logger::AddNode(Parent, Child);
}

/**
 * @brief Fetches times since system start
 *
 * @param psecs seconds since system start
 * @param pusecs microseconds within current second
 * @return 'true' - success, 'false' otherwise
 *
 */
bool rvs::lp::get_ticks(unsigned int* psecs, unsigned int* pusecs) {
  struct timespec ts;

  clock_gettime(CLOCK_MONOTONIC, &ts);
  *pusecs    = ts.tv_nsec / 1000;
  *psecs  = ts.tv_sec;

  return true;
}

/**
 * @brief Signals that RVS is about to terminate.
 *
 * This method will prevent all further
 * printout to cout and to log file if any. Log file will be closed and properly
 * terminated before returning from this function.
 *
 */
void  rvs::lp::Stop(uint16_t flags) {
  rvs::logger::Stop(flags);
}

/**
 * @brief Returns stop flag
 *
 * Checks if a module requested RVS processing to stop
 *
 */
bool  rvs::lp::Stopping() {
  return rvs::logger::Stopping();
}

/**
 * @brief Log Error output
 *
 * @param Module  Module where error happend
 * @param Message Message to log
 * @return 0 - success, non-zero otherwise
 *
 */
int rvs::lp::Err(const std::string &Message, const std::string &Module) {
  return rvs::logger::Err(Message.c_str(), Module.c_str(), nullptr);
}

/**
 * @brief Log Error output
 *
 * @param Module  Module where error happened
 * @param Action  Action where error happened
 * @param Message Message to log
 * @return 0 - success, non-zero otherwise
 *
 */
int rvs::lp::Err(const std::string &Message
      , const std::string &Module, const std::string &Action) {
  return rvs::logger::Err(Message.c_str(), Module.c_str(), Action.c_str());
}
