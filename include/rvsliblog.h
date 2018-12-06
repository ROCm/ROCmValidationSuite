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
#ifndef INCLUDE_RVSLIBLOG_H_
#define INCLUDE_RVSLIBLOG_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int   (*t_rvs_module_log)(const char*, const int);
typedef int   (*t_cbLogExt)(const char*, const int, const unsigned int Sec,
                            const unsigned int uSec);
typedef void* (*t_cbLogRecordCreate)( const char* Module, const char* Action,
                                      const int LogLevel,
                                      const unsigned int Sec,
                                      const unsigned int uSec);
typedef int   (*t_cbLogRecordFlush)( void* pLogRecord);
typedef void* (*t_cbCreateNode)(void* Parent, const char* Name);
typedef void  (*t_cbAddString)(void* Parent, const char* Key, const char* Val);
typedef void  (*t_cbAddInt)(void* Parent, const char* Key, const int Val);
typedef void  (*t_cbAddNode)(void* Parent, void* Child);
typedef void  (*t_cbStop)(uint16_t flags);
typedef bool  (*t_cbStopping)(void);
typedef int   (*t_rvs_module_err)(const char*, const char*, const char*);


/**
 * @brief Module initialization structure
 */
typedef struct tag_module_init {
  //! pointer to rvs::logger::Log() function
  t_rvs_module_log     cbLog;
  //! pointer to rvs::logger::LogExt() function
  t_cbLogExt           cbLogExt;
  //! pointer to rvs::logger::LogRecordCreate() function
  t_cbLogRecordCreate  cbLogRecordCreate;
  //! pointer to rvs::logger::LogRecordFlush() function
  t_cbLogRecordFlush   cbLogRecordFlush;
  //! pointer to rvs::logger::CreateNode() function
  t_cbCreateNode       cbCreateNode;
  //! pointer to rvs::logger::AddString() function
  t_cbAddString        cbAddString;
  //! pointer to rvs::logger::AddInt() function
  t_cbAddInt           cbAddInt;
  //! pointer to rvs::logger::AddNode() function
  t_cbAddNode          cbAddNode;
  //! pointer to rvs::logger::Stop() function
  t_cbStop             cbStop;
  //! pointer to rvs::logger::Stopping() function
  t_cbStopping         cbStopping;
  //! pointer to rvs::logger::Err() function
  t_rvs_module_err     cbErr;
} T_MODULE_INIT;

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
namespace rvs {

const int lognone     = 0;
const int logresults  = 1;
const int logerror    = 2;
const int loginfo     = 3;
const int logdebug    = 4;
const int logtrace    = 5;

}  // namespace rvs

#endif


#endif  // INCLUDE_RVSLIBLOG_H_
