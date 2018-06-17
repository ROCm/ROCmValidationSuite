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
#include "rvsloglp.h"

#include <chrono>

rvs::lp::lp() {}
rvs::lp::~lp() {}

using namespace std;

T_MODULE_INIT rvs::lp::mi;

int   rvs::lp::Initialize(const T_MODULE_INIT* pMi) {
  mi.cbLog             = pMi->cbLog;
  mi.cbLogExt          = pMi->cbLogExt;
  mi.cbLogRecordCreate = pMi->cbLogRecordCreate;
  mi.cbLogRecordFlush  = pMi->cbLogRecordFlush;
  mi.cbCreateNode      = pMi->cbCreateNode;
  mi.cbAddString       = pMi->cbAddString;
  mi.cbAddInt          = pMi->cbAddInt;
  mi.cbAddNode         = pMi->cbAddNode;

  return 0;
}

int rvs::lp::Log(const char* pMsg, const int level) {
  return (*mi.cbLog)(pMsg, level);
}

int rvs::lp::Log(const string& Msg, const int LogLevel, const unsigned int Sec, const unsigned int uSec) {
  return (*mi.cbLogExt)(Msg.c_str(), LogLevel, Sec, uSec);
}

void* rvs::lp::LogRecordCreate( const char* Module, const char* Action, const int LogLevel, const unsigned int Sec, const unsigned int uSec) {
  return (*mi.cbLogRecordCreate)(Module,  Action,  LogLevel, Sec, uSec);
}

void* rvs::lp::LogRecordCreate( const char* Module, const char* Action, const int LogLevel) {
  return (*mi.cbLogRecordCreate)(Module,  Action,  LogLevel, 0, 0);
}

void* rvs::lp::LogRecordCreate( const string Module, const string Action, const int LogLevel) {
  return (*mi.cbLogRecordCreate)(Module.c_str(),  Action.c_str(),  LogLevel, 0, 0);
}

int   rvs::lp::LogRecordFlush( void* pLogRecord) {
  return (*mi.cbLogRecordFlush)(pLogRecord);
}

void* rvs::lp::CreateNode(void* Parent, const char* Name) {
  return (*mi.cbCreateNode)(Parent, Name);
}

void  rvs::lp::AddString(void* Parent, const string& Key, const string& Val) {
  (*mi.cbAddString)(Parent, Key.c_str(), Val.c_str());
}

void  rvs::lp::AddString(void* Parent, const char* Key, const char* Val) {
  (*mi.cbAddString)(Parent, Key, Val);
}

void  rvs::lp::AddInt(void* Parent, const char* Key, const int Val) {
  (*mi.cbAddInt)(Parent, Key, Val);
}

void  rvs::lp::AddNode(void* Parent, void* Child) {
  (*mi.cbAddNode)(Parent, Child);
}

bool rvs::lp::get_ticks(unsigned int& secs, unsigned int& usecs) {
  struct timespec ts;

  clock_gettime( CLOCK_MONOTONIC, &ts );
  usecs    = ts.tv_nsec / 1000;
  secs  = ts.tv_sec;

  return true;
}
