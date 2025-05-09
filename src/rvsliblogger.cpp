/********************************************************************************
 *
 * Copyright (c) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "include/rvsliblogger.h"
#include <sys/types.h>
#include <dirent.h>
#include <sys/stat.h>

#include <unistd.h>
#include <time.h>
#include <stdio.h>
#include <cstring>

#include <iostream>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <string>
#include <mutex>

#include "include/rvstrace.h"
#include "include/rvslognode.h"
#include "include/rvslognodestring.h"
#include "include/rvslognodeint.h"
#include "include/rvslognoderec.h"
#include "include/rvsminnode.h"
#include "include/rvslognodelist.h"

using std::cerr;
using std::cout;

int   rvs::logger::loglevel_m(2);
bool  rvs::logger::tojson_m(false);
bool  rvs::logger::append_m(false);
bool  rvs::logger::isfirstrecord_m(true);
bool  rvs::logger::initModule(true);
bool  rvs::logger::isfirstaction_m(true);
std::mutex  rvs::logger::cout_mutex;
std::mutex  rvs::logger::log_mutex;
bool  rvs::logger::bStop(false);
uint16_t rvs::logger::stop_flags(0u);
bool rvs::logger::b_quiet(false);
char rvs::logger::log_file[1024];
std::string rvs::logger::json_log_file;
std::mutex  rvs::logger::json_log_mutex;
const char*  rvs::logger::loglevelname[] = {
  "NONE  ", "RESULT", "ERROR ", "INFO  ", "DEBUG ", "TRACE " };

// Json specific consts
const std::string node_start{"{"};
const std::string version_key{"version"};
const std::string version_val{"1.0"};
const std::string node_end{"}"};
const std::string kv_delimit{":"};
const std::string list_start{"["};
const std::string list_end{"]"};
const std::string newline{"\n"};
const std::string json_folder{"/var/tmp/"};

bool isPathedFile(const std::string &fname){
  return fname.find('/') != std::string::npos ;
}


bool doesFolderExist(const std::string &fname){
  auto loc = fname.find_last_of('/');
  auto dirName = fname.substr(0,loc);
  DIR* dir = opendir(dirName.c_str());
  if (dir == NULL) {
    // try creating directory, this doesnt exist. if fails return
     std::string command{"mkdir -p "};
     command += dirName;     
     int ret = system(command.c_str());
    if (ret){
      return false;
    }
  } 
  std::fstream fs;
  fs.open(fname,  std::ios::out | std::ios::trunc);
  if (fs.fail()){// unable to create file in dir
      return false;
  }
  return true;
}

/**
 * @brief helper to create json file name
 * @return json file name
 */
std::string json_filename(){
        std::string json_file;
        json_file.assign("rvs");
        std::chrono::milliseconds ms = std::chrono::duration_cast< std::chrono::milliseconds >(
            std::chrono::system_clock::now().time_since_epoch());
        json_file = json_file + "_" + std::to_string(ms.count()) + ".json";
        json_file = json_folder + json_file;
        return json_file;
}


/**
 * @brief Set 'append' flag
 *
 * @param flag new value
 *
 */
void rvs::logger::append(const bool flag) {
  append_m = flag;
}

/**
 * @brief Get 'append' flag
 *
 * @return Current flag value
 *
 */
bool rvs::logger::append() {
  return append_m;
}

void rvs::logger::set_log_file(const std::string& fname) {
    strncpy(log_file, fname.c_str(), sizeof(log_file));
    if (isPathedFile(log_file)){
        if (!doesFolderExist(log_file)){
          std::cout << "Unable to create log file, check path.";
        }
   }

}


void rvs::logger::set_json_log_file(const std::string& fname) {
    std::stringstream ss;
    if (!fname.empty()){
        json_log_file = fname;
        if (isPathedFile(fname) && !doesFolderExist(fname)){
		json_log_file = json_filename();
                ss << "Unable to create Json log file specified at" << fname << std::endl;
            }

   }else{
       json_log_file = json_filename(); 
   }
    ss << "Json log file created at " << json_log_file << std::endl;
    std::lock_guard<std::mutex> lk(cout_mutex);
    std::cout << ss.str();
}

/**
 * @brief Set logging level
 *
 * @param rLevel New logging level
 *
 */
void rvs::logger::log_level(const int rLevel) {
  loglevel_m = rLevel;
}

/**
 * @brief Fetches times since system start
 *
 * @param psecs seconds since system start
 * @param pusecs microseconds withing current second
 * @return 'true' - success, 'false otherwise
 *
 */
bool rvs::logger::get_ticks(uint32_t* psecs, uint32_t* pusecs) {
  struct timespec ts;

  clock_gettime(CLOCK_MONOTONIC, &ts);
  *pusecs    = ts.tv_nsec / 1000;
  *psecs  = ts.tv_sec;

  return true;
}

/**
 * @brief Set 'json' flag
 *
 * @param flag new value
 *
 */
void rvs::logger::to_json(const bool flag) {
  tojson_m = flag;
}

/**
 * @brief Get 'json' flag
 *
 * @return Current flag value
 *
 */
bool rvs::logger::to_json() {
  return tojson_m;
}

/**
 * @brief Output log message
 *
 * @param Message Message to log
 * @param LogLevel Logging level
 * @return 0 - success, non-zero otherwise
 *
 */
int rvs::logger::log(const std::string& Message, const int LogLevel) {
  return LogExt(Message.c_str(), LogLevel, 0, 0);
}

/**
 * @brief Output log message
 *
 * @param Message Message to log
 * @param LogLevel Logging level
 * @return 0 - success, non-zero otherwise
 *
 */
int rvs::logger::Log(const char* Message, const int LogLevel) {
  return LogExt(Message, LogLevel, 0, 0);
}

/**
 * @brief Output log message
 *
 * @param Message Message to log
 * @param LogLevel Logging level
 * @param Sec secconds from system start
 * @param uSec microseconds in current second
 * @return 0 - success, non-zero otherwise
 *
 */
int rvs::logger::LogExt(const char* Message, const int LogLevel,
                        const unsigned int Sec, const unsigned int uSec) {
  DTRACE_
  // stop logging requested?
  if (bStop) {
    DTRACE_
    if (stop_flags) {
      DTRACE_
      // just return
      return 0;
    }
  }

  if (LogLevel < lognone || LogLevel > logtrace) {
    DTRACE_
    char buff[128];
    snprintf(buff, sizeof(buff), "unknown logging level: %d", LogLevel);
    Err(buff, "CLI");
    return -1;
  }

  // log level too high?
  if (LogLevel > loglevel_m) {
    DTRACE_
    return 0;
  }

  uint32_t   secs = 0;
  uint32_t   usecs = 0;

  if (Sec|uSec) {
    DTRACE_
    secs = Sec;
    usecs = uSec;
  } else {
    DTRACE_
    get_ticks(&secs, &usecs);
  }

  DTRACE_
  char  buff[64];
  snprintf(buff, sizeof(buff), "%6d.%-6d", secs, usecs);

  std::string row("[");
  row += loglevelname[LogLevel];
  row +="] [";
  row += buff;
  row +="] ";
  row += Message;

  // if no quiet option given, output to cout
  if (!b_quiet) {
    DTRACE_
    // lock cout_mutex for the duration of this block
    std::lock_guard<std::mutex> lk(cout_mutex);
    cout << row << '\n';
  }

  // this stream does not output JSON
  if (to_json()) {
    DTRACE_
    return 0;
  }

  DTRACE_
  // send to file if requested
  if (isfirstrecord_m) {
    DTRACE_
    isfirstrecord_m = false;
  } else {
    DTRACE_
    row = RVSENDL + row;
  }
  DTRACE_

  if (true) {
    // lock log_mutex for the duration of this block
    std::lock_guard<std::mutex> lk(log_mutex);
    ToFile(row);
  }

  DTRACE_
  return 0;
}


/**
 * @brief Create log record
 *
 * Note: this API is used to construct JSON output. Use LogExt() to perform unstructured output.
 *
 * @param Module Module from which record is originating
 * @param Action Action from which record is originating
 * @param LogLevel Logging level
 * @param Sec secconds from system start
 * @param uSec microseconds in current second
 * @return 0 - success, non-zero otherwise
 *
 */
void* rvs::logger::LogRecordCreate(const char* Module, const char* Action,
                                   const int LogLevel, const unsigned int Sec,
                                   const unsigned int uSec, bool minimal) {
  uint32_t   sec;
  uint32_t   usec;
  if( json_log_file.empty()){
       json_log_file = json_filename();
       std::lock_guard<std::mutex> lk(cout_mutex);
       std::cout << "json log file is " << json_log_file<< std::endl;
  }
  if( minimal){
    rvs::MinNode* minrec = new rvs::MinNode(Action, LogLevel);
    return static_cast<void*>(minrec);
  }
  if ((Sec|uSec)) {
    sec = Sec;
    usec = uSec;
  } else  {
    get_ticks(&sec, &usec);
  }

  rvs::LogNodeRec* rec = new LogNodeRec(Action, LogLevel, sec, usec);
  AddString(rec, "action", Action);
  AddString(rec, "module", Module);
  AddString(rec, "loglevelname", (LogLevel >= lognone && LogLevel < logtrace) ?
    loglevelname[LogLevel] : "UNKNOWN");

  return static_cast<void*>(rec);
}



/**
 * @brief Create json log record
 *
 * Note: this API is used to construct JSON file start node with Action and Module and flush it. 
 *
 * @param Module Module from which record is originating
 * @param Action Action from which record is originating
 * @return 0 - success, non-zero otherwise
 *
 */
#if 1
int rvs::logger::JsonStartNodeCreate(const char* Module, const char* Action) {
    if ( json_log_file.empty()){
        json_log_file = json_filename();
        std::lock_guard<std::mutex> lk(cout_mutex);
        std::cout << "json log file is " << json_log_file<< std::endl;
  }
  std::string row{node_start};
  row += newline;
  row += std::string("\"") + version_key + std::string("\"") +kv_delimit ;
  row += std::string("\"") + version_val + std::string("\"") + "," + newline;
  row += std::string("\"") + Module + std::string("\"") + kv_delimit + node_start + newline;
  std::lock_guard<std::mutex> lk(json_log_mutex);
  return ToFile(row, true);
}

int rvs::logger::JsonActionStartNodeCreate(const char* Module, const char* Action) {
  if(initModule || json_log_file.empty()){
    rvs::logger::JsonStartNodeCreate(Module, Action);
    initModule =  false;
  }
  isfirstrecord_m = true;
  std::string row{newline};
  if (isfirstaction_m){
    isfirstaction_m = false;
  } else{
     row +=std::string(",");       
  }
  row += std::string(RVSINDENT);
  row += std::string("\"") + Action + std::string("\"") + kv_delimit + list_start + newline;
  std::lock_guard<std::mutex> lk(json_log_mutex);
  return ToFile(row, true);
}

void* rvs::logger::JsonNamedListCreate(const char* name,const int LogLevel){
    rvs::LogListNode* rec = new rvs::LogListNode(name, LogLevel);
    return static_cast<void*>(rec);

}	
int rvs::logger::JsonActionEndNodeCreate() {
  std::string row{RVSINDENT};
  row += list_end;
  std::lock_guard<std::mutex> lk(json_log_mutex);
  return ToFile(row, true);
}

/**
 * @brief Create a json file end record and flush it
 *
 * Note: this API is used to construct JSON output.
 *
 * @return 0 - success, non-zero otherwise
 *
 */

int rvs::logger::JsonEndNodeCreate(void) {
  if(json_log_file.empty())
    return -1;
  std::string row{RVSINDENT};
  row += RVSINDENT + node_end + newline;
  row += node_end;
  std::lock_guard<std::mutex> lk(json_log_mutex);
  return ToFile(row, true); 
}

#endif
/**
 * @brief Output log record
 *
 * Sends out record previously created using LogRecordCreate()
 *
 * @param pLogRecord Pointer to previously created log record
 * @return 0 - success, non-zero otherwise
 *
 */
int   rvs::logger::LogRecordFlush(void* pLogRecord, bool minimal) {
  // lock log_mutex for the duration of this block
  std::lock_guard<std::mutex> lk(json_log_mutex);
  std::string val;
  LogNode *r = nullptr;
  DTRACE_
  if(minimal){
    r = static_cast<MinNode*>(pLogRecord);	
  }else{
    r = static_cast<LogNodeRec*>(pLogRecord);
  }
  // no JSON loggin requested
  if (!to_json()) {
    DTRACE_
    delete r;
    return 0;
  }

  // assert log levl
  int level = r->LogLevel();
  if (level < lognone || level > logtrace) {
    char buff[128];
    snprintf(buff, sizeof(buff), "unknown logging level: %d", r->LogLevel());
    Err(buff, "CLI");
    delete r;
    return -1;
  }
  // if too high, ignore record
  if (level > loglevel_m) {
    DTRACE_
    delete r;
    return 0;
  }

  // do not pre-pend "," separator for the first row
  std::string row;
  if (append_m) {
    DTRACE_
    row = ",";
  } else {
    DTRACE_
    if (!isfirstrecord_m) {
      DTRACE_
      row = ",";
    }
  }
  DTRACE_
  // get JSON formatted log record
  row += r->ToJson("  ");

  // send it to file
  ToFile(row, true);
  
  // dealloc memory
  delete r;

  if (isfirstrecord_m) {
    DTRACE_
    isfirstrecord_m = false;
  }
  DTRACE_
  // return OK
  return 0;
}

/**
 * @brief Output log record to file
 *
 * Sends out string representing record to a log file
 *
 * @param Row string representing log record
 * @return 0 - success, non-zero otherwise
 *
 */
int rvs::logger::ToFile(const std::string& Row, bool json_rec) {
  if (bStop) {
    if (stop_flags)
      return 0;
  }

  std::string logfile;
  if (json_rec)
	logfile.assign(json_log_file);
  else
	logfile.assign(log_file);
  if (logfile == "")
    return -1;
  // check if folder, and if it exists/can be created.
  std::fstream fs;

  fs.open(logfile, std::fstream::out | std::fstream::app);
  if (fs.fail())
    return -1;
  fs << Row;

  fs.close();

  return 0;
}

/**
 * @brief Patch JSON log file
 *
 * In case when append "-a" option is given along with "-l --json",
 * replaces "]" terminating character with " " in order to
 * ensure well-formed JSON content.
 *
 * @param pSts non zero if patching took place
 * @return 0 - success, non-zero otherwise
 *
 */
int rvs::logger::JsonPatchAppend(int* pSts) {
  std::string logfile(json_log_file);

  FILE * pFile;
  pFile = fopen(logfile.c_str() , "r+");
  if (pFile == nullptr) {
    return -1;
  }
  fseek(pFile , -1 , SEEK_END);
  fputs(" " , pFile);
  fclose(pFile);
  *pSts = 1;
  return 0;
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
void* rvs::logger::CreateNode(void* Parent, const char* Name) {
  rvs::LogNode* p = new LogNode(Name, static_cast<rvs::LogNodeBase*>(Parent));
  return p;
}

/**
 * @brief Create and add child node of type string to the given parent node
 *
 * Note: this API is used to construct JSON output.
 *
 * @param Parent Parent node
 * @param Key Key as C string
 * @param Val Value as C string
 *
 */
void  rvs::logger::AddString(void* Parent, const char* Key, const char* Val) {
  rvs::LogNode* pp = static_cast<rvs::LogNode*>(Parent);
  rvs::LogNodeString* p = new LogNodeString(Key, Val, pp);
  pp->Add(p);
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
void  rvs::logger::AddInt(void* Parent, const char* Key, const int Val) {
  rvs::LogNode* pp = static_cast<rvs::LogNode*>(Parent);
  rvs::LogNodeInt* p = new LogNodeInt(Key, Val, pp);
  pp->Add(p);
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
void  rvs::logger::AddNode(void* Parent, void* Child) {
  rvs::LogNode* pp = static_cast<rvs::LogNode*>(Parent);
  pp->Add(static_cast<rvs::LogNodeBase*>(Child));
}


/**
 * @brief Initializes log file when "-l" command line option is given
 *
 * @return 0 - success, non-zero otherwise
 *
 */
int rvs::logger::init_log_file() {
  isfirstrecord_m = true;
  bStop = false;
  stop_flags = 0;

  std::string row;
  std::string logfile(log_file);

  // if no logg to file requested, just return
  if (logfile == "")
    return 0;

  if (append()) {
    // appnd to file, replace the closing "]" with "," in order to
    // have well formed JSON after appending
    int patch_status = -1;

    if (to_json()) {
      int sts = JsonPatchAppend(&patch_status);
      if (sts) {
        return -1;
      }
    }
  }  else {
    // logging but not appending - just truncate the file.
    std::fstream fs;
    fs.open(logfile, std::fstream::out);
    bool berror = fs.fail();
    fs.close();
    if (berror) {
      return -1;
    }
    if (to_json()) {
      row = "[";
    }
  }

  // print to log file if requested
  ToFile(row);

  return 0;
}

/**
 * @brief Performs proper termination of log file contents
 *
 * @return 0 - success, non-zero otherwise
 *
 */
int rvs::logger::terminate() {
  // if no logg to file requested, just return
  std::string logfile(log_file);
  if (logfile == "")
    return 0;

  std::string row(RVSENDL);

  if (to_json()) {
    row += "]";
  }

  // print to log file if requested
  ToFile(row);

  return 0;
}

/**
 * @brief Signals that RVS is about to terminate.
 *
 * This method will prevent all further
 * printout to cout and to log file if any. Log file will be closed and properly
 * terminated before returning from this function.
 *
 */
void rvs::logger::Stop(uint16_t flags) {
  // lock cout_mutex for the duration of this block
  std::lock_guard<std::mutex> lk(cout_mutex);

  // signal no further logging to either screen or file
  bStop = true;
  stop_flags = flags;

  // properly terminate log file if needed
  terminate();
}

/**
 * @brief Returns stop flag
 *
 * Checks if a module requested RVS processing to stop
 *
 */
bool rvs::logger::Stopping(void) {
  // lock cout_mutex for the duration of this block
  std::lock_guard<std::mutex> lk(cout_mutex);

  // return stop flag
  return bStop;
}


/**
 * @brief Output Error message
 * 
 * @param Module module where error happened
 * @param Action action where error happened
 * @param Message Message to log
 * @return 0 - success, non-zero otherwise
 *
 */
int rvs::logger::Err(const char* Message, const char* Module
        , const char* Action) {
  if (Message == nullptr) {
    return 1;
  }
  std::string module =
      Module != nullptr ? std::string(" [") + Module + "]" : "";
  std::string action =
      Action != nullptr ? std::string(" [") + Action + "]" : "";
  std::string message = Message;
  std::string out;
  out = "RVS-ERROR";
  out += module + action + std::string(" ") + message;
  {
    // lock cout_mutex for the duration of this block
    std::lock_guard<std::mutex> lk(cout_mutex);
    std::cerr << out << std::endl;
  }
  return 0;
}
