
#include "rvsliblogger.h"

#include <unistd.h>
#include <time.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <stdio.h>

#include "rvslognode.h"
#include "rvslognodestring.h"
#include "rvslognodeint.h"
#include "rvslognoderec.h"
#include "rvsoptions.h"


int 	rvs::logger::loglevel_m(2);
bool 	rvs::logger::tojson_m;
bool 	rvs::logger::append_m;
bool	rvs::logger::isfirstrecord_m;
string 	rvs::logger::logfile_m;

const char*	rvs::logger::loglevelname[] = {"NONE  ", "RESULT", "ERROR ", "INFO  ", "DEBUG ", "TRACE " };

rvs::logger::logger()
{
	isfirstrecord_m = true;
}

rvs::logger::~logger()
{
}

void rvs::logger::append(const bool flag) {
	append_m = flag;
}

bool rvs::logger::append() {
	return append_m;
}

void rvs::logger::logfile(const string& val) {
	logfile_m = val;
}

const string& rvs::logger::logfile() {
	return logfile_m;
}

void rvs::logger::log_level(const int rLevel)
{
	loglevel_m = rLevel;
}

int rvs::logger::log_level()
{
	return loglevel_m;
}

bool rvs::logger::get_ticks(uint32_t& secs, uint32_t& usecs) 
{
    struct timespec ts;

	clock_gettime( CLOCK_MONOTONIC, &ts );
    usecs  	= ts.tv_nsec / 1000;
    secs	= ts.tv_sec;
	
    return true;
}

void rvs::logger::to_json(const bool flag)
{
	tojson_m = flag;
}

bool rvs::logger::to_json()
{
	return tojson_m;
}

int rvs::logger::log(const string& Message, const int LogLevel) {
	 return LogExt(Message.c_str(), LogLevel, 0, 0);
}

int rvs::logger::Log(const char* Message, const int LogLevel) {
	return LogExt(Message, LogLevel, 0, 0);
}

int rvs::logger::LogExt(const char* Message, const int LogLevel, const unsigned int Sec, const unsigned int uSec) {
	if( LogLevel < lognone || LogLevel > logtrace)
	{
		cerr << "ERROR: unknown logging level: " << LogLevel << endl;
		return -1;
	}
	
	// log level too high?
	if( LogLevel > loglevel_m)
		return 0;

	uint32_t 	secs;
	uint32_t 	usecs;

  if( (Sec|uSec)) {
		secs = Sec;
		usecs = uSec;
  } else {
    get_ticks(secs, usecs);
  }

	char	buff[64];
	sprintf(buff,"%6d.%6d", secs, usecs);
	
// 	std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();
// 	std::time_t now_c = std::chrono::system_clock::to_time_t(now);
	
  string row("[");
  row += loglevelname[LogLevel];
  row +="] [";
  row += buff;
  row +="] ";
  row += Message;

  // if no quiet option given, output to cout
  if (!rvs::options::has_option("-q")) {
    cout << row << endl;
  }

  // this stream does not output JSON
  if (to_json())
    return 0;
	
  // send to file if requested
  if (isfirstrecord_m) {
    isfirstrecord_m = false;
  }
  else {
    row = RVSENDL + row;
  }
  ToFile(row);

	return 0;
}

void* rvs::logger::LogRecordCreate( const char* Module, const char* Action, const int LogLevel, const unsigned int Sec, const unsigned int uSec) {

	uint32_t 	sec;
	uint32_t 	usec;

	if ((Sec|uSec)) {
		sec = Sec;
		usec = uSec;
	}
	else  {
		get_ticks(sec, usec);
	}

	rvs::LogNodeRec* rec = new LogNodeRec(Action, LogLevel, sec, usec);
	AddString(rec, "action", Action);
	AddString(rec, "module", Module);
	AddString(rec, "loglevelname", (LogLevel >= lognone && LogLevel < logtrace) ? loglevelname[LogLevel] : "UNKNOWN");
	
	return static_cast<void*>(rec);
}

int   rvs::logger::LogRecordFlush( void* pLogRecord) {
  string val;

  LogNodeRec* r = static_cast<LogNodeRec*>(pLogRecord);
  // no JSON loggin requested
  if (!to_json()) {
    delete r;
    return 0;
  }

  // assert log levl
  int level = r->LogLevel();
	if( level < lognone || level > logtrace)
	{
		cerr << "ERROR: unknown logging level: " << level << endl;
    delete r;
		return -1;
	}

	// if too high, ignore record
	if( level > loglevel_m)
  {
    delete r;
		return 0;
  }

  // do not pre-pend "," separator for the first row
  string row;
  if( isfirstrecord_m) {
    isfirstrecord_m = false;
  }
  else {
    row = ",";
  }

  // get JSON formatted log record
  row += r->ToJson("  ");

  // send it to file
  ToFile(row);

  // dealloc memory
  delete r;

  // return OK
	return 0;
}

int rvs::logger::ToFile(const string& Row) {

  string logfile;
  if( !rvs::options::has_option("-l", logfile))
    return -1;

  std::fstream fs;
  fs.open (logfile, std::fstream::out | std::fstream::app);

  fs << Row;

  fs.close();

}

int rvs::logger::JsonPatchAppend() {

  string logfile;
  if( !rvs::options::has_option("-l", logfile))
    return -1;

  FILE * pFile;
  pFile = fopen ( logfile.c_str() , "r+" );
  fseek ( pFile , -1 , SEEK_END );
  fputs ( "," , pFile );
  fclose ( pFile );
  return 0;
}

void* rvs::logger::CreateNode(void* Parent, const char* Name) {
	rvs::LogNode* p = new LogNode(Name, static_cast<rvs::LogNodeBase*>(Parent));
	return p;
}
void  rvs::logger::AddString(void* Parent, const char* Key, const char* Val) {
	rvs::LogNode* pp = static_cast<rvs::LogNode*>(Parent);
	rvs::LogNodeString* p = new LogNodeString(Key, Val, pp);
	pp->Add(p);
}
void  rvs::logger::AddInt(void* Parent, const char* Key, const int Val) {
	rvs::LogNode* pp = static_cast<rvs::LogNode*>(Parent);
	rvs::LogNodeInt* p = new LogNodeInt(Key, Val, pp);
	pp->Add(p);
}

void  rvs::logger::AddNode(void* Parent, void* Child) {
	rvs::LogNode* pp = static_cast<rvs::LogNode*>(Parent);
	pp->Add(static_cast<rvs::LogNodeBase*>(Child));
}


int rvs::logger::initialize()
{
  isfirstrecord_m = true;

  std::string row;
  std::string logfile;

  // if no logg to file requested, just return
  if (!rvs::options::has_option("-l", logfile))
    return 0;

  if (append()) {

    // appnd to file, replace the closing "]" with "," in order to
    // have well formed JSON after appending
    int patch_status = -1;

    if( to_json()) {
      patch_status = JsonPatchAppend();
      if( (patch_status)) {
        row += "[";
      }
    }
  }  else {

    // logging but not appending - just truncate the file.
    std::fstream fs;
    fs.open (logfile, std::fstream::out);
    fs.close();
    if (to_json()) {
      row = "[";
    }
  }

  // print to log file if requested
  ToFile(row);

  return 0;
}

int rvs::logger::terminate()
{
  // if no logg to file requested, just return
  if (!rvs::options::has_option("-l"))
    return 0;

  std::string row(RVSENDL);

  if( to_json()) {
    row += "]";
  }

  // print to log file if requested
  ToFile(row);

  return 0;
}