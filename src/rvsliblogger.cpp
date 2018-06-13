
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
bool 	rvs::logger::tojason_m;
bool 	rvs::logger::append_m;
bool	rvs::logger::isfirstrecord_m;
string 	rvs::logger::logfile_m;

const char*	rvs::logger::loglevelname[] = {"NONE  ", "RESULTS", "ERROR ", "INFO  ", "DEBUG ", "TRACE " };

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

void rvs::logger::to_jason(const bool flag)
{
	tojason_m = flag;
}
bool rvs::logger::to_jason()
{
	return tojason_m;
}


int rvs::logger::log(const char* Message, const int level)
{
	if( level < lognone || level > logtrace)
	{
		cerr << "ERROR: unknown logging level: " << level << endl;
		return -1;
	}
	
	if( level > loglevel_m)
		return 0;
	
	uint32_t 	secs;
	uint32_t 	usecs;
	char	buff[64];
	get_ticks(secs, usecs);
	sprintf(buff,"%6d.%6d", secs, usecs);
	
// 	std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();
// 	std::time_t now_c = std::chrono::system_clock::to_time_t(now);
	
	cout << "[" <<loglevelname[level] << "]" << " [" << buff << "] " << Message << endl;
	
	return 0;
}

int rvs::logger::log(const string& Message, const int level)
{
	return log( Message.c_str(), level);
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

  int level = r->LogLevel();
	if( level < lognone || level > logtrace)
	{
		cerr << "ERROR: unknown logging level: " << level << endl;
    delete r;
		return -1;
	}

	if( level > loglevel_m)
  {
    delete r;
		return 0;
  }

  string row;

  if (tojason_m) {
    if( isfirstrecord_m) {
      isfirstrecord_m = false;
    }
    else {
      row = ",";
    }
    row += r->ToJson("  ");
  }
  else {
    delete r;
    return 0;
TODO("fix this")
//     if( isfirstrecord_m) {
//       isfirstrecord_m = false;
//     }
//     row += r->ToJson("  ");
  }


  // no --quiet option given?
  if( !rvs::options::has_option("-q", val)) {
    cout << row;
  }

  if( rvs::options::has_option("-l", val)) {
    ToFile(row);
  }

  delete r;

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


int rvs::logger::Log(const char* Message, const int level)
{
  if (tojason_m)
    return 0;

	return rvs::logger::log(Message, level);
}


int rvs::logger::initialize()
{
  isfirstrecord_m = true;

  std::string row;
  std::string val;

  // logging but no appending - just truncate the file.
  if (!append() && rvs::options::has_option("-l", val)) {
    std::fstream fs;
    fs.open (val, std::fstream::out);
    fs.close();
  }

  int patch_status = -1;
  if( to_jason() && append() && rvs::options::has_option("-l", val)) {
    patch_status = JsonPatchAppend();
  }

  if( (patch_status) && to_jason()) {
    row += "[";
  }

  // if not "quiet" print to screen
  if (!rvs::options::has_option("-q", val)) {
    cout << row;
  }

  // print to log file if requested
  ToFile(row);

  return 0;
}

int rvs::logger::terminate()
{
  std::string row(RVSENDL);
  std::string val;

  if( to_jason()) {
    row += "]";
  }

  // if not "quiet" print to screen
  if (!rvs::options::has_option("-q", val)) {
    cout << row << endl;
  }

  // print to log file if requested
  ToFile(row);

  return 0;
}