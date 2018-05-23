

#include <unistd.h>
#include <time.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include "rvsliblogger.h"

int 	rvs::lib::logger::loglevel_m(2);
bool 	rvs::lib::logger::tojason_m;
bool 	rvs::lib::logger::append_m;
string 	rvs::lib::logger::logfile_m;

const char*	rvs::lib::logger::loglevelname[] = {"[NONE  ]", "[RESULTS]", "[ERROR ]", "[DEBUG ]", "[INFO  ]", "[TRACE ]" };

rvs::lib::logger::logger()
{
}

rvs::lib::logger::~logger()
{
}

bool rvs::lib::logger::get_ticks(uint32_t& secs, uint32_t& usecs) 
{
    struct timespec ts;

	clock_gettime( CLOCK_MONOTONIC, &ts );
    usecs  	= ts.tv_nsec / 1000;
    secs	= ts.tv_sec;
	
    return true;
}

int rvs::lib::logger::log(const char* Message, const int level)
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
	
	cout << loglevelname[level] << " [" << buff << "] " << Message << endl;
	
	return 0;
}

int rvs::lib::logger::log(const string& Message, const int level)
{
	return log( Message.c_str(), level);
}


extern "C" int LoggerCallback(const char* Message, const int level)
{
	return rvs::lib::logger::log(Message, level);
}