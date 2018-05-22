
#include "rvsliblogger.h"

#include <iostream>

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

int rvs::lib::logger::log(const char* Message, const int level)
{
	if( level < lognone || level > logtrace)
	{
		log((char*)"Unknown logging level.", logerror);
		return -1;
	}
	
	if( level > loglevel_m)
		return 0;
	
	
	cout << loglevelname[level] << " " << Message << endl;
	
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