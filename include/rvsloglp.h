
#ifndef RVSLOGGERPROXY_H_
#define RVSLOGGERPROXY_H_

#include <string>

#include "rvsliblog.h"

using namespace std;

namespace rvs
{
	

class lp
{

public:
	lp();
	~lp();

	static int   Log(const char* pMsg, const int level);
	static int   Initialize(const T_MODULE_INIT* pMi);
	static void* LogRecordCreate( const char* Module, const char* Action, const int LogLevel, const unsigned int Sec, const unsigned int uSec);
	static void* LogRecordCreate( const string Module, const string Action, const int LogLevel = rvs::logresults);
	static void* LogRecordCreate( const char* Module, const char* Action, const int LogLevel = rvs::logresults);
	static int   LogRecordFlush( void* pLogRecord);
	static void* CreateNode(void* Parent, const char* Name);
	static void  AddString(void* Parent, const string& Key, const string& Val);
	static void  AddString(void* Parent, const char* Key, const char* Val);
	static void  AddInt(void* Parent, const char* Key, const int Val);
	static void  AddNode(void* Parent, void* Child);


protected:
	static T_MODULE_INIT mi;
	
};


}	// namespace rvs

#endif // RVSLOGGERPROXY_H_
