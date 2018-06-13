
#ifndef RVSLOGGERPROXY_H_
#define RVSLOGGERPROXY_H_

#include "liblog.h"


namespace rvs
{
	

class lp

public:
	lp();
	~lp();
	
	static int 		Initialize(const T_MODULE_INIT* pInitStruct);
	static void* 	LogRecordCreate( const char* Module, const char* Action, const int LogLevel);
	static int   	LogRecordFlush( void* pLogRecord);
	static void* 	CreateNode(void* Parent, const char* Name);
	static void  	AddString(void* Parent, const char* Key, const char* Val);
	static void  	AddInt(void* Parent, const char* Key, const int Val);
	static void  	AddNode(void* Parent, const void* Child);


protected:	
	static T_MODULE_INIT If;
	
	


}	// namespace rvs

#endif // RVSLOGGERPROXY_H_
