
#ifndef RVSLIBLOG_H_
#define RVSLIBLOG_H_


#ifdef __cplusplus
extern "C" {
#endif

typedef int   (*t_rvs_module_log)(const char*, const int);
typedef void* (*t_cbLogRecordCreate)( const char* Module, const char* Action, const int LogLevel, const unsigned int Sec, const unsigned int uSec);
typedef int   (*t_cbLogRecordFlush)( void* pLogRecord);
typedef void* (*t_cbCreateNode)(void* Parent, const char* Name);
typedef void  (*t_cbAddString)(void* Parent, const char* Key, const char* Val);
typedef void  (*t_cbAddInt)(void* Parent, const char* Key, const int Val);
typedef void  (*t_cbAddNode)(void* Parent, void* Child);

typedef struct tag_module_init 
{
	t_rvs_module_log	cbLog;
	t_cbLogRecordCreate	cbLogRecordCreate;
	t_cbLogRecordFlush 	cbLogRecordFlush;
	t_cbCreateNode 		cbCreateNode;
	t_cbAddString 		cbAddString;
	t_cbAddInt 			cbAddInt;
	t_cbAddNode 		cbAddNode;
	
} T_MODULE_INIT;

#ifdef __cplusplus
}
#endif


namespace rvs
{
	
const int lognone 		= 0;
const int logresults	= 1;
const int logerror		= 2;
const int loginfo		= 3;
const int logdebug		= 4;
const int logtrace		= 5;

}	// namespace rvs






#endif // RVSLIBLOG_H_
