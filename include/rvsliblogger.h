
#ifndef RVSLIBLOGGER_H_
#define RVSLIBLOGGER_H_

#include <string>
#include "rvsliblog.h"

using namespace std;


namespace rvs
{


class logger
{
protected:
  logger();
  ~logger();

public:
	static	void 	log_level(const int level);
	static	int		log_level();

	static	void 	to_jason(const bool flag);
	static	bool	to_jason();

	static	void 	append(const bool flag);
	static	bool 	append();

	static	void 	logfile(const string& filename);
	static	const 	string& logfile();
		
	static	int		log(const char* Message, const int level = 1);
	static	int 	log(const string& Message, const int level = 1);

	static	bool 	get_ticks(uint32_t& secs, uint32_t& usecs);
	
	static	int		initialize();
	static  int		terminate();

  static  int   Log(const char* Message, const int level);
  static 	void* LogRecordCreate( const char* Module, const char* Action, const int LogLevel, const unsigned int Sec, const unsigned int uSec);
  static 	int   LogRecordFlush( void* pLogRecord);
  static 	void* CreateNode(void* Parent, const char* Name);
  static 	void  AddString(void* Parent, const char* Key, const char* Val);
  static 	void  AddInt(void* Parent, const char* Key, const int Val);
  static 	void  AddNode(void* Parent, void* Child);
  static  int   LogToFile(const string& Val);
  static  int   ToFile(const string& Row);
  static  int   JsonPatchAppend(void);

protected:
	static	int 	loglevel_m;
	static	bool 	tojason_m;
	static	bool 	append_m;
	static	bool 	isfirstrecord_m;
	static	string 	logfile_m;

	static  const char* 	loglevelname[6];
};

}	// namespace rvs






#endif // RVSLIBLOGGER_H_
