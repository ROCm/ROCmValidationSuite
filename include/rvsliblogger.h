
#ifndef RVSLIBLOGGER_H_
#define RVSLIBLOGGER_H_

#include <string>
#include "rvsliblog.h"

using namespace std;

extern "C" int LoggerCallback(const char* Message, const int level);

namespace rvs
{

namespace lib
{
	

class logger
{
public:
	logger();
	~logger();

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
protected:
	static	int 	loglevel_m;
	static	bool 	tojason_m;
	static	bool 	append_m;
	static	string 	logfile_m;

	static  const char* 	loglevelname[6];
};

}	// namespace lib
}	// namespace rvs






#endif // RVSLIBLOGGER_H_
