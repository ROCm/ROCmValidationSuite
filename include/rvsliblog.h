
#ifndef RVSLIBLOG_H_
#define RVSLIBLOG_H_


extern "C" typedef int   (*t_rvs_module_log)(const char*, const int);


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
