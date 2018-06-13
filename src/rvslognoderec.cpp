
#include "rvslognoderec.h"

using namespace std;


rvs::LogNodeRec::LogNodeRec( const string& Name, const int LoggingLevel, const unsigned int Sec, const unsigned int uSec, const LogNodeBase* Parent)
:
LogNode(Name, Parent),
Level(LoggingLevel),
sec(Sec),
usec(uSec)
{
	Type = eLN::Record;
}

rvs::LogNodeRec::LogNodeRec( const char* Name, const int LoggingLevel, const unsigned int Sec, const unsigned int uSec, const LogNodeBase* Parent)
:
LogNode(Name, Parent),
Level(LoggingLevel),
sec(Sec),
usec(uSec)
{
	Type = eLN::Record;
}

rvs::LogNodeRec::~LogNodeRec()
{
}

const int rvs::LogNodeRec::LogLevel()
{
	return Level;
}

std::string rvs::LogNodeRec::ToJson(const std::string& Lead) {

	string result(RVSENDL);
	result += RVSENDL + Lead + "{";
	
	result += RVSENDL; 
	result += Lead + RVSINDENT;
	result +=  string("\"") + "loglevel" + "\"" + " : " + std::to_string(Level) + ",";
	
	int  size = Child.size();
	for(int i = 0; i < size; i++) {
//		result += RVSENDL;
		result += Child[i]->ToJson(Lead + RVSINDENT);
		if( i+ 1 < size) {
			result += ",";
		}
	}
	result += RVSENDL + Lead + "}";
	
	return result;
}