
#include "rvslognodestring.h"

using namespace std;

rvs::LogNodeString::LogNodeString( const string& Name, const string& Val, const LogNodeBase* Parent)
: 
LogNodeBase(Name, Parent),
Value(Val)
{
	Type = eLN::String;
}

rvs::LogNodeString::LogNodeString( const char* Name, const char* Val, const LogNodeBase* Parent)
:
LogNodeBase(Name, Parent),
Value(Val)
{
	Type = eLN::String;
}
	
rvs::LogNodeString::~LogNodeString()
{
}


std::string rvs::LogNodeString::ToJson(const std::string& Lead) {

	string result(RVSENDL);
	result += Lead + "\"" + Name + "\"" + " : " + "\"" + Value + "\"";
	
	return result;
}