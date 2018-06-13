
#include "rvslognodeint.h"

using namespace std;

rvs::LogNodeInt::LogNodeInt( const string& Name, const int Val, const LogNodeBase* Parent)
: 
LogNodeBase(Name, Parent),
Value(Val)
{
	Type = eLN::Integer;
}

rvs::LogNodeInt::LogNodeInt( const char* Name, const int Val, const LogNodeBase* Parent)
:
LogNodeBase(Name, Parent),
Value(Val)
{
	Type = eLN::Integer;
}
	
rvs::LogNodeInt::~LogNodeInt()
{
}

std::string rvs::LogNodeInt::ToJson(const std::string& Lead) {

	string result(RVSENDL);
	result += Lead + "\"" + Name + "\"" + " : " + std::to_string(Value);
	
	return result;
}
