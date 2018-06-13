
#include "rvslognode.h"

using namespace std;

rvs::LogNode::LogNode( const std::string& Name, const LogNodeBase* Parent)
:
LogNodeBase(Name, Parent)
{
	Type = eLN::List;
}

rvs::LogNode::LogNode( const char* Name, const LogNodeBase* Parent)
:
LogNodeBase(Name, Parent)
{
	Type = eLN::List;
}

rvs::LogNode::~LogNode()
{
	for(auto it = Child.begin(); it != Child.end(); ++it) {
		delete (*it);
	}

}

void rvs::LogNode::Add(LogNodeBase* pChild)
{
	Child.push_back(pChild);
}

std::string rvs::LogNode::ToJson(const std::string& Lead) {

	string result(RVSENDL);
	result += Lead + "\"" + Name + "\"" + " : {";
	
	int  size = Child.size();
	for(int i = 0; i < size; i++) {
		result += Child[i]->ToJson(Lead + RVSINDENT);
		if( i+ 1 < size) {
			result += ",";
		}
	}
	result += Lead + "}";
	
	return result;
}