

#include "rvsaction.h"

using namespace std;


rvs::action::action(const char* pName, void* pLibAction)
{
	name 		= pName;
	plibaction 	= pLibAction;
}

rvs::action::~action()
{
	ifmap.clear();
}

rvs::ifbase* rvs::action::get_interface(int i)
{
	auto it = ifmap.find(i);
	if( it != ifmap.end())
		return &(*(it->second));
	
	return nullptr;
}
