
#include "rvs_module.h"

#include "action.h"

using namespace std;

action::action()
{
}

action::~action()
{
	property.clear();
}

int action::property_set(const char* Key, const char* Val)
{
	return rvs::lib::actionbase::property_set(Key, Val);
}

int action::run(void)
{
	log((char*)"Hello from GPUP action::run()");
	
	return 0;
}