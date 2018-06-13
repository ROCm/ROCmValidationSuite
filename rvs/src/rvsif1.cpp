
#include "rvsif1.h"
#include "rvsliblogger.h"

using namespace std;

rvs::if1::if1()
: rvs_module_action_property_set(nullptr),
  rvs_module_action_run(nullptr),
  rvs_module_get_errstring(nullptr)

{
}

rvs::if1::~if1()
{
}

rvs::if1::if1(const if1& rhs)
{
	*this = rhs;
}

rvs::if1& rvs::if1::operator=(const rvs::if1& rhs) // copy assignment
{
	// self-assignment check
    if (this != &rhs) 
	{
		ifbase::operator=(rhs);
		rvs_module_action_property_set 	= rhs.rvs_module_action_property_set;
		rvs_module_action_run			= rhs.rvs_module_action_run;
		rvs_module_get_errstring		= rhs.rvs_module_get_errstring;
    }
    
    return *this;
}

rvs::ifbase* rvs::if1::clone(void)
{
	return new rvs::if1(*this);
}


int rvs::if1::property_set(const char* pKey, const char* pVal )
{
	rvs::logger::log(string("poperty: [") + string(pKey) + string("]   val:[")+string(pVal)+string("]"), rvs::logtrace);
	return (*rvs_module_action_property_set)(plibaction, pKey, pVal);
}

int rvs::if1::property_set( const string& Key, const string& Val)
{
	return property_set( Key.c_str(), Val.c_str());
}

int rvs::if1::run(void)
{
	return (*rvs_module_action_run)(plibaction);
}

char* rvs::if1::get_errstring(int iErrCode)
{
	return (*rvs_module_get_errstring)(iErrCode);
}
