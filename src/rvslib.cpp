

#include <rvslib.h>

using namespace std;

rvs::lib::actionbase::actionbase()
{
}

rvs::lib::actionbase::~actionbase()
{
}

int rvs::lib::actionbase::property_set(const char* pKey, const char* pVal)
{
	property.insert( pair<string, string>(pKey, pVal));
	return 0;
}