

#include <rvslib.h>

#include <chrono>
#include <unistd.h>

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

void rvs::lib::actionbase::sleep(const unsigned int ms) {
	::usleep(1000*ms);
}
