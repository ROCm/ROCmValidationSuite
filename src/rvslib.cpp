

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

bool rvs::lib::actionbase::has_property(const std::string& key, std::string& val) {

  for (auto it = property.begin(); it != property.end(); ++it) {
    if (it->first == key) {
      val = it->second;
      return true;
    }
  }

  return false;
}

bool rvs::lib::actionbase::has_property(const std::string& key) {
  string val;
  return has_property(key, val);
}
