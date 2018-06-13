
#include "rvsoptions.h"

using namespace std;

map<string,string>	rvs::options::opt;

bool  rvs::options::has_option(const string& pOptions, string& val) {
	auto it = opt.find(string(pOptions));
	if( it == opt.end())
		return false;

	val = it->second;
	return true;
}

const std::map<std::string,std::string>& rvs::options::get(void) {
  return opt;
}


