
#include "rvsoptions.h"

using namespace std;

map<string,string>	rvs::options::opt;

bool  rvs::options::has_option(const string& Option, string& val) {
	auto it = opt.find(string(Option));
	if( it == opt.end())
		return false;

	val = it->second;
	return true;
}

bool  rvs::options::has_option(const string& Option) {
	auto it = opt.find(string(Option));
	if( it == opt.end())
		return false;

	return true;
}

const std::map<std::string,std::string>& rvs::options::get(void) {
  return opt;
}


