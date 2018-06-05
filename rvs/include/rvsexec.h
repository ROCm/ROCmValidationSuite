
#ifndef RVSEXEC_H_
#define RVSEXEC_H_

#include <map>
#include <string>
#include "yaml-cpp/node/node.h"


using namespace std;


namespace rvs
{

class if1;

class exec
{
public:
	
	typedef map<string,string> options_t;
	exec(const options_t& rOptions);
	~exec();
	
	int run();
	
protected:	
	
	bool 	has_option(const char* pOption, string& val);
	options_t	options;
	
	void 	do_help(void);
	void 	do_version(void);

	int 	do_yaml(const string& config_file);
	int 	do_yaml_properties(const YAML::Node& node, const string& module_name, if1* pif1);
	bool	is_yaml_properties_collection(const string& module_name, const string& proprty_name);
	int 	do_yaml_properties_collection(const YAML::Node& node, const string& parent_name, if1* pif1);

};

	
	
}

#endif // RVSEXEC_H_
