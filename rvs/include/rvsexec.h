
#ifndef RVSEXEC_H_
#define RVSEXEC_H_

#include <string>
#include "yaml-cpp/node/node.h"


using namespace std;


namespace rvs
{

class if1;

class exec
{
public:
	
	exec();
	~exec();
	
	int run();
	
protected:	

	void 	do_help(void);
	void 	do_version(void);

	int 	do_yaml(const string& config_file);
	int 	do_yaml_properties(const YAML::Node& node, const string& module_name, if1* pif1);
	bool	is_yaml_properties_collection(const string& module_name, const string& proprty_name);
	int 	do_yaml_properties_collection(const YAML::Node& node, const string& parent_name, if1* pif1);

};

	
	
}

#endif // RVSEXEC_H_
