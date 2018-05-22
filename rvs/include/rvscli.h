
#include <stdio.h>
#include <iostream>
#include <map>
#include <stack>
#include <string>

using namespace std;

namespace rvs
{


class cli
{
typedef enum {eof, value, command} econtext;

public:
			cli();
	virtual ~cli();

	int 	parse(int Argc, char** Argv);
	const 	char* 	get_error_string();

protected:
	const 	char*	get_token();
	bool 	try_command(const string& token);
	bool 	try_value(const string& token);
	bool 	try_command_l(const string& token);
	bool 	try_command_j(const string& token);
	void	store_command(const string& token);
	void	store_value(const string& token);
			
	
protected:
	int 				argc;
	char**				argv;
	int					itoken;
	string				errstr;
	stack<econtext> 	context;
	map<string,string> 	options;
	
};



}	// namespace rvs
