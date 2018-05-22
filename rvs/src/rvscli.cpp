
#include "rvscli.h"


using namespace std;
using namespace rvs;

rvs::cli::cli() 
{
	itoken = 1;
}
rvs::cli::~cli() 
{
}

int rvs::cli::parse(int Argc, char** Argv)
{
	argc = Argc;
	argv = Argv;
	context.push(econtext::eof);
	context.push(econtext::command);
	
	for(;;)
	{
		string token = get_token();
		bool dotoken = true;
		
		while(dotoken)
		{
			econtext top = context.top();
			context.pop();
			
			switch(top)
			{
				
			case econtext::command:
				dotoken = try_command(token);
				break;
			
			case econtext::value:
				dotoken = try_value(token);
				break;
				
			case econtext::eof:
				if( token == "")
				{
					return 0;
				}
				else
				{	
					errstr = "unknown command line argument: " + token; // + get_end_of_command();
					return -1;
				}
				
			default:
					errstr = "syntax error: " + token; // + get_end_of_command();
					return -1;
			}
		}
	}

	return -1;
}

const char* rvs::cli::get_error_string()
{
	return errstr.c_str();
}

const char* rvs::cli::get_token()
{
	if(itoken >= argc)
		return nullptr;
	
	return argv[itoken++];
}

bool rvs::cli::try_command(const string& token)
{
	if( !try_command_l(token))
		return false;
	
	if( !try_command_j(token))
		return false;
	
	return true;
}
bool rvs::cli::try_command_l(const string& token)
{
	return true;
}

bool rvs::cli::try_command_j(const string& token)
{
	return true;
}

bool rvs::cli::try_value(const string& token)
{
	return false;
}


