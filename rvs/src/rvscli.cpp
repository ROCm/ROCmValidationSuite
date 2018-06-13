
#include "rvscli.h"

#include <stdio.h>
#include <iostream>


#include "rvsoptions.h"



using namespace std;
using namespace rvs;

rvs::cli::optbase::optbase(const char* ptruename, econtext s1, econtext s2, econtext s3)
{
	name = ptruename;
	new_context.push(eof);
	new_context.push(s1);
	if( s2 != eof) new_context.push(s2);
	if( s3 != eof) new_context.push(s3);
}

rvs::cli::optbase::~optbase()
{
	
}

bool rvs::cli::optbase::adjust_context(stack<econtext>& old_context)
{
	while(!old_context.empty())
		old_context.pop();
	old_context = new_context;
	
	return true;
}

rvs::cli::cli() 
{
	itoken = 1;
}
rvs::cli::~cli() 
{
}

void rvs::cli::init_grammar()
{
	shared_ptr<optbase> sp;
	
	grammar.clear();
	sp = make_shared<optbase>("--statspath", value);
	grammar.insert(gpair("--statspath", sp));

	sp = make_shared<optbase>("-a", command);
	grammar.insert(gpair("-a", sp));
	grammar.insert(gpair("--appendLog", sp));

	sp = make_shared<optbase>("-c", command, value);
	grammar.insert(gpair("-c", sp));
	grammar.insert(gpair("--config", sp));

	sp = make_shared<optbase>("--configless", command);
	grammar.insert(gpair("--configless", sp));

	sp = make_shared<optbase>("-d", command, value);
	grammar.insert(gpair("-d", sp));
	grammar.insert(gpair("--debugLevel", sp));

	sp = make_shared<optbase>("-g", command);
	grammar.insert(gpair("-g", sp));
	grammar.insert(gpair("--listGpus", sp));

	sp = make_shared<optbase>("-i", value);
	grammar.insert(gpair("-i", sp));
	grammar.insert(gpair("--indexes", sp));

	sp = make_shared<optbase>("-j", command);
	grammar.insert(gpair("-j", sp));
	grammar.insert(gpair("--json", sp));

	sp = make_shared<optbase>("-l", command, value);
	grammar.insert(gpair("-l", sp));
	grammar.insert(gpair("--debugLogFile", sp));

	sp = make_shared<optbase>("-q", command);
	grammar.insert(gpair("--quiet", sp));

	sp = make_shared<optbase>("-m", value);
	grammar.insert(gpair("-m", sp));
	grammar.insert(gpair("--modulepath", sp));

	sp = make_shared<optbase>("-s", command);
	grammar.insert(gpair("-s", sp));
	grammar.insert(gpair("--scriptable", sp));

	sp = make_shared<optbase>("-st", value);
	grammar.insert(gpair("--specifiedtest", sp));

	sp = make_shared<optbase>("-sf", command);
	grammar.insert(gpair("--statsonfail", sp));

	sp = make_shared<optbase>("-t", command);
	grammar.insert(gpair("-t", sp));
	grammar.insert(gpair("--listTests", sp));
	
	sp = make_shared<optbase>("-v", command);
	grammar.insert(gpair("-v", sp));
	grammar.insert(gpair("--verbose", sp));
	
	sp = make_shared<optbase>("-ver", command);
	grammar.insert(gpair("--version", sp));
	
	sp = make_shared<optbase>("-h", command);
	grammar.insert(gpair("-h", sp));
	grammar.insert(gpair("--help", sp));

}

int rvs::cli::parse(int Argc, char** Argv)
{
	init_grammar();

	argc = Argc;
	argv = Argv;
	context.push(econtext::eof);
	context.push(econtext::command);
	
	for(;;)
	{
		string token = get_token();
		bool token_done = false;
		
		while(!token_done)
		{
			econtext top = context.top();
			context.pop();
			
			switch(top)
			{
				
			case econtext::command:
				token_done = try_command(token);
				break;
			
			case econtext::value:
				token_done = try_value(token);
				if (!token_done)
				{
					errstr = string("syntax error: value expected after ") +  current_option;
					return -1;
				}
				break;
				
			case econtext::eof:
				if( token == "")
				{
					emit_option();
					return 0;
				}
				else
				{	
					errstr = "unexpected command line argument: " + token; // + get_end_of_command();
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
		return (const char*)"";
	
	return argv[itoken++];
}

bool rvs::cli::is_command(const string& token)
{
	auto it = grammar.find(token);
	if( it == grammar.end())
		return false;
	
	return true;
}

bool rvs::cli::emit_option()
{
	// emit previous option and its value (if andy)
	if( current_option != "")
	{
		options::opt[current_option] = current_value;
	}
	
	// reset working buffer
	current_option = "";
	current_value  = "";
	
	return true;
}


bool rvs::cli::try_command(const string& token)
{
	auto it = grammar.find(token);
	if( it == grammar.end())
		return false;
	
	// emit previous buffer contents (if any)
	emit_option();

	// token identified as command, so store it:
	current_option = token;
	
	// fill context  stack with new possible continuations:
	it->second->adjust_context(context);

	return true;
}

bool rvs::cli::try_value(const string& token)
{
	if( token == "")
		return false;
	
	//	should not be one of command line options
	auto it = grammar.find(token);
	if( it != grammar.end())
		return false;
	
	// token is value for previous command
	current_value = token;
	
	// emit previous option-value pair:
	emit_option();
	
	return true;
}

