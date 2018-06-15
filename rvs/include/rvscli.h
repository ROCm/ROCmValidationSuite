
#include <memory>
#include <map>
#include <stack>
#include <string>

using namespace std;

namespace rvs
{



class cli
{

public:
	
			cli();
	virtual ~cli();

	int 	parse(int Argc, char** Argv);
	const 	char* 	get_error_string();

protected:
	typedef enum {eof, value, command} 			econtext;

	class optbase
	{

	public:
							optbase(const char* ptruename, econtext s1, econtext s2 = eof, econtext s3 = eof);
		virtual				~optbase();		

		virtual bool 		adjust_context(stack<econtext>& old_context);
	public:
		string 				name;
		stack<econtext> 	new_context;
	};
	
	typedef pair<string,shared_ptr<optbase>> 	gpair;

protected:
	const char*	get_token();
	bool 		is_command(const string& token);
	bool 		try_command(const string& token);
	bool 		try_value(const string& token);
	bool 		emit_option(void);
	void		store_command(const string& token);
	void		store_value(const string& token);
	void 		init_grammar(void);
			
	
protected:
	int 					argc;
	char**					argv;
	int						itoken;
	string					errstr;
	string 					current_option;
	string 					current_value;
	stack<econtext> 					context;
	map<string,shared_ptr<optbase>> 	grammar;
	
};



}	// namespace rvs
