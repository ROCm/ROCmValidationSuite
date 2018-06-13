

#include <iostream>


#include "rvscli.h"
#include "rvsexec.h"


using namespace std;
using namespace rvs;


int main(int Argc, char**Argv)
{
	int sts;
	cli cli;
	
	sts =  cli.parse(Argc, Argv);
	if(sts)
	{
		cerr << "ERROR: error parsing command line:" << cli.get_error_string() << endl;
		return -1;
	}
    
  exec executor;
	sts = executor.run();
	if(sts)
	{
		cerr << "ERROR: error executing configuration: " << sts << endl;
	}
	    
	return sts;
}




