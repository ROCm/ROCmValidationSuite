
#ifndef RVSIF1_H_
#define RVSIF1_H_

#include <string>

#include "rvsmodule_if1.h"
#include "rvsif_base.h"

using namespace std;

namespace rvs 
{
	
class if1 : public ifbase
{
public:
	virtual ~if1();
    virtual int   property_set(const char*, const char*);
    virtual int   property_set(const string&, const string&);
	virtual int   run(void);
	virtual char* get_errstring(int);

protected:
	if1();
	if1(const if1&);
	
virtual if1& operator= (const if1& rhs);
virtual ifbase* clone(void);

protected:
    t_rvs_module_action_property_set 	rvs_module_action_property_set;
	t_rvs_module_action_run				rvs_module_action_run;
	t_rvs_module_get_errstring			rvs_module_get_errstring;

friend class module;
};

} // namespace rvs
#endif /* RVSIF1_H_ */
