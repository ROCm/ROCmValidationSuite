

#ifndef RVSIF0_H_
#define RVSIF0_H_


#include "rvsmodule_if0.h"
#include "rvsif_base.h"

namespace rvs 
{

class if0 : public ifbase
{
public:
	virtual ~if0();
	virtual void    get_version(int*, int*, int*);
	virtual char*   get_name(void);
	virtual char*   get_description(void);
	virtual int     has_interface(int);

protected:
	if0();
	if0(const if0&);
	
virtual if0& operator= (const if0& rhs);
virtual ifbase* clone(void);

protected:
	t_rvs_module_get_version		rvs_module_get_version;
	t_rvs_module_get_name        	rvs_module_get_name;
	t_rvs_module_get_description	rvs_module_get_description;
	t_rvs_module_has_interface		rvs_module_has_interface;
	
friend class module;

};

}	// namespace rvs

#endif /* RVSIF0_H_ */
