

#ifndef RVSACTION_H_
#define RVSACTION_H_

#include <memory>
#include <map>
#include <string>

namespace rvs 
{

class ifbase;
class module;

class action
{
	typedef std::pair< int, std::shared_ptr<ifbase> > t_impair;
public:
	virtual rvs::ifbase*     get_interface(int);

protected:
	action(const char* pName, void* pLibAction);
	virtual ~action();

protected:
	std::string	name;
	void*		plibaction;
	std::map< int, std::shared_ptr<ifbase> >  ifmap;

friend class module;
};


}	// namespace rvs 


#endif /* RVSACTION_H_ */
