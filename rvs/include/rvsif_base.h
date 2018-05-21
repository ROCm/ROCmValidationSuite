

#ifndef RVSIF_BASE_H_
#define RVSIF_BASE_H_


namespace rvs 
{


class ifbase
{
public:
	virtual ~ifbase();

protected:
			ifbase();
			ifbase(const ifbase& rhs);
	
virtual	ifbase& operator=(const ifbase& rhs);
virtual ifbase* clone(void);

protected:
	void*	plibaction;

friend class module;

};

}	// namespace rvs

#endif /* RVSIF_BASE_H_ */
