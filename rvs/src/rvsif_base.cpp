
#include "rvsif_base.h"

rvs::ifbase::ifbase()
:plibaction(nullptr)
{
}

rvs::ifbase::~ifbase()
{
}

rvs::ifbase::ifbase(const ifbase& rhs)
{
	*this = rhs;
}

rvs::ifbase& rvs::ifbase::operator=(const rvs::ifbase& rhs) // copy assignment
{
    if (this != &rhs) { // self-assignment check expected
        plibaction = rhs.plibaction;
    }
    return *this;
}

rvs::ifbase* rvs::ifbase::clone(void)
{
	return new rvs::ifbase(*this);
}
