// Copyright [year] <Copyright Owner> ... goes here
#ifndef PEQT_SO_INCLUDE_ACTION_H_
#define PEQT_SO_INCLUDE_ACTION_H_

#include "rvslib.h"

class action: public rvs::lib::actionbase {
 public:
    action();
    virtual ~action();

    virtual int property_set(const char*, const char*);
    virtual int run(void);

 protected:
};

#endif  // PEQT_SO_INCLUDE_ACTION_H_
