

#ifndef ACTION_H_
#define ACTION_H_

#include "rvsactionbase.h"

class action : public rvs::actionbase
{
public:
  action();
  virtual ~action();

  virtual int run(void);

protected:
  int do_gpu_list(void);
};

#endif /* ACTION_H_ */
