

#ifndef ACTION_H_
#define ACTION_H_

#include "rvsactionbase.h"

/**
 * @class action
 * @ingroup PQT
 *
 * @brief PQT action implementation class
 *
 * Derives from rvs::actionbase and implements actual action functionality
 * in its run() method.
 *
 */
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
