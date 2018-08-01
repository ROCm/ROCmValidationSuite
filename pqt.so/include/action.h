

#ifndef ACTION_H_
#define ACTION_H_

#include <stdlib.h>
#include <assert.h>
#include <algorithm>
#include <unistd.h>
#include <cctype>
#include <sstream>
#include <limits>

#include "rvsactionbase.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

using std::string;
using std::vector;

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
  // PQT specific config keys
  void property_get_ramp_interval(int *error);
  void property_get_log_interval(int *error);
  int create_threads();
  int run_single();
  int run_parallel();

  void ontimer();

private:

};

#endif /* ACTION_H_ */
