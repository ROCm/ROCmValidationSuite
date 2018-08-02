
#ifndef _WORKER_H_
#define _WORKER_H_

#include <string>
#include <mutex>

#include "rvsthreadbase.h"


/**
 * @class Worker
 * @ingroup PQT
 *
 * @brief Bandwidth test implementation class
 *
 * Derives from rvs::ThreadBase and implements actual test functionality
 * in its run() method.
 *
 */

namespace rvs {
class hsa;
}

class Worker : public rvs::ThreadBase {

public:
  Worker();
  virtual ~Worker();

  //! stop thread loop and exit thread
  void stop();
  //! Sets initiating action name
  void set_name(const std::string& name) { action_name = name; }
  //! sets stopping action name
  void set_stop_name(const std::string& name) { stop_action_name = name; }
  //! Sets JSON flag
  void json(const bool flag) { bjson = flag; }
  //! Returns initiating action name
  const std::string& get_name(void) { return action_name; }

  int initialize(int iSrc, int iDst, bool Bidirect);
  int do_transfer();
  void get_running_data(int* Src, int* Dst, bool* Bidirect,
                        size_t* Size, double* Duration);
  void get_final_data(int* Src, int* Dst, bool* Bidirect,
                      size_t* Size, double* Duration);

protected:
  virtual void run(void);

protected:
  //! TRUE if JSON output is required
  bool    bjson;
  //! Loops while TRUE
  bool    brun;
  //! Name of the action which initiated thread
  std::string  action_name;
  //! Name of the action which stops thread
  std::string  stop_action_name;

  rvs::hsa* pHsa;
  int src_node;
  int dst_node;
  bool bidirect;

  //! Current size of transfer data
  size_t current_size;

  // running totals
  size_t running_size;
  double running_duration;

  // final totals
  size_t total_size;
  double total_duration;

  std::mutex cntmutex;
};



#endif // _WORKER_H_