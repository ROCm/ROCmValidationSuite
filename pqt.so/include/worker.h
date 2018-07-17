
#ifndef _WORKER_H_
#define _WORKER_H_

#include <string>
#include "rvsthreadbase.h"

	
/**
 * @class Worker
 * @ingroup PQT
 *
 * @brief Monitoring implementation class
 *
 * Derives from rvs::ThreadBase and implements actual monitoring functionality
 * in its run() method.
 *
 */

class Worker : public rvs::ThreadBase {

public:
	Worker();
	~Worker();
	
	void stop(void);
  //! Sets initiating action name
	void set_name(const std::string& name) { action_name = name; }
	//! sets stopping action name
	void set_stop_name(const std::string& name) { stop_action_name = name; }
	//! Sets JSON flag
	void json(const bool flag) { bjson = flag; }
	//! Returns initiating action name
	const std::string& get_name(void) { return action_name; }
	
protected:
	virtual void run(void);
	
protected:
  //! TRUE if JSON output is required
	bool		bjson;
  //! Loops while TRUE
	bool 		brun;
  //! Name of the action which initiated monitoring
	std::string	action_name;
  //! Name of the action which stops monitoring
	std::string	stop_action_name;
};
	


#endif // _WORKER_H_