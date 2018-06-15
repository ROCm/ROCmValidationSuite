

#ifndef ACTION_H_
#define ACTION_H_

#include <vector>
#include <string>

#include "rvslib.h"

using std::vector;
using std::string;

class action : public rvs::lib::actionbase
{
public:
	action();
	virtual ~action();
	
	virtual int property_set(const char*, const char*);
	virtual int run(void);
        
private:
    
    vector<string> gpus_id;                  // the list of all gpu_id
                                             // in the <device> property

    string action_name;
    bool bjson;
    void* json_root_node;

    // configuration properties getters
    bool property_get_device(int *error, int num_nodes); // gets the device property value (list of gpu_id)
    //from the module's properties collection
    void property_get_action_name(void);  // gets the action name
    int property_get_deviceid(int *error);  // gets the deviceid
	
protected:
	
};

#endif /* ACTION_H_ */
