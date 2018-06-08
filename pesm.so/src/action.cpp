
#include "action.h"

#include "rvs_module.h"
#include "worker.h"



using namespace std;

extern const char* pcie_cap_names[];

static Worker* pworker;

action::action()
{
}

action::~action()
{
	property.clear();
}

int action::run(void)
{
	log("[PESM] in run()", rvs::logdebug);
	
	// handle "wait" property
	if (property["wait"] != "") {
		sleep(atoi(property["wait"].c_str()));
	}
	
	if(property["monitor"] == "true") {
		log("property[\"monitor\"] == \"true\"", rvs::logdebug);
		
		if (!pworker) {
		log("creating Worker", rvs::logdebug);
			pworker = new Worker();
			pworker->set_name(property["name"]);
		}
		log("starting Worker", rvs::logdebug);
		pworker->start();
// 		log("detaching Worker", rvs::logdebug);
// 		pworker->detach();
		
		log("[PESM] Monitoring started", rvs::logdebug);
	}
	else {
		log("property[\"monitor\"] != \"true\"", rvs::logdebug);
		if (pworker) {
			pworker->stop();
			delete pworker;
			pworker = nullptr;
		}
		log("[PESM] Monitoring stopped", rvs::logdebug);
	}
	return 0;
}