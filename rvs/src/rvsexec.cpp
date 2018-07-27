/********************************************************************************
 *
 * Copyright (c) 2018 ROCm Developer Tools
 *
 * MIT LICENSE:
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to do
 * so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include "rvsexec.h"

#include <iostream>
#include <memory>
#include <string>
#include <fstream>
#include "yaml-cpp/yaml.h"

#include "rvsif0.h"
#include "rvsif1.h"
#include "rvsaction.h"
#include "rvsmodule.h"
#include "rvsliblogger.h"
#include "rvsoptions.h"

#define VER "BUILD_VERSION_STRING"

using std::string;
using std::cout;
using std::cerr;
using std::endl;

//! Default constructor
rvs::exec::exec() {
}

//! Default destructor
rvs::exec::~exec() {
}


/**
 * @brief Main executor method.
 *
 * @return 0 if successful, non-zero otherwise
 *
 */
int rvs::exec::run() {
  int     sts = 0;
  string  val;
  string  path;

  options::has_option("pwd", path);

  // check -h options
  if (rvs::options::has_option("-h", val)) {
    do_help();
    return 0;
  }

  // check -v options
  if (rvs::options::has_option("-ver", val)) {
    do_version();
    return 0;
  }

  // check -d options
  if (rvs::options::has_option("-d", val)) {
    int level;
    try {
      level = std::stoi(val);
    }
    catch(...) {
      cerr << "ERROR: syntax error: logging level not integer: " << val <<endl;
      return -1;
    }
    if (level < 0 || level > 5) {
      cerr << "ERROR: syntax error: logging level not in range [0..5]: "
      << val <<endl;
      return -1;
    }
    logger::log_level(level);
  }

  // if verbose is set, set logging level to the max value (i.e. 5)
  if (rvs::options::has_option("-v")) {
    logger::log_level(5);
  }

  // check -a options
  if (rvs::options::has_option("-a", val)) {
    logger::append(true);
  }

  // check -j options
  if (rvs::options::has_option("-j", val)) {
    logger::to_json(true);
  }


  if (rvs::options::has_option("-l", val)) {
    logger::logfile(val);
  }

  string config_file;
  if (rvs::options::has_option("-c", val)) {
    config_file = val;
  } else {
    config_file = "conf/rvs.conf";
    config_file = path + config_file;
  }

  // Check if pConfig file exists
  std::ifstream file(config_file);

  if (!file.good()) {
    cerr << "ERROR: " << config_file << " file is missing.\n";
    return -1;
  } else {
    file.close();
  }

  // construct modules configuration file relative path
  val = path + ".rvsmodules.config";
  rvs::module::initialize(val.c_str());

  if (rvs::options::has_option("-t", val)) {
        cout << endl << "ROCm Validation Suite (version " <<
        LIB_VERSION_STRING << ")" << endl << endl;
        cout << "Modules available:" << endl;
    rvs::module::do_list_modules();
    return 0;
  }

  logger::initialize();

  if (rvs::options::has_option("-g")) {
    int sts = do_gpu_list();
    logger::terminate();
    rvs::module::terminate();
    return sts;
  }

  try {
    sts = do_yaml(config_file);
  } catch(std::exception& e) {
    sts = -999;
    cerr << "Error processing configuration file " << config_file << endl;
    cerr << "Exception: " << e.what() << endl;
  }

  logger::terminate();

  rvs::module::terminate();

  return sts;
}

//! Reports version strin
void rvs::exec::do_version() {
  cout << LIB_VERSION_STRING << '\n';
}

//! Prints help
void rvs::exec::do_help() {
  cout << "\nUsage: rvs [options]\n";
  cout << "\nOptions:\n\n";
  cout << "-a --appendLog     When generating a debug logfile, do not overwrite the contents\n";
  cout << "                   of a current log. Used in conjuction with the -d and -l options.\n";
  cout << "-c --config        Specify the configuration file to be used.\n";
  cout << "                   The default is <install base>/conf/RVS.conf\n";
  cout << "   --configless    Run RVS in a configless mode. Executes a \"long\" test on all\n";
  cout << "                   supported GPUs.\n";
  cout << "-d --debugLevel    Specify the debug level for the output log. The range is\n";
  cout << "                   0 to 5 with 5 being the most verbose.\n";
  cout << "                   Used in conjunction with the -l flag.\n";
  cout << "-g --listGpus      List the GPUs available and exit. This will only list GPUs\n";
  cout << "                   that are supported by RVS.\n";
  cout << "-i --indexes       Comma separated list of indexes devices to run RVS on. This will\n";
  cout << "                   override the device values specified in the configuration file for\n";
  cout << "                   every action in the configuration file, including the ‘all’ value.\n";
  cout << "-j --json          Output should use the JSON format.\n";
  cout << "-l --debugLogFile  Specify the logfile for debug information. This will produce a log\n";
  cout << "                   file intended for post-run analysis after an error.\n";
  cout << "   --quiet         No console output given. See logs and return code for errors.\n";
  cout << "-m --modulepath    Specify a custom path for the RVS modules.\n";
  cout << "   --specifiedtest Run a specific test in a configless mode. Multiple word tests\n";
  cout << "                   should be in quotes. This action will default to all devices,\n";
  cout << "                   unless the indexes option is specifie.\n";
  cout << "-t --listTests     List the modules available to be executed through RVS and exit.\n";
  cout << "                   This will list only the readily loadable modules\n";
  cout << "                   given the current path and library conditions.\n";
  cout << "-v --verbose       Enable verbose reporting. This is equivalent to\n";
  cout << "                   specifying the -d 5 option.\n";
  cout << "   --version       Displays the version information and exits.\n";
  cout << "-h --help          Display usage information and exit.\n";
}

//! Reports list of AMD GPUs presnt in the system
int rvs::exec::do_gpu_list() {
  cout << "\nROCm Validation Suite (version " << LIB_VERSION_STRING << ")\n\n";

  // create action excutor in .so
  rvs::action* pa = module::action_create("pesm");
  if (!pa) {
    cerr << "ERROR: could not list GPUs\n";
    return 1;
  }

  // obtain interface to set parameters and execute action
  if1* pif1 = static_cast<if1*>(pa->get_interface(1));
  if (!pif1) {
    cerr << "ERROR: could not obtain interface IF1\n";
    module::action_destroy(pa);
    return 1;
  }

  pif1->property_set("name", "(launcher)");

  // specify "list GPUs" action
  pif1->property_set("do_gpu_list", "");

  // set command line options:
  for (auto clit = rvs::options::get().begin();
       clit != rvs::options::get().end(); ++clit) {
    string p(clit->first);
    p = "cli." + p;
    pif1->property_set(p, clit->second);
  }

  // execute action
  int sts = pif1->run();

  // procssing finished, release action object
  module::action_destroy(pa);

  return sts;
}
