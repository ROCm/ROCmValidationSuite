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
#include "yaml-cpp/yaml.h"

#include "rvsif0.h"
#include "rvsif1.h"
#include "rvsaction.h"
#include "rvsmodule.h"
#include "rvsliblogger.h"
#include "rvsoptions.h"

#define VER "BUILD_VERSION_STRING"

using namespace std;

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
  if( rvs::options::has_option("-h", val)) {
    do_help();
    return 0;
  }

  // check -v options
  if( rvs::options::has_option("-ver", val)) {
    do_version();
    return 0;
  }

  // check -d options
  if( rvs::options::has_option("-d", val)) {
    int level;
    try {
      level = std::stoi(val);
    }
    catch(...) {
      cerr << "ERROR: syntax error: logging level not integer: " << val <<endl;
      return -1;
    }
    if( level < 0 || level > 5) {
      cerr << "ERROR: syntax error: logging level not in range [0..5]: " << val <<endl;
      return -1;
    }
    logger::log_level(level);
  }

  // check -a options
  if( rvs::options::has_option("-a", val)) {
    logger::append(true);
  }

  // check -j options
  if( rvs::options::has_option("-j", val)) {
    logger::to_json(true);
  }


  if( rvs::options::has_option("-l", val)) {
    logger::logfile(val);
  }

  string config_file;
  if( rvs::options::has_option("-c", val)) {
    config_file = val;
  }
  else {
    config_file = "conf/rvs.conf";
    config_file = path + config_file;
  }

  // construct modules configuration file relative path
  val = path + ".rvsmodules.config";
  rvs::module::initialize(val.c_str());

  if( rvs::options::has_option("-t", val)) {
        cout<< endl << "ROCm Validation Suite (version " << LIB_VERSION_STRING << ")" << endl << endl;
        cout<<"Modules available:"<<endl;
    rvs::module::do_list_modules();
    return 0;
  }

  logger::initialize();

  if( rvs::options::has_option("-g")) {
    int sts = do_gpu_list();
    logger::terminate();
    return sts;
  }

  try {
    sts = do_yaml(config_file);
  } catch(exception& e) {
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
  cout << LIB_VERSION_STRING << endl;
}

//! Prints help
void rvs::exec::do_help() {
  cout << "No help available." << endl;
}

//! Reports list of AMD GPUs presnt in the system
int rvs::exec::do_gpu_list() {

  cout << "\nROCm Validation Suite (version " << LIB_VERSION_STRING << ")\n" << endl;
  // create action excutor in .so
  rvs::action* pa = module::action_create("pesm");
  if(!pa) {
    cerr << "ERROR: could not list GPUs" << endl;
    return 1;
  }

  // obtain interface to set parameters and execute action
  if1* pif1 = (if1*)(pa->get_interface(1));
  if(!pif1) {
    cerr << "ERROR: could not obtain interface IF1"<< endl;
    module::action_destroy(pa);
    return 1;
  }

  // specify "list GPUs" action
  pif1->property_set("do_gpu_list", "");

  // set command line options:
  for (auto clit = rvs::options::get().begin(); clit != rvs::options::get().end(); ++clit) {
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
