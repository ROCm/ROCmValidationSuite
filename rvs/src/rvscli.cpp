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

#include "include/rvscli.h"

#include <sys/types.h>
#include <unistd.h>
#include <sys/stat.h>
#include <limits.h>
#include <string.h>
#include <stdio.h>

#include <iostream>
#include <cstddef>         // std::size_t
#include <memory>
#include <string>
#include <stack>

#include "include/rvsoptions.h"
#include "include/rvsliblogger.h"

#define MODULE_NAME_CAPS "CLI"

/**
 * @brief Constructor
 *
 * @param ptruename option name (some command line options may have multiple aliases)
 * @param s1 possible continuation
 * @param s2 possible continuation
 * @param s3 possible continuation
 *
 */
rvs::cli::optbase::optbase(const char* ptruename, econtext s1, econtext s2,
                           econtext s3) {
  name = ptruename;
  new_context.push(eof);
  new_context.push(s1);
  if (s2 != eof) new_context.push(s2);
  if (s3 != eof) new_context.push(s3);
}

//! Default destructor
rvs::cli::optbase::~optbase() {
}

/**
 * @brief Replaces current context with continuations allowed for this token
 *
 * @param pold_context ptr to old std::stack context
 * @return always returns TRUE
 *
 */
bool rvs::cli::optbase::adjust_context(std::stack<econtext>* pold_context) {
  while (!(*pold_context).empty())
    (*pold_context).pop();
  (*pold_context) = new_context;

  return true;
}

//! Default constructor
rvs::cli::cli() {
  itoken = 1;
}

//! Default destructor
rvs::cli::~cli() {
}

/**
 * @brief Extracts path from which rvs was invoked
 *
 * Extracts path from which rvs was invoked and puts it into
 * command line option "pwd"
 *
 */
void rvs::cli::extract_path() {
  char path[PATH_MAX];
  char dest[PATH_MAX];
  memset(dest, 0, sizeof(dest));  // readlink does not null terminate!

  pid_t pid = getpid();
  snprintf(path, sizeof(path), "/proc/%d/exe", pid);
  if (readlink(path, dest, PATH_MAX) == -1) {
    rvs::logger::Err("could not extract path to executable", MODULE_NAME_CAPS);
    return;
  }

  std::string argv0(dest);

  size_t found = argv0.find_last_of("/\\");
  options::opt["pwd"] = argv0.substr(0, found) + "/";
}


/**
 * @brief Defines grammar
 *
 * Defines possible command line options for this application
 *
 */
void rvs::cli::init_grammar() {
  std::shared_ptr<optbase> sp;

  while (!context.empty()) context.pop();
  grammar.clear();
  itoken = 1;
  errstr.clear();
  current_option.clear();
  current_value.clear();

  sp = std::make_shared<optbase>("-a", command);
  grammar.insert(gpair("-a", sp));
  grammar.insert(gpair("--appendLog", sp));

  sp = std::make_shared<optbase>("-c", command, value);
  grammar.insert(gpair("-c", sp));
  grammar.insert(gpair("--config", sp));

//   sp = std::make_shared<optbase>("--configless", command);
//   grammar.insert(gpair("--configless", sp));

  sp = std::make_shared<optbase>("-d", command, value);
  grammar.insert(gpair("-d", sp));
  grammar.insert(gpair("--debugLevel", sp));

  sp = std::make_shared<optbase>("-g", command);
  grammar.insert(gpair("-g", sp));
  grammar.insert(gpair("--listGpus", sp));

  sp = std::make_shared<optbase>("-i", command, value);
  grammar.insert(gpair("-i", sp));
  grammar.insert(gpair("--indexes", sp));

  sp = std::make_shared<optbase>("-j", command);
  grammar.insert(gpair("-j", sp));
  grammar.insert(gpair("--json", sp));

  sp = std::make_shared<optbase>("-l", command, value);
  grammar.insert(gpair("-l", sp));
  grammar.insert(gpair("--debugLogFile", sp));

  sp = std::make_shared<optbase>("-q", command);
  grammar.insert(gpair("-q", sp));
  grammar.insert(gpair("--quiet", sp));

  sp = std::make_shared<optbase>("-m", command, value);
  grammar.insert(gpair("-m", sp));
  grammar.insert(gpair("--modulepath", sp));

  //  sp = std::make_shared<optbase>("-s", command);
  //  grammar.insert(gpair("-s", sp));
  //  grammar.insert(gpair("--scriptable", sp));
  //
  //  sp = std::make_shared<optbase>("-st", value);
  //  grammar.insert(gpair("--specifiedtest", sp));
  //
  //  sp = std::make_shared<optbase>("-sf", command);
  //  grammar.insert(gpair("--statsonfail", sp));

  sp = std::make_shared<optbase>("-t", command);
  grammar.insert(gpair("-t", sp));
  grammar.insert(gpair("--listTests", sp));

  sp = std::make_shared<optbase>("-v", command);
  grammar.insert(gpair("-v", sp));
  grammar.insert(gpair("--verbose", sp));

  sp = std::make_shared<optbase>("-ver", command);
  grammar.insert(gpair("--version", sp));

  sp = std::make_shared<optbase>("-h", command);
  grammar.insert(gpair("-h", sp));
  grammar.insert(gpair("--help", sp));
}

/**
 * @brief Parse command line
 *
 * Parses command line and stores command line options and their vaules into rvs::options
 *
 * @param Argc standard C argc parameter to main()
 * @param Argv standard C argv parameter to main()
 * @return 0 - OK, non-zero if error
 *
 */
int rvs::cli::parse(int Argc, char** Argv) {
  init_grammar();

  extract_path();

  argc = Argc;
  argv = Argv;
  context.push(econtext::eof);
  context.push(econtext::command);

  for (;;) {
    std::string token = get_token();
    bool token_done = false;
    while (!token_done) {
      econtext top = context.top();
      context.pop();

      switch (top) {
      case econtext::command:
        token_done = try_command(token);
        break;

      case econtext::value:
        token_done = try_value(token);
        if (!token_done) {
          errstr = std::string("syntax error: value expected after ") +
                   current_option;
          return -1;
        }
        break;

      case econtext::eof:
        if (token == "") {
          emit_option();
          return 0;
        } else {
          errstr = "unexpected command line argument: " +
                    token;
          return -1;
        }

      default:
          errstr = "syntax error: " + token;
          return -1;
      }
    }
  }

  return -1;
}

/**
 * @brief Returns error string
 *
 * @return error string as const char*
 *
 */
const char* rvs::cli::get_error_string() {
  return errstr.c_str();
}


/**
 * @brief Returns next token from input stream
 *
 * @return token as const char*
 *
 */
const char* rvs::cli::get_token() {
  if (itoken >= argc)
    return (const char*)"";

  return argv[itoken++];
}


/**
 * @brief Sends accepted command line option, with parameter if any, into rvs::options
 *
 * @return true
 *
 */
bool rvs::cli::emit_option() {
  // emit previous option and its value (if andy)
  if (current_option != "")  {
    options::opt[current_option] = current_value;
  }

  // reset working buffer
  current_option = "";
  current_value  = "";

  return true;
}


/**
 * @brief Try interpreting given token as command line option.
 *
 * If successful, emmits previous option and stores current one in a buffer.
 * This is needed in case optionhas parameter to it.
 *
 * @param token token being processed
 * @return true if successful, false otherwise
 *
 */
bool rvs::cli::try_command(const std::string& token) {
  auto it = grammar.find(token);
  if (it == grammar.end())
    return false;

  // emit previous buffer contents (if any)
  emit_option();

  // token identified as command, so store it:
  current_option = it->second->name;

  // fill context  std::stack with new possible continuations:
  it->second->adjust_context(&context);

  return true;
}


/**
 * @brief Try interpreting given token as a vaule following previous command line option.
 *
 * If successful, stores current token in a buffer as value
 *
 * @param token token being processed
 * @return true if successful, false otherwise
 *
 */
bool rvs::cli::try_value(const std::string& token) {
  if (token == "")
    return false;

  //  should not be one of command line options
  auto it = grammar.find(token);
  if (it != grammar.end())
    return false;

  // token is value for previous command
  current_value = token;

  // emit previous option-value pair:
  emit_option();

  return true;
}

