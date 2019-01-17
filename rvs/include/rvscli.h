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
#ifndef RVS_INCLUDE_RVSCLI_H_
#define RVS_INCLUDE_RVSCLI_H_

#include <memory>
#include <map>
#include <stack>
#include <string>
#include <utility>

namespace rvs {

/**
 * @class cli
 * @ingroup Launcher
 *
 * @brief Command line interpretter class.
 *
 * Parses command line options given when invoking rvs utility.
 * Output si stored into rvs::options class.
 *
 */

class cli {
 public:
  //! Default constructor
  cli();
  //! Default destructor
  virtual ~cli();

  int parse(int Argc, char** Argv);
  const char* get_error_string();

 protected:
/**
 *
 * @brief Possible continuation types
 *
 * Defines possible type for the next token:
 * |value    |meaning
 * |---------|-------------------------
 * |eof      |end of input
 * |value    |value expected (must be different then command strings defined throuth grammar)
 * |command  |command string as defined in grammar
 *
 */
  typedef enum {eof, value, command} econtext;

  /**
 * @class optbase
 *
 * @brief Token continuation list
 *
 * Stores possible contions for a token defined in grammar
 *
 */

  class optbase {
   public:
    optbase(const char* ptruename,
            econtext s1, econtext s2 = eof, econtext s3 = eof);
    //! Default destructor
    virtual ~optbase();
    virtual bool adjust_context(std::stack<econtext>* old_context);

   public:
    //! Option name as known internally to RVS
    std::string name;
    //! Continuation stack for this option
    std::stack<econtext> new_context;
  };

  //! Token-continuation pair
  typedef std::pair<std::string, std::shared_ptr<optbase>>   gpair;

 protected:
  const char*  get_token();
  bool  try_command(const std::string& token);
  bool  try_value(const std::string& token);
  bool  emit_option(void);
  void  init_grammar(void);
  void  extract_path(void);

 protected:
  //! Helper variable to store command line parameters across function calls
  int    argc;
  //! Helper variable to store command line parameters across function calls
  char** argv;
  //! Helper variable to store current command line parameter being processed
  int    itoken;
  //! Helper variable to store error string
  std::string errstr;
  //! Helper variable to store token identified as command line option
  std::string current_option;
  //! Helper variable to store token identified as command line option value
  std::string current_value;
  //! Helper variable to store current continuation stack
  std::stack<econtext>  context;
  //! Helper variable to store grammar
  std::map<std::string, std::shared_ptr<optbase>> grammar;
};

}  // namespace rvs

#endif  // RVS_INCLUDE_RVSCLI_H_
