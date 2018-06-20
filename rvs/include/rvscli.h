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
#include <memory>
#include <map>
#include <stack>
#include <string>


namespace rvs
{

class cli {

public:

  cli();
  virtual ~cli();

  int parse(int Argc, char** Argv);
  const char* get_error_string();

protected:
  typedef enum {eof, value, command} econtext;

  class optbase {

  public:
    optbase(const char* ptruename, econtext s1, econtext s2 = eof, econtext s3 = eof);
    virtual ~optbase();
    virtual bool adjust_context(std::stack<econtext>& old_context);

  public:
    std::string name;
    std::stack<econtext> new_context;
  };

  typedef std::pair<std::string, std::shared_ptr<optbase>>   gpair;

protected:
  const char*  get_token();
  bool  is_command(const std::string& token);
  bool  try_command(const std::string& token);
  bool  try_value(const std::string& token);
  bool  emit_option(void);
  void  store_command(const std::string& token);
  void  store_value(const std::string& token);
  void  init_grammar(void);
  void  extract_path(const char* Argv0);

protected:
  int    argc;
  char** argv;
  int    itoken;
  std::string errstr;
  std::string current_option;
  std::string current_value;
  std::stack<econtext>  context;
  std::map<std::string,std::shared_ptr<optbase>> grammar;

};

}  // namespace rvs
