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
#ifndef RVSACTIONBASE_H_
#define RVSACTIONBASE_H_

#include <map>
#include <string>


namespace rvs
{

/**
 *  @class actionbase
 *
 *  @brief Base class for all module level actions
 *
 */

class actionbase
{
public:
  virtual ~actionbase();

protected:
  actionbase();
  void sleep(const unsigned int ms);

public:
  virtual int     property_set(const char*, const char*);

  //! Virtual action function. To be implemented in every derived class.
  virtual int     run(void) = 0;
  bool has_property(const std::string& key, std::string& val);
  bool has_property(const std::string& key);

protected:
/**
 *  @brief Collection of properties
 *
 * Properties represent:
 *  - content of corresponding "action" tag in YAML .conf file
 *  - command line arguments given when invoking rvs
 *  - other parameters given for specific module actions (see module action for help)
 */
  std::map<std::string, std::string>  property;
};


}  // namespace rvs


#endif // RVSACTIONBASE_H_