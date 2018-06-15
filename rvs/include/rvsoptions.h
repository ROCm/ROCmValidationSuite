
#ifndef _RVSOPTIONS_H_
#define _RVSOPTIONS_H_

#define DO_PRAGMA(x) _Pragma (#x)
#define TODO(x) DO_PRAGMA(message ("TODO - " #x))

#include <string>
#include <map>

namespace rvs
{

class options
{
public:

  static bool has_option(const std::string& pOption);
  static bool has_option(const std::string& pOption, std::string& val);
  static const std::map<std::string,std::string>& get(void);

protected:
  static      std::map<std::string,std::string>	opt;

friend class cli;
};


}


#endif // _RVSOPTIONS_H_