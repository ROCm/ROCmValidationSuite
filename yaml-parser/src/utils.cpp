#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <assert.h>
#include <utils.h>
std::string getModuleName(std::string input){
        std::stringstream is{input};
        std::string temp, last;
        while (std::getline(is, temp, ' ')) {
                last=temp;
  }
        return last;
}
std::string getModule(std::string filename){
  static int actions = 0;
        std::string module_name{};
        std::ifstream f{filename};
        assert (f.good());
        std::string line;
        while(std::getline(f, line)){
            if(line.empty())
                continue;
            line = line.substr(line.find_first_not_of(" "));
            if(line[0] == '#')
                    continue;
            if(line.find("name:") != std::string::npos){
                ++actions;
                if(actions > 1){
                    // should have found module by now, invalid conf.
                    std::cout <<" Invalid conf file, missing module name " << std::endl;
                        return module_name;
                } else{
                    continue;
                }
            }
            if (line.find("module:") != std::string::npos){
                  return getModuleName(line);
            } else{
                  continue;
            }
        }
        return module_name;
}
