#ifndef PARSER_H
#define PARSER_H

class Parser{
public:
  Parse(const std::string& config_file);
  void parse() 
  virtual ~Parser() = default;
};
#endif
