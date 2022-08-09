#ifndef UTILS_H
#define UTILS_H
#include <vector>
#include <map>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <chrono>
enum class OSType{
	Ubuntu,
	Centos,
	SLES,
	None
};
const std::string os_release_file {"/etc/os-release"};
const std::string name_key {"NAME"};
//const std::vector<std::string> op_systems{ "ubuntu", "centos", "sles"};
const std::map<std::string, OSType> op_systems{{"ubuntu",OSType::Ubuntu}, {"centos",OSType::Centos},
					{"sles",OSType::SLES}};
struct package_info{
	std::string name{};
	std::string version{};
	
};

// common funtions
std::string get_last_word(const std::string& input);
OSType getOS();
std::string remSpaces(std::string str);
std::string pfilename(const std::string& package);
/*
*/

bool getPackageInfo(const std::string& package,
    const std::string& packagemgr,
    const std::string& command,
    const std::string& option,
    std::stringstream &ss);

#endif
