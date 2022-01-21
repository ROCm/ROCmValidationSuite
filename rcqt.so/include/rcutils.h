#ifndef UTILS_H
#define UTILS_H
#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <chrono>

const std::string os_release_file {"/etc/os-release"};
const std::string name_key {"NAME"};
const std::vector<std::string> op_systems{ "ubuntu", "centos", "sles"};

struct package_info{
	std::string name{};
	std::string version{};
	
};

// common funtions
std::string get_last_word(const std::string& input);
std::string getOS();
std::string remSpaces(std::string str);
std::string pfilename(const std::string& package);
/*
*/
#endif
