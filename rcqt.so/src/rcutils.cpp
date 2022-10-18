/********************************************************************************
 *
 * Copyright (c) 2018-2022 ROCm Developer Tools
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

#include "include/rcutils.h"

OSType searchos(std::string os_name){
	std::string lowcasename{os_name};
	std::transform( lowcasename.begin(), lowcasename.end(), lowcasename.begin(),
				[](unsigned char c){ return std::tolower(c); });
	for( auto itr = op_systems.begin();itr !=op_systems.end();++itr){
		if(lowcasename.find(itr->first) != std::string::npos){
			return itr->second;		
		}
	}
	return OSType::None;	
}

OSType getOS(){
  std::ifstream rel_file(os_release_file.c_str());
  if(!rel_file.good()){
    std::cout << "No /etc/os-release file, cant fetch details " << std::endl;
    return OSType::None;
  }
  std::string line;
  while (std::getline(rel_file, line))
  {
		auto found = line.find(name_key) ;
    if (found!=std::string::npos){
      found = line.find('\"');
      auto endquote = line.find_last_of('\"');
      if(found == std::string::npos || endquote == std::string::npos)
        return OSType::None;
      std::string osame = line.substr(found+1, endquote-found-1 );
			return searchos(osame);
    }
  }
}

std::string get_last_word(const std::string& input){
  std::stringstream is{input};
  std::string temp, last;
  while (std::getline(is, temp, ' ')) {
    last=temp;
  }
  return last;
}

std::string pfilename(const std::string& package){
	std::string fname{"/tmp/"};
  fname += package;
	using namespace std::chrono;
	auto now = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
	std::ostringstream oss;
	oss << now;
	fname += {"."} ;
	fname += oss.str();
	fname += {".txt"};
	return fname;
}

std::string remSpaces(std::string str){
  str.erase(std::remove(str.begin(), str.end(), ' '), str.end());
  return str;
}

bool getPackageInfo(const std::string& package,
    const std::string& packagemgr,
    const std::string& command,
    const std::string& option,
    std::stringstream &ss){

  int read_pipe[2]; // From child to parent
  if(pipe(read_pipe) == -1){
    perror("Pipe");
    return false;
  }

  pid_t process_id = fork();
  if(process_id < 0){
    perror("Fork");
    return false;
  }
  else if(process_id == 0) {

    /* Child process */
    dup2(read_pipe[1], 1);
    close(read_pipe[0]);
    close(read_pipe[1]);

    /* Apply options in command only if present */
    if(option.empty()) {
      execlp(packagemgr.c_str(), packagemgr.c_str(), command.c_str(), package.c_str(), NULL);
    }
    else {
      execlp(packagemgr.c_str(), packagemgr.c_str(), command.c_str(), option.c_str(), package.c_str(), NULL);
    }
  }
  else {
    /* Parent process */
    int status;
    waitpid(process_id, &status,0);
    close(read_pipe[1]);
    {
      char arr[8192];
      int n = read(read_pipe[0], arr, sizeof(arr));
      ss.write(arr, n);

    }
    close(read_pipe[0]);
    return true;
  }
}

