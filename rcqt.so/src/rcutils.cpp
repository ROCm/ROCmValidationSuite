#include "include/rcutils.h"


std::string searchos(std::string os_name){
	std::string lowcasename{os_name};
	std::transform( lowcasename.begin(), lowcasename.end(), lowcasename.begin(),
				[](unsigned char c){ return std::tolower(c); });
	for( const auto& name: op_systems){
		if(lowcasename.find(name) != std::string::npos)
			return name;		
	}
  	
}
std::string getOS(){
  std::ifstream rel_file(os_release_file.c_str());
  if(!rel_file.good()){
    std::cout << "No /etc/os-release file, cant fetch details " << std::endl;
    return std::string{};
  }
  std::string line;
  while (std::getline(rel_file, line))
  {
		auto found = line.find(name_key) ;
    if (found!=std::string::npos){
      found = line.find('\"');
      auto endquote = line.find_last_of('\"');
      if(found == std::string::npos || endquote == std::string::npos)
        return std::string{};
      std::string osame = line.substr(found+1, endquote-found-1 );
      std::cout << "OS installed is : " << osame << std::endl;
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


/*
bool parser(std::string s_data, package_info& info){
	std::stringstream data{s_data};
	// first line tells if we need to proceed or not.
	std::string line;
	bool found = false;
	while(std::getline(data, line)){
		if(line.find("Version") != std::string::npos){
			info.version = get_last_word(line);
			found = true;
		} else if( line.find("Package") != std::string::npos){
			info.name = get_last_word(line);
		}		
	}
	return found;
}


void find_version(std::string package){
	int read_pipe[2]; // From child to parent
	int exit_status;
	if(pipe(read_pipe) == -1){
		perror("Pipe");
		return;
	}
	pid_t process_id = fork();
	if(process_id < 0){
		perror("Fork");
		return;
	}else if(process_id == 0) {
		dup2(read_pipe[1], 1);
		close(read_pipe[0]);
		close(read_pipe[1]);
		execlp("dpkg", "dpkg", "--status", package.c_str(), NULL);
	} else {
		// parent:
		int status;
		waitpid(process_id, &status,0);
		std::stringstream ss;
		close(read_pipe[1]);
		//dup(read_pipe[0], 0);
		{
			char arr[4096];
			int n = read(read_pipe[0], arr, sizeof(arr));
			ss.write(arr, n);

		}
		std::cout << ss.str() << std::endl;
		close(read_pipe[0]);
		std::string ver_string{};
		package_info pinfo; 
		auto res = parser(ss.str(), pinfo);
		std::cout << pinfo.name << " and " << pinfo.version << std::endl;
		return;
	}
}

int main(){
  //std::cout << getOS() << std::endl;
	print_version({"rocminfo"});
  return 0;

}
*/
