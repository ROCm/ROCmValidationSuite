#include <iostream>
#include <stdlib.h>
#include <string>
#include <map>
#include <sys/types.h>
#include <unistd.h>
#include <vector>
#include <pwd.h>
#include <grp.h>

#include "action.h"
#include "rvs_module.h"
#include "rvsliblogger.h"
#include <string.h>

#define BUFFER_SIZE 3000

using namespace std;


action::action()
{
}

action::~action()
{
    property.clear();
}

int action::property_set(const char* Key, const char* Val)
{
	return rvs::lib::actionbase::property_set(Key, Val);
}

int action::run()
{
    bool version_exists = false;
    int fd[2];

    map<string, string>::iterator iter;
    iter = property.find("package");
    if(iter == property.end())
        cerr << "no package field in config file" << endl;
    string package_name = iter->second;
    string version_name;
    iter = property.find("version");
    if(iter != property.end()){
       version_name = iter->second;
       version_exists = true;
    }
    
    pid_t pid;
    pipe(fd);

    pid = fork();
    if(pid == 0){
        // Child process
        
        dup2(fd[1], STDOUT_FILENO);
        dup2(fd[1], STDERR_FILENO);
        char buffer[256];
        snprintf(buffer, 255, "dpkg-query -W -f='${Status} ${Version}\n' %s", package_name.c_str());

        system(buffer);
        
    } else if (pid>0){
        // Parent
        
        char result[BUFFER_SIZE];
        int count;
        close(fd[1]);
           
        count=read(fd[0], result, BUFFER_SIZE);
        
        result[count]=0;
        string result1 = result;
            
        string version_value = result1.substr(21, count - 21);
        int ending = version_value.find_first_of('\n');
        if(ending > 0){
            version_value = version_value.substr(0, ending);
        }
        string passed = "packagecheck " + package_name + " TRUE";
        string failed = "packagecheck " + package_name + " FALSE";
            
        if(strstr(result, "dpkg-query:") == result)
            log(failed.c_str(), rvs::logresults);
        else if(version_exists == false){
            log(passed.c_str(), rvs::logresults);
        }else if(version_name.compare(version_value) == 0){
            log(passed.c_str(), rvs::logresults);
        }else{
            log(failed.c_str(), rvs::logresults);
        }
        
    }else   {
      // fork process error
      cerr << "Internal Error" << endl;
        return -1;
    }
    
    /*bool group_exists = false;
    iter = property.find("user");
    if(iter == property.end())
        cerr << "no user field in config file" << endl;
    string user_name = iter->second;
    string group_name;
    iter = property.find("group");
    if(iter != property.end()){
       //cout << "no version field in config file" << endl;
       group_name = iter->second;
       //cout << version_name << "size " << version_name.length() <<endl;
       group_exists= true;
    }
       
    
    struct passwd *p;
    struct group *g;
    string user_exists = "[rcqt] usercheck " + user_name + " user exists";
    string user_not_exists = "[rcqt] usercheck " + user_name + " user not exists";
    if((p = getpwnam(user_name.c_str())) == nullptr){
        log(user_not_exists.c_str(), rvs::logresults);
        return -1;
    }else{
        log( user_exists.c_str(), rvs::logresults);
    }
    if(group_exists == true){
        if((g = getgrnam(group_name.c_str())) == nullptr){
            cerr << "group doesn't exist" << endl;
            return -1;
        }
    }   
    
    //string user_is_in_group = "[rcqt] 
    
    int i;
    int j=0;
    if(group_exists){
        for(i=0; g->gr_mem[i]!=NULL; i++){
            if(strcmp(g->gr_mem[i], group_name.c_str()) == 0){
                //log("[rcqt] usercheck
                printf("user is in group\n");
                j=1;
                break;
            }
        }
    }
        
    if(j==0) printf("user is not in the group\n");
    */
    
    return 0;
}