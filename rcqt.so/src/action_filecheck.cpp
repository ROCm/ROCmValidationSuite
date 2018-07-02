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
   #include "action.h"
   
   #include <iostream>
   #include <stdlib.h>
   #include <map>
   #include <sys/types.h>
   #include <unistd.h>
   #include <pwd.h>
   #include <grp.h>
   #include <string.h>
   #include <vector>
   #include <sys/types.h>
   #include <sys/stat.h>
   #include <unistd.h>
   #include <cmath>
   
   #include "rvs_module.h"
   #include "rvsliblogger.h"
   #include "rvs_util.h"
   #include "rvsloglp.h"
   
   #define BUFFER_SIZE 3000
   
   using namespace std;
   
   
   action::action() {
   }
   
   action::~action() {
     property.clear();
   }
   
   int action::property_set(const char* Key, const char* Val) {
     return rvs::lib::actionbase::property_set(Key, Val);
   }
   
   int action::run()
   {
   
   //filecheck
    string file,owner,group,msg,check;
    int permission,type;
    bool exists,pass;
    string exists_string;
    
    map<string, string>::iterator iter;
    iter = property.find("name");
    std::string action_name = iter->second;
    
    iter = property.find("file");
    file = iter->second;
    
    struct stat info;
    iter = property.find("exists");
    
    if (iter==property.end())
      exists=true;
    else exists_string = iter->second;
    if(exists_string=="false")
      exists=false;
    else if (exists_string=="true")
      exists=true;

        
    unsigned int sec,usec;
    rvs::lp::get_ticks(sec, usec);
    void* r = rvs::lp::LogRecordCreate("RCQT-filecheck", action_name.c_str(), rvs::loginfo, sec, usec);
    if (exists==false){
      if (stat(file.c_str(), &info)<0)
        check="true";
      else check="false";
      string msg = "[" + action_name + "] " + " filecheck "+ file +" DNE " + check;
      rvs::lp::AddString(r, "pass", std::to_string(exists));
      rvs::lp::Log( msg.c_str(),rvs::logresults);
    }
    else{
      if (stat(file.c_str(), &info)<0)
        cerr<<"File is not found"<<endl;
      else{
      iter = property.find("owner");
      if(iter == property.end())
        cerr << "Ownership is not tested." << endl;
      else{
        owner = iter->second;
        struct passwd *pws;
        pws = getpwuid(info.st_uid);
        if (pws->pw_name==owner)
          check="true";
        else check="false";
        cout<<"owner:"<<pws->pw_name<<endl;
        string msg = "[" + action_name + "] " + " filecheck " + owner +" owner:"+check;
        rvs::lp::AddString(r, "pass", owner);
        rvs::lp::Log( msg.c_str(),rvs::logresults);   
      }
      iter = property.find("group");
      if(iter == property.end())
        cerr << "Group ownership is not tested." << endl;
      else{
        group = iter->second;
        struct group *g;
        g = getgrgid(info.st_gid);
        if (g->gr_name==group)
          check="true";
        else check="false";
        cout<<"group:"<<g->gr_name<<endl;
        string msg = "[" + action_name + "] " + " filecheck " + group+ " group:"+check;
        rvs::lp::AddString(r, "pass", group);
        rvs::lp::Log( msg.c_str(),rvs::logresults);  
      }
      iter = property.find("permission");
      if(iter == property.end())
        cerr << "Permissions are not tested." << endl;
      else{
        permission = std::atoi (iter->second.c_str());
//        cout<<info.st_mode%512<<endl;
//        cout<<std::oct<<info.st_mode<<endl;
        if (info.st_mode==permission)
          check="true";
        else check="false";
        string msg = "[" + action_name + "] " + " filecheck " + std::to_string(permission);
        rvs::lp::AddString(r, "pass", std::to_string(permission));
        rvs::lp::Log( msg.c_str(),rvs::logresults); 
      }
      iter = property.find("type");
      if(iter == property.end())
        cerr << "File type is not tested." << endl; 
      else{
        type = std::atoi (iter->second.c_str());
        string msg = "[" + action_name + "] " + " filecheck " + std::to_string(type)+" type:"+check;
        rvs::lp::AddString(r, "pass", std::to_string(type));
        rvs::lp::Log( msg.c_str(),rvs::logresults); 
      }
      
      }
    }