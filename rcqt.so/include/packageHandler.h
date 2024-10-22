/********************************************************************************
 *
 * Copyright (c) 2018-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef PACKAGE_HANDLER_H
#define PACKAGE_HANDLER_H

#include <string>
#include <map>
#include <memory>
#include "include/rcutils.h"
#include "metaPackageInfo.h"
#include "include/rvsactionbase.h"

class PackageHandler{
public:
  PackageHandler() = default;
  void setManifest(std::string manifest){
    m_manifest = manifest;
  }
  std::string getManifest() const{
    return m_manifest;
  }
  virtual bool parseManifest(); // and load m_pkgversionmap
  void  validatePackages();
  const std::map<std::string, std::string>& getPackageMap() const{
    return m_pkgversionmap;
  }
  virtual std::string getInstalledVersion(const std::string& package) = 0;
  virtual bool pkgrOutputParser(const std::string& s_data, package_info& info) = 0;
  void setPkg(const std::string& pkg) {
  }
  
  void setPackageList(const std::vector<std::string>& pkglist) {
    m_pkglist = pkglist;
  }
  const std::vector<std::string>& getPackageList() const {
    return m_pkglist;
  }
  void listPackageVersion();
  virtual ~PackageHandler(){}
	
  //! Set action callback 
  void setCallback(rvs::callback_t _callback, void * _user_param) {
    callback = _callback;
    user_param = _user_param;
  }
  void setAction(std::string _action){
    action_name = _action;
  }

  std::string getAction(){
    return action_name;
  }

  void setModule(std::string _module){
    module_name = _module;
  }

  std::string getModule(){
    return module_name;
  }
  void log_to_json(int log_level, std::vector<std::string> kvlist);
private:
  std::map<std::string, std::string> m_pkgversionmap;
  std::vector<std:: string> m_pkglist;
  std::string action_name;
  std::string module_name;
protected:
  std::unique_ptr<PackageInfo> metaInfo;
  std::string m_manifest;	
  // callback
  rvs::callback_t callback;
  // User parameter
  void * user_param;
};
#endif
