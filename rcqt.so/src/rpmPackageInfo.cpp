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

#include <iostream>
#include <sstream>
#include <fstream>
#include "include/rpmPackageInfo.h"

bool RpmPackageInfo::readMetaPackageInfo(std::string ss){

  std::stringstream inss{ss};
  std::string line;
  bool found = false;
  size_t pos = std::string::npos;
  std::ofstream os;

  os.open(getFileName() , std::ofstream::out | std::ofstream::app);

  while(std::getline(inss, line)){

    pos = line.find("=");

    if(std::string::npos != pos){

      /* Get depend package and its version */
      auto firstSpacePos = line.find(" ");

      auto depPackageName = line.substr(0, firstSpacePos);
      auto depPackageVersion = line.substr(pos + 2);

      /* Write depend package name and its version to file */
      os << depPackageName << " " << depPackageVersion << std::endl;

      pos = std::string::npos;

      if (false == found){
        found = true;
      }
    }
  }

  os.close();

  return found;
}
