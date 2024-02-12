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

#include <iostream>
#include <sstream>
#include <fstream>
#include "include/zypPackageInfo.h"

bool ZypPackageInfo::readMetaPackageInfo(std::string ss){

  std::stringstream inss{ss};
  std::string line;
  bool found = false;
  size_t pos = std::string::npos;
  std::ofstream os;
  std::string greaterVers{};
  os.open(getFileName() , std::ofstream::out | std::ofstream::app);

  while(std::getline(inss, line)){

    pos = line.find("Requires");

    if(std::string::npos != pos){

      while(std::getline(inss, line)){

        pos = line.find("=");

        if(std::string::npos != pos){
        size_t gpos = line.find(">=");
	if(std::string::npos != gpos && gpos == pos-1)
		greaterVers = "+"; 
          /* Get dependent package and its version */

          size_t firstCharPos = line.find_first_not_of(" ");
          size_t midSpacePos = line.find(" ", firstCharPos);

          auto depPackageName = line.substr(firstCharPos, midSpacePos - firstCharPos);

          midSpacePos = line.find_last_of(" ");
          auto depPackageVersion = line.substr(midSpacePos + 1);
	  auto p = depPackageVersion.find("-");
	  if (std::string::npos != p)
          	depPackageVersion = depPackageVersion.substr(0, p);
          /* Write dependent package name and its version to file */
	  depPackageVersion+=greaterVers;
	  greaterVers.erase();
          os << depPackageName << " " << depPackageVersion << std::endl;

          pos = std::string::npos;

          if (false == found){
            found = true;
          }
        }
      }
    }
  }

  os.close();

  return found;
}
