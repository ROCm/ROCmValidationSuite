#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

#include "include/metaPackageInfo.h"
#include "include/rcutils.h"

bool PackageInfo::fillPkgInfo(){

  std::stringstream ss;
  bool status;

  status = getPackageInfo(m_pkgname, m_pkgmgrname, m_dependcmd1name, m_dependcmd2name, ss);
  if (true != status) {
    std::cout << "getPackageInfo failed !!!" << std::endl;
    return false;
  }

  readMetaPackageInfo(ss.str());

  return true;

}

