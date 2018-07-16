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

//! @~English
//! @mainpage RVS
//!
//! @section name NAME
//!
//! (RVS description)
//!
//! @section synopsis SYNOPSIS
//!
//! @section description DESCRIPTION
//!
//! @section exitstatus EXIT STATUS
//!
//! @section history HISTORY
//!
//! @section author AUTHOR
//!
//! @version 1.1
//! $Date: Sun Dec 25 07:02:41 2016 -0200 $


/** \defgroup Launcher Launcher module
 *
 * \brief Launcher module implementation
 *
 * This is the starting point of rvs utility. Launcher will parse the command line
 * as well as the input YAML configuration file and load appropriate test modules.
 * Then, it will invoke tests specified in .conf file and provide logging functionality
 * so that modules can print on screen of in the log file.
 */

#include <iostream>


#include "rvscli.h"
#include "rvsexec.h"


using namespace std;
using namespace rvs;

/**
 *
 * \ingroup Launcher
 * \brief Main method
 * @section 1 Test
 * standard C main() method.
 *
 * @param Argc standard C argc parameter to main()
 * @param Argv standard C argv parameter to main()
 * @return 0 - all OK, non-zero error
 *
 * */
int main(int Argc, char**Argv)
{
  int sts;
  cli cli;

  sts =  cli.parse(Argc, Argv);
  if(sts)
  {
    cerr << "ERROR: error parsing command line:" << cli.get_error_string() << endl;
    return -1;
  }

  exec executor;
  sts = executor.run();

  return sts;
}




