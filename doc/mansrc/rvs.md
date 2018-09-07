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

@file

@page rvs
ROCm Validation Suite

@section synopsis SYNOPSIS
<b>rvs</b>  [<b>-h</b>|<b>-g</b>|<b>-t</b>|<b>--version</b>|<b>--help</b>|<b>--listTests</b>|<b>--listGpus</b>]\n
<b>rvs</b> [[[<b>-d</b>|<b>--debugLevel</b>] 0|<b>--quiet</b>] | [[<b>-d</b>|<b>--debugLevel</b>] 1|2|3|4] |\n [[<b>-d</b>|<b>--debugLevel</b>] 5|<b>--verbose</b>|<b>-v</b>]]\n
[<b>-c</b> <i>path/config_file</i>]\n
[<b>-l</b> <i>path/log_file</i> [<b>-a</b>] [<b>-j</b>]] \n
[<b>-m</b> <i>module_path</i>]
@section description DESCRIPTION
The ROCm Validation Suite (RVS) is a system administrator’s and cluster manager's tool for detecting and troubleshooting common problems affecting AMD GPU(s) running in a high-performance computing environment, enabled using the ROCm software stack on a compatible platform.

The RVS is a collection of tests, benchmarks and qualification tools each targeting a specific sub-system of the ROCm platform. All of the tools are implemented in software and share a common command line interface. Each set of tests are implemented in a “module” which is a library encapsulating the functionality specific to the tool. The CLI can specify the directory containing modules to use when searching for libraries to load. Each module may have a set of options that it defines and a configuration file that supports its execution.

@section options OPTIONS
@verbatim
-a --appendLog     When generating a debug logfile, do not overwrite the contents
                   of a current log. Used in conjuction with the -d and -l options
-c --config        Specify the configuration file to be used.
                   The default is <install base>/conf/RVS.conf
   --configless    Run RVS in a configless mode. Executes a "long" test on all
                   supported GPUs.
-d --debugLevel    Specify the debug level for the output log. The range is
                   0 to 5 with 5 being the most verbose.
                   Used in conjunction with the -l flag.
-g --listGpus      List the GPUs available and exit. This will only list GPUs
                   that are supported by RVS.
-i --indexes       Comma separated list of indexes devices to run RVS on. This will
                   override the device values specified in the configuration file for
                   every action in the configuration file, including the ‘all’ value.
-j --json          Output should use the JSON format.
-l --debugLogFile  Specify the logfile for debug information. This will produce a log
                   file intended for post-run analysis after an error.
   --quiet         No console output given. See logs and return code for errors.
-m --modulepath    Specify a custom path for the RVS modules.
   --specifiedtest Run a specific test in a configless mode. Multiple word tests
                   should be in quotes. This action will default to all devices,
                   unless the indexes option is specifie.
-t --listTests     List the modules available to be executed through RVS and exit.
                   This will list only the readily loadable modules
                   given the current path and library conditions.
-v --verbose       Enable verbose reporting. This is equivalent to
                   specifying the -d 5 option.
   --version       Displays the version information and exits.
-h --help          Display usage information and exit.

@endverbatim

@section exitstatus EXIT STATUS

@verbatim
0        - if OK
non-zero - otherwise

@endverbatim

@section examples EXAMPLES
<b>rvs</b> \n
Runs rvs with the default test configuration file <i>[install_base]/conf/rvs.conf</i>

<b>rvs -c conf/gpup1.conf -d 3 -j -l mylog.txt</b> \n
Runs rvs with configuration file <i>conf/gpup1.conf</i> and writes output \n
into log file <i>mylog.txt</i> using logging level 3 (INFO) in JSON format

For more details consult the User Guide located in:
<i>[install_base]/userguide/html/index.html</i>