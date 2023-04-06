# CLI 

## Synopsis

<b>rvs</b>  [<b>-h</b>|<b>-g</b>|<b>-t</b>|<b>--version</b>|<b>--help</b>|<b>--listTests</b>|<b>--listGpus</b>]
<b>rvs</b> [[[<b>-d</b>|<b>--debugLevel</b>] 0|<b>--quiet</b>] | [[<b>-d</b>|<b>--debugLevel</b>] 1|2|3|4] | [[<b>-d</b>|<b>--debugLevel</b>] 5|<b>--verbose</b>|<b>-v</b>]]
[<b>-c</b> <i>path/config_file</i>]
[<b>-l</b> <i>path/log_file</i> [<b>-a</b>] [<b>-j</b>]] 
[<b>-m</b> <i>module_path</i>]

## Options

```
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
```

## Exit Status

```
0        - if OK
non-zero - otherwise
```

## Examples

<b>rvs</b>
Runs rvs with the default test configuration file <i>[install_base]/conf/rvs.conf</i>

<b>rvs -c conf/gpup1.conf -d 3 -j -l mylog.txt</b>
Runs rvs with configuration file <i>conf/gpup1.conf</i> and writes output into log file <i>mylog.txt</i> using logging level 3 (INFO) in JSON format

For more details consult the User Guide located in:
<i>[install_base]/userguide/html/index.html</i>
