# CLI 

## Synopsis

<b>rvs</b>  [<b>-h</b>|<b>-g</b>|<b>-t</b>|<b>--version</b>|<b>--help</b>|<b>--listTests</b>|<b>--listGpus</b>]
<b>rvs</b> [[[<b>-d</b>|<b>--debugLevel</b>] 0|<b>--quiet</b>] | [[<b>-d</b>|<b>--debugLevel</b>] 1|2|3|4] | [[<b>-d</b>|<b>--debugLevel</b>] 5|<b>--verbose</b>|<b>-v</b>]]
[<b>-c</b> <i>path/config_file</i>]
[<b>-l</b> <i>path/log_file</i> [<b>-a</b>] [<b>-j</b>]] 

## Options

```
-a --appendLog     When generating a debug logfile, do not overwrite the content
                   of the current log. Use in conjuction with -d and -l options.

-c --config        Specify the test configuration file to use. This is a mandatory
                   field for test execution.

-d --debugLevel    Specify the debug level for the output log. The range is 0-5 with
                   5 being the highest verbose level.

-g --listGpus      List all the GPUs available in the machine, that RVS supports and
                   has visibility.

-i --indexes       Comma separated list of GPU ids/indexes to run test on. This overrides
                   the device/device_index values specified for every actions in the
                   configuration file, including the ‘all’ value.

-j --json          Generate output file in JSON format.
                   if a path follows this argument, that will be used as json log file;
                   else a file created in /var/tmp/ with timestamp in name.
-l --debugLogFile  Generate log file with output and debug information.


-t --listTests     List the test modules present in RVS.

-v --verbose       Enable verbose reporting. Equivalent to specifying -d 5 option.

-n --numTimes      Number of times the test repeatedly executes. Use in conjunction
                   with -c option.

   --quiet         No console output given. See logs and return code for errors.

   --version       Display version information and exit.

-h --help          Display usage information and exit.
```

## Exit Status

```
0        - if OK
non-zero - otherwise
```

## Examples


<b>rvs -c conf/gpup1.conf -d 3 -j -l mylog.txt</b>
Runs rvs with configuration file <i>conf/gpup1.conf</i> and writes text output into log file <i>mylog.txt</i> using logging level 3 (INFO) and writes to a file in /var/tmp/ folder in JSON format.Name of json log file will be printed to stdout/text log file

For more details consult the User Guide located in:
<i>[install_base]/userguide/html/index.html</i>
