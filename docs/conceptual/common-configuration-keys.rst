


Common Configuration Keys
--------------------------

Common configuration keys applicable to most module are summarized in the table below:

+----------------------------------------------------------------------------------+
| <table>                                                                          |
+==================================================================================+
| <tr><th>Short option</th><th>Long option</th><th> Description</th></tr>          |
| <tr><td>-a</td><td>\-\-appendLog</td><td>When generating a debug logfile,        |
| do not overwrite the contents                                                    |
| of a current log. Used in conjunction with the -d and -l options.                |
| </td></tr>                                                                       |
|                                                                                  |
| <tr><td>-c</td><td>\-\-config</td><td>Specify the configuration file to be used. |
| The default is \<installbase\>/RVS/conf/RVS.conf                                 |
| </td></tr>                                                                       |
|                                                                                  |
| <tr><td></td><td>\-\-configless</td><td>Run RVS in a configless mode.            |
| Executes a "long" test on all supported GPUs.</td></tr>                          |
|                                                                                  |
| <tr><td>-d</td><td>\-\-debugLevel</td><td>Specify the debug level for the output |
| log. The range is 0 to 5 with 5 being the most verbose.                          |
| Used in conjunction with the -l flag.</td></tr>                                  |
|                                                                                  |
| <tr><td>-g</td><td>\-\-listGpus</td><td>List the GPUs available and exit.        |
| This will only list GPUs that are supported by RVS.</td></tr>                    |
|                                                                                  |
| <tr><td>-i</td><td>\-\-indexes</td><td>Comma separated list of  devices to run   |
| RVS on. This will override the device values specified in the configuration file |
| for every action in the configuration file, including the "all" value.</td></tr> |
|                                                                                  |
| <tr><td>-j</td><td>\-\-json</td><td>Output should use the JSON format.</td></tr> |
|                                                                                  |
| <tr><td>-l</td><td>\-\-debugLogFile</td><td>Specify the logfile for debug        |
| information. This will produce a log file intended for post-run analysis after   |
| an error.</td></tr>                                                              |
|                                                                                  |
| <tr><td></td><td>\-\-quiet</td><td>No console output given. See logs and return  |
| code for errors.</td></tr>                                                       |
|                                                                                  |
| <tr><td>-m</td><td>\-\-modulepath</td><td>Specify a custom path for the RVS      |
| modules.</td></tr>                                                               |
|                                                                                  |
| <tr><td></td><td>\-\-specifiedtest</td><td>Run a specific test in a configless   |
| mode. Multiple word tests should be in quotes. This action will default to all   |
| devices, unless the \-\-indexes option is specifie.</td></tr>                    |
|                                                                                  |
| <tr><td>-t</td><td>\-\-listTests</td><td>List the modules available to be        |
| executed through RVS and exit. This will list only the readily loadable modules  |
| given the current path and library conditions.</td></tr>                         |
|                                                                                  |
| <tr><td>-v</td><td>\-\-verbose</td><td>Enable verbose reporting. This is         |
| equivalent to specifying the -d 5 option.</td></tr>                              |
|                                                                                  |
| <tr><td></td><td>\-\-version</td><td>Displays the version information and exits. |
| </td></tr>                                                                       |
|                                                                                  |
| <tr><td>-h</td><td>\-\-help</td><td>Display usage information and exit.          |
| </td></tr>                                                                       |
|                                                                                  |
| </table>                                                                         |
+----------------------------------------------------------------------------------+
