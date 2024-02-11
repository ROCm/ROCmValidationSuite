


Common Configuration Keys
--------------------------

Common configuration keys applicable to most module are summarized in the table below:


.. raw:: html

   <table>

.. raw:: html

   <tr>

.. raw:: html

   <th>

Short option

.. raw:: html

   </th>

.. raw:: html

   <th>

Long option

.. raw:: html

   </th>

.. raw:: html

   <th>

Description

.. raw:: html

   </th>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

-a

.. raw:: html

   </td>

.. raw:: html

   <td>

--appendLog

.. raw:: html

   </td>

.. raw:: html

   <td>

When generating a debug logfile, do not overwrite the contents of a
current log. Used in conjunction with the -d and -l options.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

-c

.. raw:: html

   </td>

.. raw:: html

   <td>

--config

.. raw:: html

   </td>

.. raw:: html

   <td>

Specify the configuration file to be used. The default is
<installbase>/RVS/conf/RVS.conf

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

.. raw:: html

   </td>

.. raw:: html

   <td>

--configless

.. raw:: html

   </td>

.. raw:: html

   <td>

Run RVS in a configless mode. Executes a “long” test on all supported
GPUs.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

-d

.. raw:: html

   </td>

.. raw:: html

   <td>

--debugLevel

.. raw:: html

   </td>

.. raw:: html

   <td>

Specify the debug level for the output log. The range is 0 to 5 with 5
being the most verbose. Used in conjunction with the -l flag.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

-g

.. raw:: html

   </td>

.. raw:: html

   <td>

--listGpus

.. raw:: html

   </td>

.. raw:: html

   <td>

List the GPUs available and exit. This will only list GPUs that are
supported by RVS.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

-i

.. raw:: html

   </td>

.. raw:: html

   <td>

--indexes

.. raw:: html

   </td>

.. raw:: html

   <td>

Comma separated list of devices to run RVS on. This will override the
device values specified in the configuration file for every action in
the configuration file, including the “all” value.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

-j

.. raw:: html

   </td>

.. raw:: html

   <td>

--json

.. raw:: html

   </td>

.. raw:: html

   <td>

Output should use the JSON format.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

-l

.. raw:: html

   </td>

.. raw:: html

   <td>

--debugLogFile

.. raw:: html

   </td>

.. raw:: html

   <td>

Specify the logfile for debug information. This will produce a log file
intended for post-run analysis after an error.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

.. raw:: html

   </td>

.. raw:: html

   <td>

--quiet

.. raw:: html

   </td>

.. raw:: html

   <td>

No console output given. See logs and return code for errors.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

-m

.. raw:: html

   </td>

.. raw:: html

   <td>

--modulepath

.. raw:: html

   </td>

.. raw:: html

   <td>

Specify a custom path for the RVS modules.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

.. raw:: html

   </td>

.. raw:: html

   <td>

--specifiedtest

.. raw:: html

   </td>

.. raw:: html

   <td>

Run a specific test in a configless mode. Multiple word tests should be
in quotes. This action will default to all devices, unless the --indexes
option is specifie.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

-t

.. raw:: html

   </td>

.. raw:: html

   <td>

--listTests

.. raw:: html

   </td>

.. raw:: html

   <td>

List the modules available to be executed through RVS and exit. This
will list only the readily loadable modules given the current path and
library conditions.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

-v

.. raw:: html

   </td>

.. raw:: html

   <td>

--verbose

.. raw:: html

   </td>

.. raw:: html

   <td>

Enable verbose reporting. This is equivalent to specifying the -d 5
option.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

.. raw:: html

   </td>

.. raw:: html

   <td>

--version

.. raw:: html

   </td>

.. raw:: html

   <td>

Displays the version information and exits.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

-h

.. raw:: html

   </td>

.. raw:: html

   <td>

--help

.. raw:: html

   </td>

.. raw:: html

   <td>

Display usage information and exit.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   </table>
