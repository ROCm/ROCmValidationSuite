.. meta::
  :description: rocm validation suite documentation 
  :keywords: rocm validation suite, ROCm, documentation

.. _GPU-monitor-module:


GM Module
************

The GPU monitor module can be used to monitor and characterize the response of a GPU to different levels of use. This module is intended to run concurrently with other actions and provides a ‘start’ and ‘stop’ configuration key to start the monitoring and then stop it after testing has been completed. The module can also be configured with bounding box values for interested GPU parameters. If any of the GPU’s parameters exceed the bounding values on a specific GPU an INFO warning message will be printed to stdout while the bounding value is still exceeded.

Module Specific Keys
-----------

.. raw:: html

   <table>

.. raw:: html

   <thead>

.. raw:: html

   <tr class="header">

.. raw:: html

   <th>

Config Key

.. raw:: html

   </th>

.. raw:: html

   <th>

Type

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

   </thead>

.. raw:: html

   <tbody>

.. raw:: html

   <tr class="odd">

.. raw:: html

   <td>

monitor

.. raw:: html

   </td>

.. raw:: html

   <td>

Bool

.. raw:: html

   </td>

.. raw:: html

   <td>

If this this key is set to true, the GM module will start monitoring on
specified devices. If this key is set to false, all other keys are
ignored and monitoring of the specified device will be stopped.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr class="even">

.. raw:: html

   <td>

metrics

.. raw:: html

   </td>

.. raw:: html

   <td>

Collection of Structures, specifying the metric, if there are bounds and
the bound values. The structures have the following
format::raw-latex:`\n{String, Bool, Integer, Integer}`

.. raw:: html

   </td>

.. raw:: html

   <td>

The set of metrics to monitor during the monitoring period. Example
values
are::raw-latex:`\n{‘temp’, ‘true’, max_temp, min_temp}`:raw-latex:`\n {‘clock’, ‘false’,
max_clock, min_clock}`:raw-latex:`\n {‘mem_clock’, ‘true’, max_mem_clock,
min_mem_clock}`:raw-latex:`\n {‘fan’, ‘true’, max_fan, min_fan}`:raw-latex:`\n {‘power’, ‘true’,
max_power, min_power}`:raw-latex:`\n `The set of upper bounds for each
metric are specified as an integer. The units and values for each metric
are::raw-latex:`\n` temp - degrees Celsius:raw-latex:`\n `clock - MHz
:raw-latex:`\n `mem_clock - MHz :raw-latex:`\n `fan - Integer between 0
and 255 :raw-latex:`\n `power - Power in Watts

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr class="odd">

.. raw:: html

   <td>

sample_interval

.. raw:: html

   </td>

.. raw:: html

   <td>

Integer

.. raw:: html

   </td>

.. raw:: html

   <td>

If this key is specified metrics will be sampled at the given rate. The
units for the sample_interval are milliseconds. The default value is
1000.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr class="even">

.. raw:: html

   <td>

log_interval

.. raw:: html

   </td>

.. raw:: html

   <td>

Integer

.. raw:: html

   </td>

.. raw:: html

   <td>

If this key is specified informational messages will be emitted at the
given interval, providing the current values of all parameters
specified. This parameter must be equal to or greater than the sample
rate. If this value is not specified, no logging will occur.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr class="odd">

.. raw:: html

   <td>

terminate

.. raw:: html

   </td>

.. raw:: html

   <td>

Bool

.. raw:: html

   </td>

.. raw:: html

   <td>

If the terminate key is true the GM monitor will terminate the RVS
process when a bounds violation is encountered on any of the metrics
specified.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr class="even">

.. raw:: html

   <td>

force

.. raw:: html

   </td>

.. raw:: html

   <td>

Bool

.. raw:: html

   </td>

.. raw:: html

   <td>

If ‘true’ and terminate key is also ‘true’ the RVS process will
terminate immediately. **Note:** this may cose resource leaks within
GPUs.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   </tbody>

.. raw:: html

   </table>

Output
-------

Module-specific output keys are described in the table below:

.. raw:: html

   <table>

.. raw:: html

   <tr>

.. raw:: html

   <th>

Output Key

.. raw:: html

   </th>

.. raw:: html

   <th>

Type

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

metric_values

.. raw:: html

   </td>

.. raw:: html

   <td>

Time Series Collection of Result Integers

.. raw:: html

   </td>

.. raw:: html

   <td>

A collection of integers containing the result values for each of the
metrics being monitored.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

metric_violations

.. raw:: html

   </td>

.. raw:: html

   <td>

Collection of Result Integers

.. raw:: html

   </td>

.. raw:: html

   <td>

A collection of integers containing the violation count for each of the
metrics being monitored.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

metric_average

.. raw:: html

   </td>

.. raw:: html

   <td>

Collection of Result Integers

.. raw:: html

   </td>

.. raw:: html

   <td>

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   </table>

When monitoring is started for a target GPU, a result message is logged
with the following format:

::

   [RESULT][<timestamp>][<action name>] gm <gpu id> started

In addition, an informational message is provided for each for each
metric being monitored:

::

   [INFO ][<timestamp>][<action name>] gm <gpu id> monitoring <metric> bounds min:<min_metric> max: <max_metric>

During the monitoring informational output regarding the metrics of the
GPU will be sampled at every interval specified by the sample_rate key.
If a bounding box violation is discovered during a sampling interval, a
warning message is logged with the following format:

::

   [INFO ][<timestamp>][<action name>] gm <gpu id> <metric> bounds violation <metric value>

If the log_interval value is set an information message for each metric
is logged at every interval using the following format:

::

   [INFO ][<timestamp>][<action name>] gm <gpu id> <metric> <metric_value>

When monitoring is stopped for a target GPU, a result message is logged
with the following format:

::

   [RESULT][<timestamp>][<action name>] gm <gpu id> gm stopped

The following messages, reporting the number of metric violations that
were sampled over the duration of the monitoring and the average metric
value is reported:

::

   [RESULT][<timestamp>][<action name>] gm <gpu id> <metric> violations <metric_violations>
   [RESULT][<timestamp>][<action name>] gm <gpu id> <metric> average <metric_average>


Examples
--------

**Example 1:**

Consider action:

::

   actions:
   - name: action_1
     module: gm
     device: all
     monitor: true
     metrics:
       temp: true 20 0
       fan: true 10 0
     duration: 5000
   - name: another_action
   ...

This action will monitor temperature and fan speed for 5 seconds and
then continue with the next action. Output for such configuration may
be:

::

   [RESULT] [694381.521373] [action_1] gm 33367 started
   [INFO  ] [694381.531803] action_1 gm 33367  monitoring temp bounds min:0 max:20
   [INFO  ] [694381.531817] action_1 gm 33367  monitoring temp bounds min:0 max:20
   [INFO  ] [694381.531828] action_1 gm 33367  monitoring fan bounds min:0 max:10
   [RESULT] [694381.521373] [action_1] gm 3254 started
   [INFO  ] [694381.532257] action_1 gm 3254  monitoring temp bounds min:0 max:20
   [INFO  ] [694381.532276] action_1 gm 3254  monitoring temp bounds min:0 max:20
   [INFO  ] [694381.532293] action_1 gm 3254  monitoring fan bounds min:0 max:10
   [RESULT] [694381.521373] [action_1] gm 50599 started
   [INFO  ] [694381.534471] action_1 gm 50599  monitoring temp bounds min:0 max:20
   [INFO  ] [694381.534487] action_1 gm 50599  monitoring temp bounds min:0 max:20
   [INFO  ] [694381.534502] action_1 gm 50599  monitoring fan bounds min:0 max:10
   [INFO  ] [694381.534623] action_1 gm 33367 temp  bounds violation 22C
   [INFO  ] [694381.534822] action_1 gm 3254 temp  bounds violation 22C
   [INFO  ] [694381.534946] action_1 gm 50599 temp  bounds violation 22C
   [INFO  ] [694382.535329] action_1 gm 33367 temp  bounds violation 22C
   ...
   [INFO  ] [694385.537777] action_1 gm 50599 temp  bounds violation 21C
   [RESULT] [694386.538037] [action_1] gm 3254 stopped
   [RESULT] [694386.538037] [action_1] gm 50599 stopped
   [RESULT] [694386.538037] [action_1] gm 33367 stopped
   [RESULT] [694386.521449] [action_1] gm 3254 temp violations 1
   [RESULT] [694386.521449] [action_1] gm 3254 temp average 19C
   [RESULT] [694386.521449] [action_1] gm 3254 fan violations 0
   [RESULT] [694386.521449] [action_1] gm 3254 fan average 0%
   [RESULT] [694386.521449] [action_1] gm 50599 temp violations 5
   [RESULT] [694386.521449] [action_1] gm 50599 temp average 21C
   [RESULT] [694386.521449] [action_1] gm 50599 fan violations 0
   [RESULT] [694386.521449] [action_1] gm 50599 fan average 0%
   [RESULT] [694386.521449] [action_1] gm 33367 temp violations 5
   [RESULT] [694386.521449] [action_1] gm 33367 temp average 22C
   [RESULT] [694386.521449] [action_1] gm 33367 fan violations 0
   [RESULT] [694386.521449] [action_1] gm 33367 fan average 0%

**Example 2:**

Consider action:

::

   actions:
   - name: action_1
     module: gm
     device: all
     monitor: true
     metrics:
       temp: true 20 0
       fan: true 10 0
       power: true 100 0
     sample_interval: 1000
     log_interval: 1200
     terminate: false
     duration: 5000

This configuration is similar to that in *Example 1* but has explicitly
given values for *sample_interval* and *log_interval*. Output is similar
to the previous one but averaging and the printout are performed at a
different rate.

**Example 3:**

Consider action with syntax error (‘temp’ key is missing lower value):

::

   actions:
   - name: action_1
     module: gm
     device: 33367 50599
     monitor: true
     metrics:
       temp: true 20
       fan: true 10 0
       power: true 100 0
     sample_interval: 1000
     log_interval: 1200

Output for such configuration is:

::

   RVS-GM: action: action_1 Wrong number of metric parameters

**Example 4:**

Consider action with logical error:

::

   actions:
   - name: action_1
     module: gm
     device: all
     monitor: true
     metrics:
       temp: false 20 0
       clock: true 1500 852
       power: true 100 0
     sample_interval: 5000
     log_interval: 4000
     duration: 8000

Output for such configuration is:

::

   RVS-GM: action: action_1 Log interval has a lower value than the sample interval
