


IET Module
----------

The Input EDPp Test can be used to characterize the peak power
capabilities of a GPU to different levels of use. This tool can leverage
the functionality of the GST to drive the compute load on the GPU, but
the test will use different configuration and output keys and should
focus on driving power usage rather than calculating compute load. The
purpose of the IET module is to bring the GPU(s) to a pre-configured
power level in watts by gradually increasing the compute load on the
GPUs until the desired power level is achieved. This verifies that the
GPUs can sustain a power level for a reasonable amount of time without
problems like thermal violations arising.:raw-latex:`\n`

This module should be used in conjunction with the GPU Monitor, to watch
for thermal, power and related anomalies while the target GPU(s) are
under realistic load conditions. By setting the appropriate parameters a
user can ensure that all GPUs in a node or cluster reach desired
performance levels. Further analysis of the generated stats can also
show variations in the required power, clocks or temperatures to reach
these targets, and thus highlight GPUs or nodes that are operating less
efficiently.

Module Specific Keys
~~~~~~~~~~~~~~~~~~~~

Module specific keys are described in the table below:

.. raw:: html

   <table>

.. raw:: html

   <tr>

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

   <tr>

.. raw:: html

   <td>

target_power

.. raw:: html

   </td>

.. raw:: html

   <td>

Float

.. raw:: html

   </td>

.. raw:: html

   <td>

This is a floating point value specifying the target sustained power
level for the test.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

ramp_interval

.. raw:: html

   </td>

.. raw:: html

   <td>

Integer

.. raw:: html

   </td>

.. raw:: html

   <td>

This is an time interval, specified in milliseconds, given to the test
to determine the compute load that will sustain the target power. The
default value is 5000 (5 seconds). This time is counted against the
duration of the test.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

tolerance

.. raw:: html

   </td>

.. raw:: html

   <td>

Float

.. raw:: html

   </td>

.. raw:: html

   <td>

A value indicating how much the target_power can fluctuate after the
ramp period for the test to succeed. The default value is 0.1 or 10%.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

max_violations

.. raw:: html

   </td>

.. raw:: html

   <td>

Integer

.. raw:: html

   </td>

.. raw:: html

   <td>

The number of tolerance violations that can occur after the
ramp_interval for the test to still pass. The default value is 0.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

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

The sampling rate for target_power values given in milliseconds. The
default value is 100 (.1 seconds).

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

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

This is a positive integer, given in milliseconds, that specifies an
interval over which the moving average of the bandwidth will be
calculated and logged.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   </table>

Output
~~~~~~

Module specific output keys are described in the table below:

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

current_power

.. raw:: html

   </td>

.. raw:: html

   <td>

Time Series Floats

.. raw:: html

   </td>

.. raw:: html

   <td>

The current measured power of the GPU.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

power_violations

.. raw:: html

   </td>

.. raw:: html

   <td>

Integer

.. raw:: html

   </td>

.. raw:: html

   <td>

The number of power reading that violated the tolerance of the test
after the ramp interval.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

pass

.. raw:: html

   </td>

.. raw:: html

   <td>

Bool

.. raw:: html

   </td>

.. raw:: html

   <td>

‘true’ if the GPU achieves its desired sustained power level in the ramp
interval.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   </table>

Examples
~~~~~~~~

**Example 1:**

A regular IET configuration file looks like this:

::

   actions:
   - name: action_1
     device: all
     module: iet
     parallel: false
     count: 2
     wait: 100
     duration: 10000
     ramp_interval: 5000
     sample_interval: 500
     log_interval: 500
     max_violations: 1
     target_power: 135
     tolerance: 0.1
     matrix_size: 5760

*Please note:* - when setting the ‘device’ configuration key to ‘all’,
the RVS will detect all the AMD compatible GPUs and run the test on all
of them - the test will run 2 times on each GPU (count = 2) - only one
power violation is allowed. If the total number of violations is bigger
than 1 the IET test result will be marked as ‘failed’

When the RVS tool runs against such a configuration file, it will do the
following: - run the test on all AMD compatible GPUs

-  log a start message containing the GPU ID and the target_power, e.g.:

   [INFO ] [167316.308057] action_1 iet 50599 start 135.000000

-  emit, each log_interval (e.g.: 500ms), a message containing the power
   for the current GPU

   [INFO ] [167319.266707] action_1 iet 50599 current power 136.878342

-  log a message as soon as the current GPU reaches the given
   target_power

   [INFO ] [167318.793062] action_1 iet 50599 target achieved 135.000000

-  log a ‘ramp time exceeded’ message if the GPU was not able to reach
   the target_power in the ramp_interval time frame (e.g.: 5000ms). In
   such a case, the test will also terminate

   [INFO ] [167648.832413] action_1 iet 50599 ramp time exceeded 5000

-  log a ‘power violation message’ when the current power (for the last
   sample_interval, e.g.; 500ms) violates the bounds set by the
   tolerance configuration key (e.g.: 0.1). Please note that this
   message is never logged during the ramp_interval time frame

   [INFO ] [161251.971277] action_1 iet 3254 power violation 73.783211

-  log the test result, when the stress test completes.

   [RESULT] [167305.260051] action_1 iet 33367 pass: TRUE

The output for such a configuration file may look like this:

::

   [INFO ] [167261.27161 ] action_1 iet 33367 start 135.000000
   [INFO ] [167263.516803] action_1 iet 33367 current power 136.934479
   [INFO ] [167263.521355] action_1 iet 33367 target achieved 135.000000
   [INFO ] [167264.16925 ] action_1 iet 33367 current power 138.421844
   [INFO ] [167264.517018] action_1 iet 33367 current power 138.394608
   ...
   [INFO ] [167271.518402] action_1 iet 33367 current power 139.231918
   [RESULT] [167272.67686 ] action_1 iet 33367 pass: TRUE
   [INFO ] [167272.68029 ] action_1 iet 3254 start 135.000000
   [INFO ] [167274.552026] action_1 iet 3254 current power 139.363525
   [INFO ] [167274.552059] action_1 iet 3254 target achieved 135.000000
   [INFO ] [167275.52168 ] action_1 iet 3254 current power 138.661453
   [INFO ] [167275.552241] action_1 iet 3254 current power 138.857635
   ...
   [INFO ] [167282.553983] action_1 iet 3254 current power 140.069687
   [RESULT] [167283.95763 ] action_1 iet 3254 pass: TRUE
   [INFO ] [167283.96158 ] action_1 iet 50599 start 135.000000
   [INFO ] [167285.532999] action_1 iet 50599 current power 137.205032
   [INFO ] [167285.543084] action_1 iet 50599 target achieved 135.000000
   [INFO ] [167286.33050 ] action_1 iet 50599 current power 136.137115
   ...
   [INFO ] [167293.534672] action_1 iet 50599 current power 139.753464
   [RESULT] [167294.131420] action_1 iet 50599 pass: TRUE

**Example 2:**

Another configuration file, which may raise some ‘power violation’
messages (due to the small tolerance value) looks like this

::

   - name: action_1
     device: all
     module: iet
     parallel: false
     count: 1
     wait: 100
     duration: 8000
     ramp_interval: 5000
     sample_interval: 700
     log_interval: 700
     max_violations: 1
     target_power: 80
     tolerance: 0.06
     matrix_size: 5760

The output for such a configuration file may look like this:

::

   [INFO ] [161236.677785] action_1 iet 33367 start 80.000000
   [INFO ] [161239.350055] action_1 iet 33367 current power 84.186142
   [INFO ] [161239.354542] action_1 iet 33367 target achieved 80.000000
   ...
   [INFO ] [161241.450517] action_1 iet 33367 current power 77.001945
   [INFO ] [161241.459600] action_1 iet 33367 power violation 75.163689
   [INFO ] [161242.150642] action_1 iet 33367 current power 82.063576
   [RESULT] [161245.698113] action_1 iet 33367 pass: TRUE
   [INFO ] [161245.698525] action_1 iet 3254 start 80.000000
   [INFO ] [161248.394003] action_1 iet 3254 current power 78.842796
   [INFO ] [161248.418631] action_1 iet 3254 target achieved 80.000000
   [INFO ] [161249.94149 ] action_1 iet 3254 current power 79.938454
   ...
   [INFO ] [161249.794201] action_1 iet 3254 current power 76.511711
   [INFO ] [161249.818803] action_1 iet 3254 power violation 74.279594
   [INFO ] [161250.494263] action_1 iet 3254 current power 74.615120
   ...
   [INFO ] [161254.117386] action_1 iet 3254 power violation 73.682312
   [RESULT] [161254.738939] action_1 iet 3254 pass: FALSE
   [INFO ] [161254.739387] action_1 iet 50599 start 80.000000
   [INFO ] [161257.374079] action_1 iet 50599 current power 81.560165
   [INFO ] [161257.392085] action_1 iet 50599 target achieved 80.000000
   [INFO ] [161258.774304] action_1 iet 50599 current power 75.057304
   ...
   [INFO ] [161262.974833] action_1 iet 50599 current power 80.200668
   [RESULT] [161263.771631] action_1 iet 50599 pass: TRUE

*Important notes:*

-  all the missing configuration keys (if any) will have their default
   values. For more information about the default values please consult
   the dedicated sections (3.3 Common Configuration Keys and 13.1 Module
   specific keys).

-  if a mandatory configuration key is missing, the RVS tool will log an
   error message and terminate the execution of the current module. For
   example, if the target_power is missing, the RVS to terminate with
   the following error message: “RVS-IET: action: action_1 key
   ‘target_power’ was not found”

-  it is essential that all the configuration keys will be
   adjusted/fine-tuned according to the actual GPUs and HW platform
   capabilities.

   For example, a matrix size of 5760 should fit the VEGA 10 GPUs while 8640 should work with the VEGA 20 GPUs

   For small target_power values (e.g.: 30-40W), the sample_interval should be increased, otherwise the IET may fail either to      achieve the given target_power or to sustain it (e.g.: ramp_interval = 1500 for target_power = 40)

   In case there are problems reaching/sustaining the given target_power, increase the ramp_interval and/or the tolerance           value(s) and try again (in case of a ‘ramp time exceeded’ message), and increase the tolerance value (in case too many ‘power
   violation message’ is logged out).
