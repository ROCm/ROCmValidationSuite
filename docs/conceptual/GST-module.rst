.. meta::
  :description: rocm validation suite documentation 
  :keywords: rocm validation suite, ROCm, documentation

.. _gst-module:


GST module
----------

The GPU Stress Test modules purpose is to bring the CUs of the specified
GPU(s) to a target performance level in gigaflops by doing large matrix
multiplications using SGEMM/DGEMM (Single/Double-precision General
Matrix Multiplication) available in a library like rocBlas. The GPU
stress module may be configured so it does not copy the source arrays to
the GPU before every matrix multiplication. This allows the GPU
performance to not be capped by device to host bandwidth transfers. The
module calculates how many matrix operations per second are necessary to
achieve the configured performance target and fails if it cannot achieve
that target. :raw-latex:`\n`:raw-latex:`\n`

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

target_stress

.. raw:: html

   </td>

.. raw:: html

   <td>

Float

.. raw:: html

   </td>

.. raw:: html

   <td>

The maximum relative performance the GPU will attempt to achieve in
gigaflops. This parameter is required.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

copy_matrix

.. raw:: html

   </td>

.. raw:: html

   <td>

Bool

.. raw:: html

   </td>

.. raw:: html

   <td>

This parameter indicates if each operation should copy the matrix data
to the GPU before executing. The default value is true.

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
to reach the given target_stress gigaflops. The default value is 5000 (5
seconds). This time is counted against the duration of the test. If the
target gflops, or stress, is not achieved in this time frame, the test
will fail. If the target stress (gflops) is achieved the test will
attempt to run for the rest of the duration specified by the action,
sustaining the stress load during that time.

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

A value indicating how much the target_stress can fluctuate after the
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

   <tr>

.. raw:: html

   <td>

matrix_size

.. raw:: html

   </td>

.. raw:: html

   <td>

Integer

.. raw:: html

   </td>

.. raw:: html

   <td>

Size of the matrices of the SGEMM operations. The default value is 5760.

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

target_stress

.. raw:: html

   </td>

.. raw:: html

   <td>

Time Series Floats

.. raw:: html

   </td>

.. raw:: html

   <td>

The average gflops over the last log interval.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

max_gflops

.. raw:: html

   </td>

.. raw:: html

   <td>

Float

.. raw:: html

   </td>

.. raw:: html

   <td>

The maximum sustained performance obtained by the GPU during the test.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

stress_violations

.. raw:: html

   </td>

.. raw:: html

   <td>

Integer

.. raw:: html

   </td>

.. raw:: html

   <td>

The number of gflops readings that violated the tolerance of the test
after the ramp interval.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

flops_per_op

.. raw:: html

   </td>

.. raw:: html

   <td>

Integer

.. raw:: html

   </td>

.. raw:: html

   <td>

Flops (floating point operations) per operation queued to the GPU queue.
One operation is one call to SGEMM/DGEMM.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

bytes_copied_per_op

.. raw:: html

   </td>

.. raw:: html

   <td>

Integer

.. raw:: html

   </td>

.. raw:: html

   <td>

Calculated number of ops/second necessary to achieve target gigaflops.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

try_ops_per_sec

.. raw:: html

   </td>

.. raw:: html

   <td>

Float

.. raw:: html

   </td>

.. raw:: html

   <td>

Calculated number of ops/second necessary to achieve target gigaflops.

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

‘true’ if the GPU achieves its desired sustained performance level.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   </table>

An informational message indicating will be emitted when the test starts
execution:

::

   [INFO ][<timestamp>][<action name>] gst <gpu id> start <target_stress> copy matrix: <copy_matrix>

During the execution of the test, informational output providing the
moving average the GPU(s) gflops will be logged at each log_interval:

::

   [INFO ][<timestamp>][<action name>] gst Gflops: <interval_gflops>

When the target gflops is achieved, the following message will be
logged:

::

   [INFO ][<timestamp>][<action name>] gst <gpu id> target achieved <target_stress>

If the target gflops, or stress, is not achieved in the “ramp_interval”
provided, the test will terminate and the following message will be
logged:

::

   [INFO ][<timestamp>][<action name>] gst <gpu id> ramp time exceeded <ramp_time>

In this case the test will fail.:raw-latex:`\n`

If the target stress (gflops) is achieved the test will attempt to run
for the rest of the duration specified by the action, sustaining the
stress load during that time. If the stress level violates the bounds
set by the tolerance level during that time a violation message will be
logged:

::

   [INFO ][<timestamp>][<action name>] gst <gpu id> stress violation <interval_gflops>

When the test completes, the following result message will be printed:

::

   [RESULT][<timestamp>][<action name>] gst <gpu id> Gflop: <max_gflops> flops_per_op:<flops_per_op> bytes_copied_per_op: <bytes_copied_per_op> try_ops_per_sec: <try_ops_per_sec> pass: <pass>

The test will pass if the target_stress is reached before the end of the ramp_interval and the stress_violations value is less than the given
max_violations value. Otherwise, the test will fail.

Examples
~~~~~~~~

When running the **GST** module, users should provide at least an action
name, the module name (gst), a list of GPU IDs, the test duration and a
target stress value (gigaflops). Thus, the most basic configuration file
looks like this:

::

   actions:
   - name: action_gst_1
     module: gst
     device: all
     target_stress: 3500
     duration: 8000

For the above configuration file, all the missing configuration keys
will have their default values (e.g.: **copy_matrix=true**,
**matrix_size=5760** etc.). For more information about the default
values, consult the dedicated sections (**3.3 Common Configuration
Keys** and **5.1 Configuration keys**).

When the **RVS** tool runs against such a configuration file, it will do
the following: 
- run the stress test on all available (and compatible)
AMD GPUs, one after the other 

- log a start message containing the GPU
ID, the **target_stress** and the value of the **copy_matrix**:

::

   [INFO  ] [164337.932824] action_gst_1 gst 50599 start 3500.000000 copy matrix:true

-  emit, each **log_interval** (e.g.: 1000ms), a message containing the
   gigaflops value that the current GPU achieved:

   [INFO ] [164355.111207] action_gst_1 gst 33367 Gflops 3535.670231

-  log a message as soon as the current GPU reaches the given
   **target_stress**:

   [INFO ] [164350.804843] action_gst_1 gst 33367 target achieved
   500.000000

-  log a **ramp time exceeded** message if the GPU was not able to reach
   the **target_stress** in the **ramp_interval** time frame (e.g.:
   5000). In such a case, the test will also terminate:
