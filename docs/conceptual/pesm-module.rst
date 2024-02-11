.. meta::
  :description: rocm validation suite documentation 
  :keywords: rocm validation suite, ROCm, documentation

.. _pesm-module:



PESM Module The PCIe State Monitor (PESM) tool is used to actively monitor the PCIe interconnect between the host platform and the GPU. The
module registers “listener” on a target GPUs PCIe interconnect, and log a message whenever it detects a state change. The PESM is able to detect
the following state changes:

1. PCIe link speed changes
2. GPU device power state changes

This module is intended to run concurrently with other actions, and provides a ‘start’ and ‘stop’ configuration key to start the monitoring
and then stop it after testing has completed. For information on GPU power state monitoring please consult the 7.6. PCI Power Management
Capability Structure, Gen 3 spec, page 601, device states D0-D3. For information on link status changes please consult the 7.8.8. Link Status
Register (Offset 12h), Gen 3 spec, page 635.

Monitoring is performed by polling respective PCIe registers roughly every 1ms (one millisecond).

Module Specific Keys
----------------------

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

This this key is set to true, the PESM module will start monitoring on
specified devices. If this key is set to false, all other keys are
ignored and monitoring will be stopped for all devices.

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

state

.. raw:: html

   </td>

.. raw:: html

   <td>

String

.. raw:: html

   </td>

.. raw:: html

   <td>

A string detailing the current power state of the GPU or the speed of
the PCIe link.

.. raw:: html

   </td>

.. raw:: HTML

When monitoring is started for a target GPU, a result message is logged
with the following format:

::

   [RESULT][<timestamp>][<action name>] pesm <gpu id> started

When monitoring is stopped for a target GPU, a result message is logged
with the following format:

::

   [RESULT][<timestamp>][<action name>] pesm all stopped

When monitoring is enabled, any detected state changes in link speed or
GPU power state will generate the following informational messages:

::

   [INFO ][<timestamp>][<action name>] pesm <gpu id> power state change <state>
   [INFO ][<timestamp>][<action name>] pesm <gpu id> link speed change <state>

   </tr>

.. raw:: HTML


Examples
~~~~~~~~

**Example 1**

Here is a typical check utilizing PESM functionality:

::

   actions:
   - name: action_1
     device: all
     module: pesm
     monitor: true
   - name: action_2
     device: 33367
     module: gst
     parallel: false
     count: 2
     wait: 100
     duration: 18000
     ramp_interval: 7000
     log_interval: 1000
     max_violations: 1
     copy_matrix: false
     target_stress: 5000
     tolerance: 0.07
     matrix_size: 5760
   - name: action_3
     device: all
     module: pesm
     monitor: false

-  **action_1** will initiate monitoring on all devices by setting key
   **monitor** to **true**\ :raw-latex:`\n`
-  **action_2** will start GPU stress test
-  **action_3** will stop monitoring

If executed like this:

::

   sudo rvs -c conf/pesm8.conf -d 3

output similar to this one can be produced:

::

   [RESULT] [497544.637462] [action_1] pesm all started
   [INFO  ] [497544.648299] [action_1] pesm 33367 link speed change 8 GT/s
   [INFO  ] [497544.648299] [action_1] pesm 33367 power state change D0
   [INFO  ] [497544.648733] [action_1] pesm 3254 link speed change 8 GT/s
   [INFO  ] [497544.648733] [action_1] pesm 3254 power state change D0
   [INFO  ] [497544.650413] [action_1] pesm 50599 link speed change 8 GT/s
   [INFO  ] [497544.650413] [action_1] pesm 50599 power state change D0
   [INFO  ] [497545.170392] [action_2] gst 33367 start 5000.000000 copy matrix:false
   [INFO  ] [497547.36602 ] [action_2] gst 33367 Gflops 6478.066983
   [INFO  ] [497548.69221 ] [action_2] gst 33367 target achieved 5000.000000
   [INFO  ] [497549.101219] [action_2] gst 33367 Gflops 5189.993529
   [INFO  ] [497550.132376] [action_2] gst 33367 Gflops 5189.993529
   ...
   [INFO  ] [497563.569370] [action_2] gst 33367 Gflops 5174.935520
   [RESULT] [497564.86904 ] [action_2] gst 33367 Gflop: 6478.066983 flops_per_op: 382.205952x1e9 bytes_copied_per_op: 398131200 try_ops_per_sec: 13.081952 pass: TRUE
   [INFO  ] [497564.220311] [action_2] gst 33367 start 5000.000000 copy matrix:false
   [INFO  ] [497566.70585 ] [action_2] gst 33367 Gflops 6521.049418
   [INFO  ] [497567.99929 ] [action_2] gst 33367 target achieved 5000.000000
   [INFO  ] [497568.143096] [action_2] gst 33367 Gflops 5130.281235
   ...
   [INFO  ] [497582.683893] [action_2] gst 33367 Gflops 5135.204729
   [RESULT] [497583.130945] [action_2] gst 33367 Gflop: 6521.049418 flops_per_op: 382.205952x1e9 bytes_copied_per_op: 398131200 try_ops_per_sec: 13.081952 pass: TRUE
   [RESULT] [497583.155470] [action_3] pesm all stopped

**Example 2:**

Consider this file:

::

   actions:
   - name: act1
     device: all
     deviceid: xxx
     module: pesm
     monitor: true

This file has and invalid entry in **deviceid** key. If execute, an
error will be reported:

::

   RVS-PESM: action: act1  invalide 'deviceid' key value: xxx

   </table>
