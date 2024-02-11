.. meta::
  :description: rocm validation suite documentation 
  :keywords: rocm validation suite, ROCm, documentation

.. _pebb-module:


PEBB Module
-----------

The PCIe Bandwidth Benchmark attempts to saturate the PCIe bus with DMA
transfers between system memory and a target GPU card’s memory. These
are known as host-to-device or device- to-host transfers, and can be
either unidirectional or bidirectional transfers. The maximum bandwidth
obtained is reported.

Module Specific Keys
~~~~~~~~~~~~~~~~~~~~

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

host_to_device

.. raw:: html

   </td>

.. raw:: html

   <td>

Bool

.. raw:: html

   </td>

.. raw:: html

   <td>

This key indicates if host to device transfers will be considered. The
default value is true.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

device_to_host

.. raw:: html

   </td>

.. raw:: html

   <td>

Bool

.. raw:: html

   </td>

.. raw:: html

   <td>

This key indicates if device to host transfers will be considered. The
default value is true.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

parallel

.. raw:: html

   </td>

.. raw:: html

   <td>

Bool

.. raw:: html

   </td>

.. raw:: html

   <td>

This option is only used if the test_bandwidth key is
true.:raw-latex:`\n` - true – Run all test transfers in
parallel.:raw-latex:`\n` - false – Run test transfers one by one.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

duration

.. raw:: html

   </td>

.. raw:: html

   <td>

Integer

.. raw:: html

   </td>

.. raw:: html

   <td>

This option is only used if test_bandwidth is true. This key specifies
the duration a transfer test should run, given in milliseconds. If this
key is not specified, the default value is 10000 (10 seconds).

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

This option is only used if test_bandwidth is true. This is a positive
integer, given in milliseconds, that specifies an interval over which
the moving average of the bandwidth will be calculated and logged. The
default value is 1000 (1 second). It must be smaller than the duration
key.:raw-latex:`\n` if this key is 0 (zero), results are displayed as
soon as the test transfer is completed.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

block_size

.. raw:: html

   </td>

.. raw:: html

   <td>

Collection of Integers

.. raw:: html

   </td>

.. raw:: html

   <td>

Optional. Defines list of block sizes to be used in transfer
tests.:raw-latex:`\n` If “all” or missing list of block sizes used in
rocm_bandwidth_test is used: - 1 \* 1024 - 2 \* 1024 - 4 \* 1024 - 8 \*
1024 - 16 \* 1024 - 32 \* 1024 - 64 \* 1024 - 128 \* 1024 - 256 \* 1024
- 512 \* 1024 - 1 \* 1024 \* 1024 - 2 \* 1024 \* 1024 - 4 \* 1024 \*
1024 - 8 \* 1024 \* 1024 - 16 \* 1024 \* 1024 - 32 \* 1024 \* 1024 - 64
\* 1024 \* 1024 - 128 \* 1024 \* 1024 - 256 \* 1024 \* 1024 - 512 \*
1024 \* 1024

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

b2b_block_size

.. raw:: html

   </td>

.. raw:: html

   <td>

Integer

.. raw:: html

   </td>

.. raw:: html

   <td>

This option is only used if both ‘test_bandwidth’ and ‘parallel’ keys
are true. This is a positive integer indicating size in Bytes of a data
block to be transferred continuously (“back-to-back”) for the duration
of one test pass. If the key is not present, ordinary transfers with
size indicated in ‘block_size’ key will be performed.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

link_type

.. raw:: html

   </td>

.. raw:: html

   <td>

Integer

.. raw:: html

   </td>

.. raw:: html

   <td>

This is a positive integer indicating type of link to be included in
bandwidth test. Numbering follows that listed in
**hsa_amd_link_info_type_t** in **hsa_ext_amd.h** file.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   </table>

Note that suitable values for **log_interval** and **duration** depend on your system.

-  **log_interval**, in sequential mode, should be long enough to allow
   all transfer tests to finish at lest once or “(pending)” and “(\*)”
   will be displayed (see below). Number of transfers depends on number
   of peer NUMA nodes in your system. In parallel mode, it should be
   roughly 1.5 times the duration of single longest individual test.
-  **duration**, regardless of mode should be at least, 4 \*
   log_interval.

You may obtain an indication of how long a single transfer between two NUMA nodes takes by running a test with the “-d 4” switch and observing DEBUG messages for transfer start/finish. An output may look like this:

::

   [DEBUG ] [187024.729433] [action_1] pebb transfer 0 6 start
   [DEBUG ] [187029.327818] [action_1] pebb transfer 0 6 finish
   [DEBUG ] [187024.299150] [action_1] pebb transfer 1 6 start
   [DEBUG ] [187029.473378] [action_1] pebb transfer 1 6 finish
   [DEBUG ] [187023.227009] [action_1] pebb transfer 1 5 start
   [DEBUG ] [187029.530203] [action_1] pebb transfer 1 5 finish
   [DEBUG ] [187025.737675] [action_1] pebb transfer 3 5 start
   [DEBUG ] [187030.134100] [action_1] pebb transfer 3 5 finish
   [DEBUG ] [187027.19961 ] [action_1] pebb transfer 2 6 start
   [DEBUG ] [187030.421181] [action_1] pebb transfer 2 6 finish
   [DEBUG ] [187027.41475 ] [action_1] pebb transfer 2 5 start
   [DEBUG ] [187031.293998] [action_1] pebb transfer 2 5 finish
   [DEBUG ] [187027.71717 ] [action_1] pebb transfer 0 5 start
   [DEBUG ] [187031.605326] [action_1] pebb transfer 0 5 finish

From this printout, it can be concluded that a single transfer takes, on average, 5500ms. Values for **log_interval** and **duration** should be set accordingly.

Output
~~~~~~

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

CPU node

.. raw:: html

   </td>

.. raw:: html

   <td>

Integer

.. raw:: html

   </td>

.. raw:: html

   <td>

Particular CPU node involved in the transfer

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

distance

.. raw:: html

   </td>

.. raw:: html

   <td>

Integer

.. raw:: html

   </td>

.. raw:: html

   <td>

NUMA distance for these two peers

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

hop_type

.. raw:: html

   </td>

.. raw:: html

   <td>

String

.. raw:: html

   </td>

.. raw:: html

   <td>

Link type for each link hop (e.g., PCIe, HyperTransport, QPI, …)

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

hop_distance

.. raw:: html

   </td>

.. raw:: html

   <td>

Integer

.. raw:: html

   </td>

.. raw:: html

   <td>

NUMA distance for this particular hop

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

transfer_id

.. raw:: html

   </td>

.. raw:: html

   <td>

String

.. raw:: html

   </td>

.. raw:: html

   <td>

String with format "<transfer_index>/<transfer_number>" where - transfer_index - is number, starting from
1, for each device-peer combination - transfer_number - is total number
of device-peer combinations

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

interval_bandwidth

.. raw:: html

   </td>

.. raw:: html

   <td>

Float

.. raw:: html

   </td>

.. raw:: html

   <td>

The average bandwidth of a p2p transfer, during the log_interval time
period.:raw-latex:`\n `This field may also take values: - (pending) -
this means that no measurement has taken place yet. - xxxGBps (\*) -
this means no measurement within current log_interval but average from
previous measurements is displayed.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

bandwidth

.. raw:: html

   </td>

.. raw:: html

   <td>

Float

.. raw:: html

   </td>

.. raw:: html

   <td>

The average bandwidth of a p2p transfer, averaged over the entire test
duration of the interval. This field may also take value: - (not
measured) - this means no test transfer completed for those peers. You
may need to increase test duration.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

duration

.. raw:: html

   </td>

.. raw:: html

   <td>

Float

.. raw:: html

   </td>

.. raw:: html

   <td>

Cumulative duration of all transfers between the two particular nodes

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   </table>

At the beginning, test will display link infor for every CPU/GPU pair:

::

   [RESULT][<timestamp>][<action name>] pcie-bandwidth [<transfer_id>] <cpu node> <gpu node> <gpu id> distance:<distance> <hop_type>:<hop_dist>[ <hop_type>:<hop_dist>]

During the execution of the benchmark, informational output providing
the moving average of the bandwidth of the transfer will be calculated
and logged. This interval is provided by the log_interval parameter and
will have the following output format:

::

   [INFO ][<timestamp>][<action name>] pcie-bandwidth [<transfer_id>] <cpu node> <gpu id> h2d: <host_to_device> d2h: <device_to_host> <interval_bandwidth>

At the end of test, the average bytes/second will be calculated over the
entire test duration, and will be logged as a result:

::

   [RESULT][<timestamp>][<action name>] pcie-bandwidth [<transfer_id>] <cpu node> <gpu id> h2d: <host_to_device> d2h: <device_to_host> <bandwidth> <duration>

Examples
~~~~~~~~

**Example 1:**

Consider action:

::

   actions:
   - name: action_1
     device: all
     module: pebb
     log_interval: 0
     duration: 0
     device_to_host: false
     host_to_device: true
     parallel: false

This will initiate host to device transfer to all GPUs with immediate
output (**parallel: false**, **log_interval: 0**):raw-latex:`\n` Output
from this action might look like:

::

   [RESULT] [1658774.978614] [action_1] pcie-bandwidth 0 4 3254  distance:36 HyperTransport:36
   [RESULT] [1658774.978664] [action_1] pcie-bandwidth 1 4 3254  distance:20 PCIe:20
   [RESULT] [1658774.978695] [action_1] pcie-bandwidth 2 4 3254  distance:36 HyperTransport:36
   [RESULT] [1658774.978728] [action_1] pcie-bandwidth 3 4 3254  distance:36 HyperTransport:36
   [RESULT] [1658774.978763] [action_1] pcie-bandwidth 0 5 50599  distance:36 HyperTransport:36
   [RESULT] [1658774.978795] [action_1] pcie-bandwidth 1 5 50599  distance:36 HyperTransport:36
   [RESULT] [1658774.978825] [action_1] pcie-bandwidth 2 5 50599  distance:20 PCIe:20
   [RESULT] [1658774.978856] [action_1] pcie-bandwidth 3 5 50599  distance:36 HyperTransport:36
   [RESULT] [1658774.978889] [action_1] pcie-bandwidth 0 6 33367  distance:36 HyperTransport:36
   [RESULT] [1658774.978922] [action_1] pcie-bandwidth 1 6 33367  distance:36 HyperTransport:36
   [RESULT] [1658774.978952] [action_1] pcie-bandwidth 2 6 33367  distance:36 HyperTransport:36
   [RESULT] [1658774.978982] [action_1] pcie-bandwidth 3 6 33367  distance:20 PCIe:20
   [INFO  ] [1658774.983743] [action_1] pcie-bandwidth  [1/12] 0 3254  h2d: true  d2h: false  12.233 GBps
   [INFO  ] [1658774.988272] [action_1] pcie-bandwidth  [2/12] 1 3254  h2d: true  d2h: false  12.227 GBps
   [INFO  ] [1658774.993197] [action_1] pcie-bandwidth  [3/12] 2 3254  h2d: true  d2h: false  11.770 GBps
   [INFO  ] [1658774.998105] [action_1] pcie-bandwidth  [4/12] 3 3254  h2d: true  d2h: false  11.313 GBps
   [INFO  ] [1658775.4457  ] [action_1] pcie-bandwidth  [5/12] 0 50599  h2d: true  d2h: false  12.218 GBps
   [INFO  ] [1658775.9589  ] [action_1] pcie-bandwidth  [6/12] 1 50599  h2d: true  d2h: false  10.292 GBps
   [INFO  ] [1658775.14627 ] [action_1] pcie-bandwidth  [7/12] 2 50599  h2d: true  d2h: false  10.456 GBps
   [INFO  ] [1658775.19664 ] [action_1] pcie-bandwidth  [8/12] 3 50599  h2d: true  d2h: false  10.614 GBps
   [INFO  ] [1658775.26210 ] [action_1] pcie-bandwidth  [9/12] 0 33367  h2d: true  d2h: false  12.222 GBps
   [INFO  ] [1658775.31188 ] [action_1] pcie-bandwidth  [10/12] 1 33367  h2d: true  d2h: false  12.215 GBps
   [INFO  ] [1658775.36137 ] [action_1] pcie-bandwidth  [11/12] 2 33367  h2d: true  d2h: false  12.219 GBps
   [INFO  ] [1658775.41117 ] [action_1] pcie-bandwidth  [12/12] 3 33367  h2d: true  d2h: false  12.219 GBps
   [RESULT] [1658775.42219 ] [action_1] pcie-bandwidth  [1/12] 0 3254  h2d: true  d2h: false  12.233 GBps  duration: 0.000780 sec
   [RESULT] [1658775.42235 ] [action_1] pcie-bandwidth  [2/12] 1 3254  h2d: true  d2h: false  12.227 GBps  duration: 0.000780 sec
   [RESULT] [1658775.42246 ] [action_1] pcie-bandwidth  [3/12] 2 3254  h2d: true  d2h: false  11.770 GBps  duration: 0.000810 sec
   [RESULT] [1658775.42256 ] [action_1] pcie-bandwidth  [4/12] 3 3254  h2d: true  d2h: false  11.313 GBps  duration: 0.000843 sec
   [RESULT] [1658775.42271 ] [action_1] pcie-bandwidth  [5/12] 0 50599  h2d: true  d2h: false  12.218 GBps  duration: 0.000781 sec
   [RESULT] [1658775.42286 ] [action_1] pcie-bandwidth  [6/12] 1 50599  h2d: true  d2h: false  10.292 GBps  duration: 0.000927 sec
   [RESULT] [1658775.42297 ] [action_1] pcie-bandwidth  [7/12] 2 50599  h2d: true  d2h: false  10.456 GBps  duration: 0.000912 sec
   [RESULT] [1658775.42309 ] [action_1] pcie-bandwidth  [8/12] 3 50599  h2d: true  d2h: false  10.614 GBps  duration: 0.000898 sec
   [RESULT] [1658775.42321 ] [action_1] pcie-bandwidth  [9/12] 0 33367  h2d: true  d2h: false  12.222 GBps  duration: 0.000780 sec
   [RESULT] [1658775.42332 ] [action_1] pcie-bandwidth  [10/12] 1 33367  h2d: true  d2h: false  12.215 GBps  duration: 0.000781 sec
   [RESULT] [1658775.42344 ] [action_1] pcie-bandwidth  [11/12] 2 33367  h2d: true  d2h: false  12.219 GBps  duration: 0.000780 sec
   [RESULT] [1658775.42355 ] [action_1] pcie-bandwidth  [12/12] 3 33367  h2d: true  d2h: false  12.219 GBps  duration: 0.000780 sec

**Example 2:**

Consider action:

::

   actions:
   - name: action_1
     device: all
     module: pebb
     log_interval: 500
     duration: 5000
     device_to_host: true
     host_to_device: true
     parallel: true

Here, although parallel execution of transfers is requested,
log_interval is to short for some transfers to complete. For them,
cumulative average is displayed and marked with (\*):

::

   [RESULT] [1659672.517170] [action_1] pcie-bandwidth 0 4 3254  distance:36 HyperTransport:36
   [RESULT] [1659672.517222] [action_1] pcie-bandwidth 1 4 3254  distance:20 PCIe:20
   [RESULT] [1659672.517257] [action_1] pcie-bandwidth 2 4 3254  distance:36 HyperTransport:36
   [RESULT] [1659672.517290] [action_1] pcie-bandwidth 3 4 3254  distance:36 HyperTransport:36
   [RESULT] [1659672.517324] [action_1] pcie-bandwidth 0 5 50599  distance:36 HyperTransport:36
   [RESULT] [1659672.517357] [action_1] pcie-bandwidth 1 5 50599  distance:36 HyperTransport:36
   [RESULT] [1659672.517388] [action_1] pcie-bandwidth 2 5 50599  distance:20 PCIe:20
   [RESULT] [1659672.517419] [action_1] pcie-bandwidth 3 5 50599  distance:36 HyperTransport:36
   [RESULT] [1659672.517452] [action_1] pcie-bandwidth 0 6 33367  distance:36 HyperTransport:36
   [RESULT] [1659672.517483] [action_1] pcie-bandwidth 1 6 33367  distance:36 HyperTransport:36
   [RESULT] [1659672.517515] [action_1] pcie-bandwidth 2 6 33367  distance:36 HyperTransport:36
   [RESULT] [1659672.517546] [action_1] pcie-bandwidth 3 6 33367  distance:20 PCIe:20
   [INFO  ] [1659673.49782 ] [action_1] pcie-bandwidth  [1/12] 0 3254  h2d: true  d2h: true  1.489 GBps
   [INFO  ] [1659673.49814 ] [action_1] pcie-bandwidth  [2/12] 1 3254  h2d: true  d2h: true  2.701 GBps
   ...
   [INFO  ] [1659673.582639] [action_1] pcie-bandwidth  [1/12] 0 3254  h2d: true  d2h: true  1.489 GBps (*)
   [INFO  ] [1659673.582686] [action_1] pcie-bandwidth  [2/12] 1 3254  h2d: true  d2h: true  16.367 GBps
   [INFO  ] [1659673.582700] [action_1] pcie-bandwidth  [3/12] 2 3254  h2d: true  d2h: true  17.300 GBps
   ...
   [INFO  ] [1659677.851697] [action_1] pcie-bandwidth  [1/12] 0 3254  h2d: true  d2h: true  16.793 GBps
   [INFO  ] [1659677.851727] [action_1] pcie-bandwidth  [2/12] 1 3254  h2d: true  d2h: true  16.872 GBps (*)
   [INFO  ] [1659677.851741] [action_1] pcie-bandwidth  [3/12] 2 3254  h2d: true  d2h: true  14.796 GBps (*)
   [INFO  ] [1659677.851754] [action_1] pcie-bandwidth  [4/12] 3 3254  h2d: true  d2h: true  20.358 GBps
   [INFO  ] [1659677.851770] [action_1] pcie-bandwidth  [5/12] 0 50599  h2d: true  d2h: true  15.632 GBps (*)
   [INFO  ] [1659677.851828] [action_1] pcie-bandwidth  [6/12] 1 50599  h2d: true  d2h: true  14.541 GBps (*)
   ...
   [RESULT] [1659678.148280] [action_1] pcie-bandwidth  [1/12] 0 3254  h2d: true  d2h: true  16.309 GBps  duration: 0.061316 sec
   [RESULT] [1659678.148318] [action_1] pcie-bandwidth  [2/12] 1 3254  h2d: true  d2h: true  16.871 GBps  duration: 0.118547 sec
   [RESULT] [1659678.148332] [action_1] pcie-bandwidth  [3/12] 2 3254  h2d: true  d2h: true  13.360 GBps  duration: 0.149705 sec
   [RESULT] [1659678.148349] [action_1] pcie-bandwidth  [4/12] 3 3254  h2d: true  d2h: true  15.371 GBps  duration: 0.130115 sec
   [RESULT] [1659678.148363] [action_1] pcie-bandwidth  [5/12] 0 50599  h2d: true  d2h: true  15.631 GBps  duration: 0.127954 sec
   [RESULT] [1659678.148377] [action_1] pcie-bandwidth  [6/12] 1 50599  h2d: true  d2h: true  14.185 GBps  duration: 0.140989 sec
   [RESULT] [1659678.148390] [action_1] pcie-bandwidth  [7/12] 2 50599  h2d: true  d2h: true  15.242 GBps  duration: 0.131245 sec
   [RESULT] [1659678.148404] [action_1] pcie-bandwidth  [8/12] 3 50599  h2d: true  d2h: true  16.071 GBps  duration: 0.124452 sec
   [RESULT] [1659678.148418] [action_1] pcie-bandwidth  [9/12] 0 33367  h2d: true  d2h: true  16.505 GBps  duration: 0.121178 sec
   [RESULT] [1659678.148432] [action_1] pcie-bandwidth  [10/12] 1 33367  h2d: true  d2h: true  16.720 GBps  duration: 0.059807 sec
   [RESULT] [1659678.148445] [action_1] pcie-bandwidth  [11/12] 2 33367  h2d: true  d2h: true  15.604 GBps  duration: 0.128168 sec
   [RESULT] [1659678.148458] [action_1] pcie-bandwidth  [12/12] 3 33367  h2d: true  d2h: true  16.193 GBps  duration: 0.123525 sec

Please note that in link information results, some records could be marked with (R). This means, that communication is possible if initiated by the destination NUMA node HSA agent.
