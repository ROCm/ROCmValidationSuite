.. meta::
  :description: rocm validation suite documentation 
  :keywords: rocm validation suite, ROCm, documentation

.. _pbqt-module:



PBQT Module
-----------

The P2P Qualification Tool is designed to provide the list of all GPUs
that support P2P and characterize the P2P links between peers. In
addition to testing for P2P compatibility, this test will perform a
peer-to-peer throughput test between all unique P2P pairs for
performance evaluation. These are known as device-to-device transfers,
and can be either uni-directional or bi-directional. The average
bandwidth obtained is reported to help debug low bandwidth issues.

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

peers

.. raw:: html

   </td>

.. raw:: html

   <td>

Collection of Strings

.. raw:: html

   </td>

.. raw:: html

   <td>

This is a required key, and specifies the set of GPU(s) considered being
peers of the GPU specified in the action. If ‘all’ is specified, all
other GPU(s) on the system will be considered peers. Otherwise only the
GPU ids specified in the list will be considered.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

peer_deviceid

.. raw:: html

   </td>

.. raw:: html

   <td>

Integer

.. raw:: html

   </td>

.. raw:: html

   <td>

This is an optional parameter, but if specified it restricts the peers
list to a specific device type corresponding to the deviceid.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

test_bandwidth

.. raw:: html

   </td>

.. raw:: html

   <td>

Bool

.. raw:: html

   </td>

.. raw:: html

   <td>

If this key is set to true the P2P bandwidth benchmark will run if a
pair of devices pass the P2P check.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

bidirectional

.. raw:: html

   </td>

.. raw:: html

   <td>

Bool

.. raw:: html

   </td>

.. raw:: html

   <td>

This option is only used if test_bandwidth key is true. This specifies
the type of transfer to run::raw-latex:`\n` - true – Do a bidirectional
transfer test:raw-latex:`\n` - false – Do a unidirectional transfer test
from one node to another.

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

Please note that suitable values for **log_interval** and **duration**
depend on your system.

-  **log_interval**, in sequential mode, should be long enough to allow
   all transfer tests to finish at lest once or “(pending)” and “(\*)”
   will be displayed (see below). Number of transfers depends on number
   of peer NUMA nodes in your system. In parallel mode, it should be
   roughly 1.5 times the duration of a single longest individual test.
-  **duration**, regardless of mode, should be at least 4 \*
   log_interval.

You may obtain an indication of how long a single transfer between two NUMA
nodes takes by running a test with “-d 4” switch and observing DEBUG
messages for transfer start/finish. An output may look like this:

::

   [DEBUG ] [183940.634118] [action_1] pbqt transfer 6 5 start
   [DEBUG ] [183941.311671] [action_1] pbqt transfer 6 5 finish
   [DEBUG ] [183941.312746] [action_1] pbqt transfer 4 5 start
   [DEBUG ] [183941.990174] [action_1] pbqt transfer 4 5 finish
   [DEBUG ] [183941.991244] [action_1] pbqt transfer 4 6 start
   [DEBUG ] [183942.668687] [action_1] pbqt transfer 4 6 finish
   [DEBUG ] [183942.669756] [action_1] pbqt transfer 5 4 start
   [DEBUG ] [183943.340957] [action_1] pbqt transfer 5 4 finish
   [DEBUG ] [183943.342037] [action_1] pbqt transfer 5 6 start
   [DEBUG ] [183944.17957 ] [action_1] pbqt transfer 5 6 finish
   [DEBUG ] [183944.19032 ] [action_1] pbqt transfer 6 4 start
   [DEBUG ] [183944.700868] [action_1] pbqt transfer 6 4 finish

From this printout, it can be concluded that a single transfer takes an average of 800ms. Values for **log_interval** and **duration** should be set accordingly.

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

p2p_result

.. raw:: html

   </td>

.. raw:: html

   <td>

Bool

.. raw:: html

   </td>

.. raw:: html

   <td>

Indicates if the gpu and the specified peer have P2P capabilities. If
this quantity is true, the GPU pair tested has p2p capabilities. If
false, they are not peers.

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
1, for each device-peer combination - transfer_number - is the total number of device-peer combinations

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

If the value of test_bandwidth key is false, the tool will only try to
determine if the GPU(s) in the peers key are P2P to the action’s GPU. In
this case the bidirectional and log_interval values will be ignored, if
they are specified. If a gpu is a P2P peer to the device the test will
pass, otherwise it will fail. A message indicating the result will be
provided for each GPUs specified. It will have the following format:

::

   [RESULT][<timestamp>][<action name>] p2p <gpu id> <peer gpu id> peers:<p2p_result> distance:<distance> <hop_type>:<hop_dist>[ <hop_type>:<hop_dist>]

If the value of test_bandwidth is true bandwidth testing between the
device and each of its peers will take place in parallel or in sequence,
depending on the value of the parallel flag. During the duration of
bandwidth benchmarking, informational output providing the moving
average of the transfer’s bandwidth will be calculated and logged at
every time increment specified by the log_interval parameter. The
messages will have the following output:

::

   [INFO  ][<timestamp>][<action name>] p2p-bandwidth [<transfer_id>] <gpu id> <peer gpu id> bidirectional: <bidirectional> <interval_bandwidth>

At the end of the test the average bytes/second will be calculated over
the entire test duration, and will be logged as a result:

::

   [RESULT][<timestamp>][<action name>] p2p-bandwidth [<transfer_id>] <gpu id> <peer gpu id> bidirectional: <bidirectional> <bandwidth> <duration>

Examples
~~~~~~~~

**Example 1:**

Here all source GPUs (device: all) with all destination GPUs (peers:
all) are tested for p2p capability with no bandwidth testing
(test_bandwidth: false).

::

   actions:
   - name: action_1
     device: all
     module: pbqt
     peers: all
     test_bandwidth: false

Possible result is:

::

   [RESULT] [1656631.262875] [action_1] p2p 3254 3254 peers:false distance:-1
   [RESULT] [1656631.262968] [action_1] p2p 3254 50599 peers:true distance:56 HyperTransport:56
   [RESULT] [1656631.263039] [action_1] p2p 3254 33367 peers:true distance:56 HyperTransport:56
   [RESULT] [1656631.263103] [action_1] p2p 50599 3254 peers:true distance:56 HyperTransport:56
   [RESULT] [1656631.263151] [action_1] p2p 50599 50599 peers:false distance:-1
   [RESULT] [1656631.263203] [action_1] p2p 50599 33367 peers:true distance:56 HyperTransport:56
   [RESULT] [1656631.263265] [action_1] p2p 33367 3254 peers:true distance:56 HyperTransport:56
   [RESULT] [1656631.263321] [action_1] p2p 33367 50599 peers:true distance:56 HyperTransport:56
   [RESULT] [1656631.263360] [action_1] p2p 33367 33367 peers:false distance:-1

From the first line of result, we can see that GPU (ID 3254) can’t
access itself. From the second line of result, we can see that source
GPU (ID 3254) can access destination GPU (ID 50599).

**Example 2:**

Here all source GPUs (device: all) with all destination GPUs (peers:
all) are tested for p2p capability including bandwidth testing
(test_bandwidth: true) with bidirectional transfers (bidirectional:
true) and with emmediate output for each completed transfer
(log_interval: 0)

::

   actions:
   - name: action_1
     device: all
     module: pbqt
     log_interval: 0
     duration: 0
     peers: all
     test_bandwidth: true
     bidirectional: true

When run with “-d 3” switch, possible result is:

::

   [RESULT] [1657122.364752] [action_1] p2p 3254 3254 peers:false distance:-1
   [RESULT] [1657122.364845] [action_1] p2p 3254 50599 peers:true distance:56 HyperTransport:56
   [RESULT] [1657122.364917] [action_1] p2p 3254 33367 peers:true distance:56 HyperTransport:56
   [RESULT] [1657122.364985] [action_1] p2p 50599 3254 peers:true distance:56 HyperTransport:56
   [RESULT] [1657122.365037] [action_1] p2p 50599 50599 peers:false distance:-1
   [RESULT] [1657122.365094] [action_1] p2p 50599 33367 peers:true distance:56 HyperTransport:56
   [RESULT] [1657122.365157] [action_1] p2p 33367 3254 peers:true distance:56 HyperTransport:56
   [RESULT] [1657122.365221] [action_1] p2p 33367 50599 peers:true distance:56 HyperTransport:56
   [RESULT] [1657122.365270] [action_1] p2p 33367 33367 peers:false distance:-1
   [INFO  ] [1657123.644203] [action_1] p2p-bandwidth  [1/6] 3254 50599  bidirectional: true  7.013 GBps
   [INFO  ] [1657123.644376] [action_1] p2p-bandwidth  [2/6] 3254 33367  bidirectional: true  6.615 GBps
   [INFO  ] [1657123.644453] [action_1] p2p-bandwidth  [3/6] 50599 3254  bidirectional: true  2.367 GBps
   [INFO  ] [1657123.644522] [action_1] p2p-bandwidth  [4/6] 50599 33367  bidirectional: true  7.504 GBps
   [INFO  ] [1657123.644590] [action_1] p2p-bandwidth  [5/6] 33367 3254  bidirectional: true  8.207 GBps
   [INFO  ] [1657123.644673] [action_1] p2p-bandwidth  [6/6] 33367 50599  bidirectional: true  7.680 GBps
   [INFO  ] [1657124.926221] [action_1] p2p-bandwidth  [1/6] 3254 50599  bidirectional: true  6.646 GBps
   [INFO  ] [1657124.926368] [action_1] p2p-bandwidth  [2/6] 3254 33367  bidirectional: true  8.418 GBps
   [INFO  ] [1657124.926438] [action_1] p2p-bandwidth  [3/6] 50599 3254  bidirectional: true  7.402 GBps
   [INFO  ] [1657124.926506] [action_1] p2p-bandwidth  [4/6] 50599 33367  bidirectional: true  6.161 GBps
   [INFO  ] [1657124.926573] [action_1] p2p-bandwidth  [5/6] 33367 3254  bidirectional: true  9.024 GBps
   [INFO  ] [1657124.926640] [action_1] p2p-bandwidth  [6/6] 33367 50599  bidirectional: true  8.740 GBps
   [INFO  ] [1657126.208742] [action_1] p2p-bandwidth  [1/6] 3254 50599  bidirectional: true  5.680 GBps
   [INFO  ] [1657126.208905] [action_1] p2p-bandwidth  [2/6] 3254 33367  bidirectional: true  8.011 GBps
   [INFO  ] [1657126.208990] [action_1] p2p-bandwidth  [3/6] 50599 3254  bidirectional: true  3.918 GBps
   [INFO  ] [1657126.209066] [action_1] p2p-bandwidth  [4/6] 50599 33367  bidirectional: true  6.058 GBps
   [INFO  ] [1657126.209140] [action_1] p2p-bandwidth  [5/6] 33367 3254  bidirectional: true  6.650 GBps
   [INFO  ] [1657126.209213] [action_1] p2p-bandwidth  [6/6] 33367 50599  bidirectional: true  0.000 GBps
   [RESULT] [1657126.742128] [action_1] p2p-bandwidth  [1/6] 3254 50599  bidirectional: true  5.767 GBps  duration: 0.368453 sec
   [RESULT] [1657126.743287] [action_1] p2p-bandwidth  [2/6] 3254 33367  bidirectional: true  6.013 GBps  duration: 0.498944 sec
   [RESULT] [1657126.744411] [action_1] p2p-bandwidth  [3/6] 50599 3254  bidirectional: true  5.278 GBps  duration: 0.380393 sec
   [RESULT] [1657126.745534] [action_1] p2p-bandwidth  [4/6] 50599 33367  bidirectional: true  4.160 GBps  duration: 0.484577 sec
   [RESULT] [1657126.746684] [action_1] p2p-bandwidth  [5/6] 33367 3254  bidirectional: true  5.219 GBps  duration: 0.407190 sec
   [RESULT] [1657126.747827] [action_1] p2p-bandwidth  [6/6] 33367 50599  bidirectional: true  4.001 GBps  duration: 0.562350 sec

We can see that on this particular machine there are three GPUs and six
possible device-to-peer transfers.

**Example 3:**

Here some source GPUs (device: 50599) are targeting some destination
GPUs (peers: 33367 3254) with specified log interval (log_interval:
1000) and duration (duration: 5000). Bandwidth is tested
(test_bandwidth: true) but only unidirectional (bidirectional: false)
without parallel execution (parallel: false).

::

   actions:
   - name: action_1
     device: 50599
     module: pbqt
     log_interval: 1000
     duration: 5000
     count: 0
     peers: 33367 3254
     test_bandwidth: true
     bidirectional: false
     parallel: false

Possible output is:

::

   [RESULT] [1657218.801555] [action_1] p2p 50599 3254 peers:true distance:56 HyperTransport:56
   [RESULT] [1657218.801655] [action_1] p2p 50599 33367 peers:true distance:56 HyperTransport:56
   [INFO  ] [1657219.871532] [action_1] p2p-bandwidth  [1/2] 50599 3254  bidirectional: false  4.517 GBps
   [INFO  ] [1657219.871717] [action_1] p2p-bandwidth  [2/2] 50599 33367  bidirectional: false  4.475 GBps
   [INFO  ] [1657220.940263] [action_1] p2p-bandwidth  [1/2] 50599 3254  bidirectional: false  4.476 GBps
   [INFO  ] [1657220.940461] [action_1] p2p-bandwidth  [2/2] 50599 33367  bidirectional: false  4.601 GBps
   [INFO  ] [1657222.7589  ] [action_1] p2p-bandwidth  [1/2] 50599 3254  bidirectional: false  4.488 GBps
   [INFO  ] [1657222.7760  ] [action_1] p2p-bandwidth  [2/2] 50599 33367  bidirectional: false  4.470 GBps
   [INFO  ] [1657223.74647 ] [action_1] p2p-bandwidth  [1/2] 50599 3254  bidirectional: false  4.666 GBps
   [INFO  ] [1657223.74810 ] [action_1] p2p-bandwidth  [2/2] 50599 33367  bidirectional: false  4.576 GBps
   [RESULT] [1657224.181106] [action_1] p2p-bandwidth  [1/2] 50599 3254  bidirectional: false  4.539 GBps  duration: 1.321909 sec
   [RESULT] [1657224.182255] [action_1] p2p-bandwidth  [2/2] 50599 33367  bidirectional: false  4.551 GBps  duration: 1.318517 sec

From the last line of result, we can see that source GPU (ID 50599) can
access destination GPU (ID 33367) and that the bandwidth is 4.495 GBps.

**Example 4:**

Here, all GPUs are targeted with bidirectional transfers and parallel
execution of tests:

::

   actions:
   - name: action_1
     device: all
     module: pbqt
     log_interval: 1200
     duration: 4000
     peers: all
     test_bandwidth: true
     bidirectional: true
     parallel: true

Possible output is:

::

   [RESULT] [1657295.937184] [action_1] p2p 3254 3254 peers:false distance:-1
   [RESULT] [1657295.937267] [action_1] p2p 3254 50599 peers:true distance:56 HyperTransport:56
   [RESULT] [1657295.937324] [action_1] p2p 3254 33367 peers:true distance:56 HyperTransport:56
   [RESULT] [1657295.937379] [action_1] p2p 50599 3254 peers:true distance:56 HyperTransport:56
   [RESULT] [1657295.937429] [action_1] p2p 50599 50599 peers:false distance:-1
   [RESULT] [1657295.937482] [action_1] p2p 50599 33367 peers:true distance:56 HyperTransport:56
   [RESULT] [1657295.937543] [action_1] p2p 33367 3254 peers:true distance:56 HyperTransport:56
   [RESULT] [1657295.937607] [action_1] p2p 33367 50599 peers:true distance:56 HyperTransport:56
   [RESULT] [1657295.937655] [action_1] p2p 33367 33367 peers:false distance:-1
   [INFO  ] [1657297.216212] [action_1] p2p-bandwidth  [1/6] 3254 50599  bidirectional: true  4.972 GBps
   [INFO  ] [1657297.216351] [action_1] p2p-bandwidth  [2/6] 3254 33367  bidirectional: true  8.183 GBps
   [INFO  ] [1657297.216423] [action_1] p2p-bandwidth  [3/6] 50599 3254  bidirectional: true  8.911 GBps
   [INFO  ] [1657297.216490] [action_1] p2p-bandwidth  [4/6] 50599 33367  bidirectional: true  7.690 GBps
   [INFO  ] [1657297.216558] [action_1] p2p-bandwidth  [5/6] 33367 3254  bidirectional: true  7.768 GBps
   [INFO  ] [1657297.216642] [action_1] p2p-bandwidth  [6/6] 33367 50599  bidirectional: true  4.589 GBps
   [INFO  ] [1657298.487427] [action_1] p2p-bandwidth  [1/6] 3254 50599  bidirectional: true  8.778 GBps
   [INFO  ] [1657298.487593] [action_1] p2p-bandwidth  [2/6] 3254 33367  bidirectional: true  7.921 GBps
   [INFO  ] [1657298.487730] [action_1] p2p-bandwidth  [3/6] 50599 3254  bidirectional: true  8.164 GBps
   [INFO  ] [1657298.487807] [action_1] p2p-bandwidth  [4/6] 50599 33367  bidirectional: true  8.921 GBps
   [INFO  ] [1657298.487878] [action_1] p2p-bandwidth  [5/6] 33367 3254  bidirectional: true  8.487 GBps
   [INFO  ] [1657298.487956] [action_1] p2p-bandwidth  [6/6] 33367 50599  bidirectional: true  7.648 GBps
   [INFO  ] [1657299.760175] [action_1] p2p-bandwidth  [1/6] 3254 50599  bidirectional: true  7.210 GBps
   [INFO  ] [1657299.760249] [action_1] p2p-bandwidth  [2/6] 3254 33367  bidirectional: true  4.274 GBps
   [INFO  ] [1657299.760284] [action_1] p2p-bandwidth  [3/6] 50599 3254  bidirectional: true  0.000 GBps
   [INFO  ] [1657299.760318] [action_1] p2p-bandwidth  [4/6] 50599 33367  bidirectional: true  5.942 GBps
   [INFO  ] [1657299.760349] [action_1] p2p-bandwidth  [5/6] 33367 3254  bidirectional: true  0.001 GBps
   [INFO  ] [1657299.760381] [action_1] p2p-bandwidth  [6/6] 33367 50599  bidirectional: true  5.490 GBps
   [RESULT] [1657300.293126] [action_1] p2p-bandwidth  [1/6] 3254 50599  bidirectional: true  6.964 GBps  duration: 0.287248 sec
   [RESULT] [1657300.294334] [action_1] p2p-bandwidth  [2/6] 3254 33367  bidirectional: true  3.960 GBps  duration: 0.536554 sec
   [RESULT] [1657300.295528] [action_1] p2p-bandwidth  [3/6] 50599 3254  bidirectional: true  5.442 GBps  duration: 0.368977 sec
   [RESULT] [1657300.296691] [action_1] p2p-bandwidth  [4/6] 50599 33367  bidirectional: true  4.187 GBps  duration: 0.477756 sec
   [RESULT] [1657300.297840] [action_1] p2p-bandwidth  [5/6] 33367 3254  bidirectional: true  4.942 GBps  duration: 0.607009 sec
   [RESULT] [1657300.299016] [action_1] p2p-bandwidth  [6/6] 33367 50599  bidirectional: true  3.828 GBps  duration: 0.523495 sec

It can be seen that transfers [2/6] and [5/6] did not take place in the second log interval so average from the previous cycle is displayed instead and marked with “(\*)”
