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
