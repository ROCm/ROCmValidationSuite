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

Please note that suitable values for **log_interval** and **duration**
depend on your system.

-  **log_interval**, in sequential mode, should be long enough to allow
   all transfer tests to finish at lest once or “(pending)” and “(\*)”
   will be displayed (see below). Number of transfers depends on number
   of peer NUMA nodes in your system. In parallel mode, it should be
   roughly 1.5 times the duration of single longest individual test.
-  **duration**, regardless of mode should be at least, 4 \*
   log_interval.

You may obtain indication of how long single transfer between two NUMA
nodes take by running test with “-d 4” switch and observing DEBUG
messages for transfer start/finish. An output may look like this:

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

From this printout, it can be concluded that single transfer takes on
average 5500ms. Values for **log_interval** and **duration** should be
set accordingly.
