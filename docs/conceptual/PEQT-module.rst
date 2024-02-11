.. meta::
  :description: rocm validation suite documentation 
  :keywords: rocm validation suite, ROCm, documentation

.. _peqt-module:


PEQT Module
-----------

PCI Express Qualification Tool module targets and qualifies the
configuration of the platforms PCIe connections to the GPUs. The purpose
of the PEQT module is to provide an extensible, OS independent and
scriptable interface capable of performing the PCIe interconnect
configuration checks required for ROCm support of GPUs. This information
can be obtained through the sysfs PCIe interface or by using the PCIe
development libraries to extract values from various PCIe control,
status and capabilities registers. These registers are specified in the
PCI Express Base Specification, Revision 3. Iteration keys, i.e. count,
wait and duration will be ignored for actions using the PEQT module.

Module Specific Keys
~~~~~~~~~~~~~~~~~~~~

Module specific output keys are described in the table below:

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

capability

.. raw:: html

   </td>

.. raw:: html

   <td>

Collection of Structures with the following
format::raw-latex:`\n{String,String}`

.. raw:: html

   </td>

.. raw:: html

   <td>

The PCIe capability key contains a collection of structures that specify
which PCIe capability to check and the expected value of the capability.
A check structure must contain the PCIe capability value, but an
expected value may be omitted. The value of all valid capabilities that
are a part of this collection will be entered into the capability_value
field. Possible capabilities, and their value types
are::raw-latex:`\n`:raw-latex:`\n` link_cap_max_speed:raw-latex:`\n`
link_cap_max_width:raw-latex:`\n` link_stat_cur_speed:raw-latex:`\n`
link_stat_neg_width:raw-latex:`\n` slot_pwr_limit_value:raw-latex:`\n`
slot_physical_num:raw-latex:`\n` bus_id:raw-latex:`\n`
atomic_op_32_completer:raw-latex:`\n`
atomic_op_64_completer:raw-latex:`\n`
atomic_op_128_CAS_completer:raw-latex:`\n`
atomic_op_routing:raw-latex:`\n` dev_serial_num:raw-latex:`\n`
kernel_driver:raw-latex:`\n` pwr_base_pwr:raw-latex:`\n`
pwr_rail_type:raw-latex:`\n` device_id:raw-latex:`\n`
vendor_id:raw-latex:`\n`:raw-latex:`\n`

The expected value String is a regular expression that is used to check
the actual value of the capability.

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

capability_value

.. raw:: html

   </td>

.. raw:: html

   <td>

Collection of Strings

.. raw:: html

   </td>

.. raw:: html

   <td>

For each of the capabilities specified in the capability key, the actual
value of the capability will be returned, represented as a String.

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

String

.. raw:: html

   </td>

.. raw:: html

   <td>

‘true’ if all of the properties match the values given, ‘false’
otherwise.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   </table>

The qualification check queries the specified PCIe capabilities and
properties and checks that their actual values satisfy the regular
expression provided in the ‘expected value’ field for that capability.
The pass output key will be true and the test will pass if all of the
properties match the values given. After the check is finished, the
following informational messages will be generated:

::

   [INFO  ][<timestamp>][<action name>] peqt <capability> <capability_value>
   [RESULT][<timestamp>][<action name>] peqt <pass>

For details regarding each of the capabilities and current values
consult the chapters in the PCI Express Base Specification, Revision 3.

Examples
~~~~~~~~

**Example 1:**

A regular PEQT configuration file looks like this:

::

   actions:
   - name: pcie_act_1
     module: peqt
     capability:
       link_cap_max_speed:
       link_cap_max_width:
       link_stat_cur_speed:
       link_stat_neg_width:
       slot_pwr_limit_value:
       slot_physical_num:
       device_id:
       vendor_id:
       kernel_driver:
       dev_serial_num:
       D0_Maximum_Power_12V:
       D0_Maximum_Power_3_3V:
       D0_Sustained_Power_12V:
       D0_Sustained_Power_3_3V:
       atomic_op_routing:
       atomic_op_32_completer:
       atomic_op_64_completer:
       atomic_op_128_CAS_completer:
     device: all

Please note: - when setting the ‘device’ configuration key to ‘all’, the
RVS will detect all the AMD compatible GPUs and run the test on all of
them

-  there are no regular expression for this .conf file, therefore RVS
   will report TRUE if at least one AMD compatible GPU is registered
   within the system. Otherwise it will report FALSE.

Please note that the Power Budgeting capability is a dynamic one, having
the following form:

::

   <PM_State>_<Type>_<Power rail>

where:

::

   PM_State = D0/D1/D2/D3
   Type=PMEAux/Auxiliary/Idle/Sustained/Maximum
   PowerRail = Power_12V/Power_3_3V/Power_1_5V_1_8V/Thermal

When the RVS tool runs against such a configuration file, it will query
for the all the PCIe capabilities specified under the capability list
(and log the corresponding values) for all the AMD compatible GPUs. For
those PCIe capabilities that are not supported by the HW platform were
the RVS is running, a “NOT SUPPORTED” message will be logged.

The output for such a configuration file may look like this:

::

   [INFO ] [177628.401176] pcie_act_1 peqt D0_Maximum_Power_12V NOT SUPPORTED
   [INFO ] [177628.401229] pcie_act_1 peqt D0_Maximum_Power_3_3V NOT SUPPORTED
   [INFO ] [177628.401248] pcie_act_1 peqt D0_Sustained_Power_12V NOT SUPPORTED
   [INFO ] [177628.401269] pcie_act_1 peqt D0_Sustained_Power_3_3V NOT SUPPORTED
   [INFO ] [177628.401282] pcie_act_1 peqt atomic_op_128_CAS_completer FALSE
   [INFO ] [177628.401291] pcie_act_1 peqt atomic_op_32_completer FALSE
   [INFO ] [177628.401303] pcie_act_1 peqt atomic_op_64_completer FALSE
   [INFO ] [177628.401311] pcie_act_1 peqt atomic_op_routing TRUE
   [INFO ] [177628.401317] pcie_act_1 peqt dev_serial_num NOT SUPPORTED
   [INFO ] [177628.401323] pcie_act_1 peqt device_id 26720
   [INFO ] [177628.401334] pcie_act_1 peqt kernel_driver amdgpu
   [INFO ] [177628.401342] pcie_act_1 peqt link_cap_max_speed 8 GT/s
   [INFO ] [177628.401352] pcie_act_1 peqt link_cap_max_width x16
   [INFO ] [177628.401359] pcie_act_1 peqt link_stat_cur_speed 8 GT/s
   [INFO ] [177628.401367] pcie_act_1 peqt link_stat_neg_width x16
   [INFO ] [177628.401375] pcie_act_1 peqt slot_physical_num #0
   [INFO ] [177628.401396] pcie_act_1 peqt slot_pwr_limit_value 0.000W
   [INFO ] [177628.401402] pcie_act_1 peqt vendor_id 4098
   [INFO ] [177628.401656] pcie_act_1 peqt D0_Maximum_Power_12V NOT SUPPORTED
   [INFO ] [177628.401675] pcie_act_1 peqt D0_Maximum_Power_3_3V NOT SUPPORTED
   [INFO ] [177628.401692] pcie_act_1 peqt D0_Sustained_Power_12V NOT SUPPORTED
   [INFO ] [177628.401709] pcie_act_1 peqt D0_Sustained_Power_3_3V NOT SUPPORTED
   [INFO ] [177628.401719] pcie_act_1 peqt atomic_op_128_CAS_completer FALSE
   [INFO ] [177628.401728] pcie_act_1 peqt atomic_op_32_completer FALSE
   [INFO ] [177628.401736] pcie_act_1 peqt atomic_op_64_completer FALSE
   [INFO ] [177628.401745] pcie_act_1 peqt atomic_op_routing TRUE
   [INFO ] [177628.401750] pcie_act_1 peqt dev_serial_num NOT SUPPORTED
   [INFO ] [177628.401757] pcie_act_1 peqt device_id 26720
   [INFO ] [177628.401771] pcie_act_1 peqt kernel_driver amdgpu
   [INFO ] [177628.401781] pcie_act_1 peqt link_cap_max_speed 8 GT/s
   [INFO ] [177628.401788] pcie_act_1 peqt link_cap_max_width x16
   [INFO ] [177628.401794] pcie_act_1 peqt link_stat_cur_speed 8 GT/s
   [INFO ] [177628.401800] pcie_act_1 peqt link_stat_neg_width x16
   [INFO ] [177628.401806] pcie_act_1 peqt slot_physical_num #0
   [INFO ] [177628.401814] pcie_act_1 peqt slot_pwr_limit_value 0.000W
   [INFO ] [177628.401819] pcie_act_1 peqt vendor_id 4098
   [RESULT] [177628.403781] pcie_act_1 peqt TRUE

**Example 2:**

Another example of a configuration file, which queries for a smaller
subset of PCIe capabilities but adds regular expressions check, is given
below

::

   actions:
   - name: pcie_act_1
     module: peqt
     capability:
       link_cap_max_speed: '^(2\.5 GT\/s|5 GT\/s|8 GT\/s)$'
       link_cap_max_width:
       link_stat_cur_speed: '^(2\.5 GT\/s|5 GT\/s|8 GT\/s)$'
       link_stat_neg_width:
       slot_pwr_limit_value: '[a-b][d-'
       slot_physical_num:
       device_id:
       vendor_id:
       kernel_driver:
     device: all

For this example, the expected PEQT check result is TRUE if:

-  at least one AMD compatible GPU is registered within the system and:
-  all <link_cap_max_speed> values for all AMD compatible GPUs match the
   given regular expression and
-  all <link_stat_cur_speed> values for all AMD compatible GPUs match
   the given regular expression

Please note that the <slot_pwr_limit_value> regular expression is not
valid and will be skipped without affecting the PEQT module’s check
RESULT (however, an error will be logged out)

**Example 3:**

Another example with even more regular expressions is given below. The
expected PEQT check result is TRUE if at least one AMD compatible GPU
having the ID 3254 or 33367 is registered within the system and all the
PCIe capabilities values match their corresponding regular expressions.

::

   actions:
   - name: pcie_act_1
     module: peqt
     deviceid: 26720
     capability:
       link_cap_max_speed: '^(2\.5 GT\/s|5 GT\/s|8 GT\/s)$'
       link_cap_max_width: ^(x8|x16)$
       link_stat_cur_speed: '^(8 GT\/s)$'
       link_stat_neg_width: ^(x8|x16)$
       kernel_driver: ^amdgpu$
       atomic_op_routing: ^((TRUE|FALSE){1})$
       atomic_op_32_completer: ^((TRUE|FALSE){1})$
       atomic_op_64_completer: ^((TRUE|FALSE){1})$
       atomic_op_128_CAS_completer: ^((TRUE|FALSE){1})$
     device: 3254 33367

SMQT Module
-----------

The GPU SBIOS mapping qualification tool is designed to verify that a
platform’s SBIOS has satisfied the BAR mapping requirements for VDI and
Radeon Instinct products for ROCm support. These are the current BAR
requirements::raw-latex:`\n`:raw-latex:`\n`

BAR 1: GPU Frame Buffer BAR – In this example it happens to be 256M, but
typically this will be size of the GPU memory (typically 4GB+). This BAR
has to be placed < 2^40 to allow peer- to-peer access from other GFX8
AMD GPUs. For GFX9 (Vega GPU) the BAR has to be placed < 2^44 to allow
peer-to-peer access from other GFX9 AMD
GPUs.:raw-latex:`\n`:raw-latex:`\n`

BAR 2: Doorbell BAR – The size of the BAR is typically will be < 10MB
(currently fixed at 2MB) for this generation GPUs. This BAR has to be
placed < 2^40 to allow peer-to-peer access from other current generation
AMD GPUs.:raw-latex:`\n`:raw-latex:`\n` BAR 3: IO BAR - This is for
legacy VGA and boot device support, but since this the GPUs in this
project are not VGA devices (headless), this is not a concern even if
the SBIOS does not setup.:raw-latex:`\n`:raw-latex:`\n`

BAR 4: MMIO BAR – This is required for the AMD Driver SW to access the
configuration registers. Since the reminder of the BAR available is only
1 DWORD (32bit), this is placed < 4GB. This is fixed at
256KB.:raw-latex:`\n`:raw-latex:`\n`

BAR 5: Expansion ROM – This is required for the AMD Driver SW to access
the GPU’s video-BIOS. This is currently fixed at
128KB.:raw-latex:`\n`:raw-latex:`\n`

Refer to the ROCm Use of Advanced PCIe Features and Overview of How BAR
Memory is Used In ROCm Enabled System web page for more information
about how BAR memory is initialized by VDI and Radeon products.
Iteration keys, i.e. count, wait and duration will be ignored.

.. _module-specific-keys-1:

Module Specific Keys
~~~~~~~~~~~~~~~~~~~~

Module specific output keys are described in the table below:

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

bar1_req_size

.. raw:: html

   </td>

.. raw:: html

   <td>

Integer

.. raw:: html

   </td>

.. raw:: html

   <td>

This is an integer specifying the required size of the BAR1 frame buffer
region.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

bar1_base_addr_min

.. raw:: html

   </td>

.. raw:: html

   <td>

Integer

.. raw:: html

   </td>

.. raw:: html

   <td>

This is an integer specifying the minimum value the BAR1 base address
can be.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

bar1_base_addr_max

.. raw:: html

   </td>

.. raw:: html

   <td>

Integer

.. raw:: html

   </td>

.. raw:: html

   <td>

This is an integer specifying the maximum value the BAR1 base address
can be.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

bar2_req_size

.. raw:: html

   </td>

.. raw:: html

   <td>

Integer

.. raw:: html

   </td>

.. raw:: html

   <td>

This is an integer specifying the required size of the BAR2 frame buffer
region.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

bar2_base_addr_min

.. raw:: html

   </td>

.. raw:: html

   <td>

Integer

.. raw:: html

   </td>

.. raw:: html

   <td>

This is an integer specifying the minimum value the BAR2 base address
can be.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

bar2_base_addr_max

.. raw:: html

   </td>

.. raw:: html

   <td>

Integer

.. raw:: html

   </td>

.. raw:: html

   <td>

This is an integer specifying the maximum value the BAR2 base address
can be.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

bar4_req_size

.. raw:: html

   </td>

.. raw:: html

   <td>

Integer

.. raw:: html

   </td>

.. raw:: html

   <td>

This is an integer specifying the required size of the BAR4 frame buffer
region.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

bar4_base_addr_min

.. raw:: html

   </td>

.. raw:: html

   <td>

Integer

.. raw:: html

   </td>

.. raw:: html

   <td>

This is an integer specifying the minimum value the BAR4 base address
can be.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

bar4_base_addr_max

.. raw:: html

   </td>

.. raw:: html

   <td>

Integer

.. raw:: html

   </td>

.. raw:: html

   <td>

This is an integer specifying the maximum value the BAR4 base address
can be.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

bar5_req_size

.. raw:: html

   </td>

.. raw:: html

   <td>

Integer

.. raw:: html

   </td>

.. raw:: html

   <td>

This is an integer specifying the required size of the BAR5 frame buffer
region.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   </table>

.. _output-1:

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

bar1_size

.. raw:: html

   </td>

.. raw:: html

   <td>

Integer

.. raw:: html

   </td>

.. raw:: html

   <td>

The actual size of BAR1.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

bar1_base_addr

.. raw:: html

   </td>

.. raw:: html

   <td>

Integer

.. raw:: html

   </td>

.. raw:: html

   <td>

The actual base address of BAR1 memory.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

bar2_size

.. raw:: html

   </td>

.. raw:: html

   <td>

Integer

.. raw:: html

   </td>

.. raw:: html

   <td>

The actual size of BAR2.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

bar2_base_addr

.. raw:: html

   </td>

.. raw:: html

   <td>

Integer

.. raw:: html

   </td>

.. raw:: html

   <td>

The actual base address of BAR2 memory.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

bar4_size

.. raw:: html

   </td>

.. raw:: html

   <td>

Integer

.. raw:: html

   </td>

.. raw:: html

   <td>

The actual size of BAR4.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

bar4_base_addr

.. raw:: html

   </td>

.. raw:: html

   <td>

Integer

.. raw:: html

   </td>

.. raw:: html

   <td>

The actual base address of BAR4 memory.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

bar5_size

.. raw:: html

   </td>

.. raw:: html

   <td>

Integer

.. raw:: html

   </td>

.. raw:: html

   <td>

The actual size of BAR5.

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

String

.. raw:: html

   </td>

.. raw:: html

   <td>

‘true’ if all of the properties match the values given, ‘false’
otherwise.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   </table>

The qualification check will query the specified bar properties and
check that they satisfy the give parameters. The pass output key will be
true and the test will pass if all of the BAR properties satisfy the
constraints. After the check is finished, the following informational
messages will be generated:

::

   [INFO  ][<timestamp>][<action name>] smqt bar1_size <bar1_size>
   [INFO  ][<timestamp>][<action name>] smqt bar1_base_addr <bar1_base_addr>
   [INFO  ][<timestamp>][<action name>] smqt bar2_size <bar2_size>
   [INFO  ][<timestamp>][<action name>] smqt bar2_base_addr <bar2_base_addr>
   [INFO  ][<timestamp>][<action name>] smqt bar4_size <bar4_size>
   [INFO  ][<timestamp>][<action name>] smqt bar4_base_addr <bar4_base_addr>
   [INFO  ][<timestamp>][<action name>] smqt bar5_size <bar5_size>
   [RESULT][<timestamp>][<action name>] smqt <pass>

.. _examples-1:

Examples
~~~~~~~~

**Example 1:**

Consider this file (sizes are in bytes):

::

   actions:
   - name: action_1
     device: all
     module: smqt
     bar1_req_size: 17179869184
     bar1_base_addr_min: 0
     bar1_base_addr_max: 17592168044416
     bar2_req_size: 2097152
     bar2_base_addr_min: 0
     bar2_base_addr_max: 1099511627776
     bar4_req_size: 262144
     bar4_base_addr_min: 0
     bar4_base_addr_max: 17592168044416
     bar5_req_size: 131072

Results for three GPUs are:

::

   [INFO  ] [257936.568768] [action_1]  smqt bar1_size      17179869184 (16.00 GB)
   [INFO  ] [257936.568768] [action_1]  smqt bar1_base_addr 13C0000000C
   [INFO  ] [257936.568768] [action_1]  smqt bar2_size      2097152 (2.00 MB)
   [INFO  ] [257936.568768] [action_1]  smqt bar2_base_addr 13B0000000C
   [INFO  ] [257936.568768] [action_1]  smqt bar4_size      524288 (512.00 KB)
   [INFO  ] [257936.568768] [action_1]  smqt bar4_base_addr E4B00000
   [INFO  ] [257936.568768] [action_1]  smqt bar5_size      0 (0.00 B)
   [RESULT] [257936.568920] [action_1]  smqt fail
   [INFO  ] [257936.569234] [action_1]  smqt bar1_size      17179869184 (16.00 GB)
   [INFO  ] [257936.569234] [action_1]  smqt bar1_base_addr 1A00000000C
   [INFO  ] [257936.569234] [action_1]  smqt bar2_size      2097152 (2.00 MB)
   [INFO  ] [257936.569234] [action_1]  smqt bar2_base_addr 19F0000000C
   [INFO  ] [257936.569234] [action_1]  smqt bar4_size      524288 (512.00 KB)
   [INFO  ] [257936.569234] [action_1]  smqt bar4_base_addr E9900000
   [INFO  ] [257936.569234] [action_1]  smqt bar5_size      0 (0.00 B)
   [RESULT] [257936.569281] [action_1]  smqt fail
   [INFO  ] [257936.570798] [action_1]  smqt bar1_size      17179869184 (16.00 GB)
   [INFO  ] [257936.570798] [action_1]  smqt bar1_base_addr 16C0000000C
   [INFO  ] [257936.570798] [action_1]  smqt bar2_size      2097152 (2.00 MB)
   [INFO  ] [257936.570798] [action_1]  smqt bar2_base_addr 1710000000C
   [INFO  ] [257936.570798] [action_1]  smqt bar4_size      524288 (512.00 KB)
   [INFO  ] [257936.570798] [action_1]  smqt bar4_base_addr E7300000
   [INFO  ] [257936.570798] [action_1]  smqt bar5_size      0 (0.00 B)
   [RESULT] [257936.570837] [action_1]  smqt fail

In this example, BAR sizes reported by GPUs match those listed in
configuration key except for the BAR5, hence the test fails.
