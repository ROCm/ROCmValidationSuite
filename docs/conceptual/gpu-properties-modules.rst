
.. meta::
  :description: rocm validation suite documentation 
  :keywords: rocm validation suite, ROCm, documentation

.. _gpup-modules:




GPUP Modules
**************
The GPU properties module provides an interface to easily dump the static characteristics of a GPU. This information is stored in the sysfs file system
for the kfd, with the following path:

.. code-block::

    /sys/class/kfd/kfd/topology/nodes/<node id>

Each of the GPU nodes in the directory is identified with a number, indicating the device index of the GPU. This module will ignore count, duration
or wait key values.


Module Specific Keys
-----------------------

.. role:: raw-latex(raw)
   :format: latex
..

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

properties

.. raw:: html

   </td>

.. raw:: html

   <td>

Collection of Strings

.. raw:: html

   </td>

.. raw:: html

   <td>

The properties key specifies what configuration property or properties
the query is interested in. Possible values are::raw-latex:`\n` all -
collect all settings:raw-latex:`\n` gpu_id:raw-latex:`\n`
cpu_cores_count:raw-latex:`\n` simd_count:raw-latex:`\n`
mem_banks_count:raw-latex:`\n` caches_count:raw-latex:`\n`
io_links_count:raw-latex:`\n` cpu_core_id_base:raw-latex:`\n`
simd_id_base:raw-latex:`\n` max_waves_per_simd:raw-latex:`\n`
lds_size_in_kb:raw-latex:`\n` gds_size_in_kb:raw-latex:`\n`
wave_front_size:raw-latex:`\n` array_count:raw-latex:`\n`
simd_arrays_per_engine:raw-latex:`\n` cu_per_simd_array:raw-latex:`\n`
simd_per_cu:raw-latex:`\n` max_slots_scratch_cu:raw-latex:`\n`
vendor_id:raw-latex:`\n` device_id:raw-latex:`\n`
location_id:raw-latex:`\n` drm_render_minor:raw-latex:`\n`
max_engine_clk_fcompute:raw-latex:`\n` local_mem_size:raw-latex:`\n`
fw_version:raw-latex:`\n` capability:raw-latex:`\n`
max_engine_clk_ccompute:raw-latex:`\n`

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

io_links-properties

.. raw:: html

   </td>

.. raw:: html

   <td>

Collection of Strings

.. raw:: html

   </td>

.. raw:: html

   <td>

The properties key specifies what configuration property or properties
the query is interested in. Possible values are::raw-latex:`\n` all -
collect all settings:raw-latex:`\n` count - the number of
io_links:raw-latex:`\n` type:raw-latex:`\n` version_major:raw-latex:`\n`
version_minor:raw-latex:`\n` node_from:raw-latex:`\n`
node_to:raw-latex:`\n` weight:raw-latex:`\n` min_latency:raw-latex:`\n`
max_latency:raw-latex:`\n` min_bandwidth:raw-latex:`\n`
max_bandwidth:raw-latex:`\n` recommended_transfer_size:raw-latex:`\n`
flags:raw-latex:`\n`

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   </table>

Output
---------

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

properties-values

.. raw:: html

   </td>

.. raw:: html

   <td>

Collection of Integers

.. raw:: html

   </td>

.. raw:: html

   <td>

The collection will contain a positive integer value for each of the
valid properties specified in the properties config key.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

io_links-propertiesvalues

.. raw:: html

   </td>

.. raw:: html

   <td>

Collection of Integers

.. raw:: html

   </td>

.. raw:: html

   <td>

The collection will contain a positive integer value for each of the
valid properties specified in the io_links-properties config key.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   </table>

Each of the settings specified has a positive integer value. For each setting requested in the properties key, a message with the following format will
be returned:

.. code-block::

    [RESULT][<timestamp>][<action name>] gpup <gpu id> <property> <property value>

For each setting in the io_links-properties key, a message with the following format will be returned:

.. code-block::

    [RESULT][<timestamp>][<action name>] gpup <gpu id> <io_link id> <property> <property value>

Examples
---------

**Example 1:**

Consider action>

.. code-block::

    actions:
    - name: action_1
      device: all
      module: gpup
      properties:
        all:
      io_links-properties:
        all:


Action will display all properties for all compatible GPUs present in the system. Output for such configuration may be like this:

.. code-block::

    [RESULT] [597737.498442] action_1 gpup 3254 cpu_cores_count 0
    [RESULT] [597737.498517] action_1 gpup 3254 simd_count 256
    [RESULT] [597737.498558] action_1 gpup 3254 mem_banks_count 1
    [RESULT] [597737.498598] action_1 gpup 3254 caches_count 96
    [RESULT] [597737.498637] action_1 gpup 3254 io_links_count 1
    [RESULT] [597737.498680] action_1 gpup 3254 cpu_core_id_base 0
    [RESULT] [597737.498725] action_1 gpup 3254 simd_id_base 2147487744
    [RESULT] [597737.498768] action_1 gpup 3254 max_waves_per_simd 10
    [RESULT] [597737.498812] action_1 gpup 3254 lds_size_in_kb 64
    [RESULT] [597737.498856] action_1 gpup 3254 gds_size_in_kb 0
    [RESULT] [597737.498901] action_1 gpup 3254 wave_front_size 64
    [RESULT] [597737.498945] action_1 gpup 3254 array_count 4
    [RESULT] [597737.498990] action_1 gpup 3254 simd_arrays_per_engine 1
    [RESULT] [597737.499035] action_1 gpup 3254 cu_per_simd_array 16
    [RESULT] [597737.499081] action_1 gpup 3254 simd_per_cu 4
    [RESULT] [597737.499128] action_1 gpup 3254 max_slots_scratch_cu 32
    [RESULT] [597737.499175] action_1 gpup 3254 vendor_id 4098
    [RESULT] [597737.499222] action_1 gpup 3254 device_id 26720
    [RESULT] [597737.499270] action_1 gpup 3254 location_id 8960
    [RESULT] [597737.499318] action_1 gpup 3254 drm_render_minor 128
    [RESULT] [597737.499369] action_1 gpup 3254 max_engine_clk_ccompute 2200
    [RESULT] [597737.499419] action_1 gpup 3254 local_mem_size 17163091968
    [RESULT] [597737.499468] action_1 gpup 3254 fw_version 405
    [RESULT] [597737.499518] action_1 gpup 3254 capability 8832
    [RESULT] [597737.499569] action_1 gpup 3254 max_engine_clk_ccompute 2200
    [RESULT] [597737.499633] action_1 gpup 3254 0 count 1
    [RESULT] [597737.499675] action_1 gpup 3254 0 type 2
    [RESULT] [597737.499695] action_1 gpup 3254 0 version_major 0
    [RESULT] [597737.499716] action_1 gpup 3254 0 version_minor 0
    [RESULT] [597737.499736] action_1 gpup 3254 0 node_from 4
    [RESULT] [597737.499763] action_1 gpup 3254 0 node_to 1
    [RESULT] [597737.499783] action_1 gpup 3254 0 weight 20
    [RESULT] [597737.499808] action_1 gpup 3254 0 min_latency 0
    [RESULT] [597737.499830] action_1 gpup 3254 0 max_latency 0
    [RESULT] [597737.499853] action_1 gpup 3254 0 min_bandwidth 0
    [RESULT] [597737.499878] action_1 gpup 3254 0 max_bandwidth 0
    [RESULT] [597737.499902] action_1 gpup 3254 0 recommended_transfer_size 0
    [RESULT] [597737.499927] action_1 gpup 3254 0 flags 1
    [RESULT] [597737.500208] action_1 gpup 50599 cpu_cores_count 0
    [RESULT] [597737.500254] action_1 gpup 50599 simd_count 256
    ...
    [RESULT] [597737.501603] action_1 gpup 50599 0 recommended_transfer_size 0
    [RESULT] [597737.501626] action_1 gpup 50599 0 flags 1
    [RESULT] [597737.501877] action_1 gpup 33367 cpu_cores_count 0
    [RESULT] [597737.501921] action_1 gpup 33367 simd_count 256
    ...
    [RESULT] [597737.503258] action_1 gpup 33367 0 recommended_transfer_size 0
    [RESULT] [597737.503282] action_1 gpup 33367 0 flags 1
    ...

**Example 2:**

Consider action:

.. code-block::

    actions:
    - name: action_1
      device: all
      module: gpup
      properties:
        simd_count:
        mem_banks_count:
        io_links_count:
        vendor_id:
        device_id:
        location_id:
        max_engine_clk_ccompute:
      io_links-properties:
        version_major:
        type:
        version_major:
        version_minor:
        node_from:
        node_to:
        recommended_transfer_size:
        flags:

This action explicitly lists some of the properties.

Output for such configuration may be:

.. code-block::

    [RESULT] [597868.690637] action_1 gpup 3254 device_id 26720
    [RESULT] [597868.690713] action_1 gpup 3254 io_links_count 1
    [RESULT] [597868.690766] action_1 gpup 3254 location_id 8960
    [RESULT] [597868.690819] action_1 gpup 3254 max_engine_clk_ccompute 2200
    [RESULT] [597868.690862] action_1 gpup 3254 mem_banks_count 1
    [RESULT] [597868.690903] action_1 gpup 3254 simd_count 256
    [RESULT] [597868.690950] action_1 gpup 3254 vendor_id 4098
    [RESULT] [597868.691029] action_1 gpup 3254 0 flags 1
    [RESULT] [597868.691053] action_1 gpup 3254 0 node_from 4
    [RESULT] [597868.691075] action_1 gpup 3254 0 node_to 1
    [RESULT] [597868.691099] action_1 gpup 3254 0 recommended_transfer_size 0
    [RESULT] [597868.691119] action_1 gpup 3254 0 type 2
    [RESULT] [597868.691138] action_1 gpup 3254 0 version_major 0
    [RESULT] [597868.691158] action_1 gpup 3254 0 version_minor 0
    [RESULT] [597868.691425] action_1 gpup 50599 device_id 26720
    [RESULT] [597868.691469] action_1 gpup 50599 io_links_count 1
    [RESULT] [597868.691517] action_1 gpup 50599 location_id 17152
    ...
    [RESULT] [597868.692159] action_1 gpup 33367 device_id 26720
    [RESULT] [597868.692204] action_1 gpup 33367 io_links_count 1
    [RESULT] [597868.692252] action_1 gpup 33367 location_id 25344
    ...
    [RESULT] [597868.692619] action_1 gpup 33367 0 version_minor 0

**Example 3:**

Consider this action:

.. code-block::

    actions:
    - name: action_1
      device: all
      module: gpup
      deviceid: 267
      properties:
        all:
      io_links-properties:
        all:

Action lists deviceid 267 which is not present in the system.
Output for such configuration is:

.. code-block::

    RVS-GPUP: action: action_1  invalid 'deviceid' key value
