
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
