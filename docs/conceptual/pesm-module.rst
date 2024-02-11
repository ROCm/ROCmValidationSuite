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

.. raw:: html

   </tr>

.. raw:: html

   </table>
