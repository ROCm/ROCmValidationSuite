
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
