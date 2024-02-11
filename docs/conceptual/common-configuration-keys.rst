.. meta::
  :description: rocm validation suite documentation 
  :keywords: rocm validation suite, ROCm, documentation

.. _common-configuration-keys:

Common configuration keys
--------------------------

Common configuration keys applicable to most module are summarized in the table below:\n

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

name

.. raw:: html

   </td>

.. raw:: html

   <td>

String

.. raw:: html

   </td>

.. raw:: html

   <td>

The name of the defined action.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

device

.. raw:: html

   </td>

.. raw:: html

   <td>

Collection of String

.. raw:: html

   </td>

.. raw:: html

   <td>

This is a list of device indexes (gpu ids), or the keyword “all”. The
defined actions will be executed on the specified device, as long as the
action targets a device specifically (some are platform actions). If an
invalid device id value or no value is specified the tool will report
that the device was not found and terminate execution, returning an
error regarding the configuration file.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

deviceid

.. raw:: html

   </td>

.. raw:: html

   <td>

Integer

.. raw:: html

   </td>

.. raw:: html

   <td>

This is an optional parameter, but if specified it restricts the action
to a specific device type corresponding to the deviceid.

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

If this key is false, actions will be run on one device at a time, in
the order specified in the device list, or the natural ordering if the
device value is “all”. If this parameter is true, actions will be run on
all specified devices in parallel. If a value isn’t specified the
default value is false.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

count

.. raw:: html

   </td>

.. raw:: html

   <td>

Integer

.. raw:: html

   </td>

.. raw:: html

   <td>

This specifies number of times to execute the action. If the value is 0,
execution will continue indefinitely. If a value isn’t specified the
default is 1. Some modules will ignore this parameter.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

wait

.. raw:: html

   </td>

.. raw:: html

   <td>

Integer

.. raw:: html

   </td>

.. raw:: html

   <td>

This indicates how long the test should wait between executions, in
milliseconds. Some modules will ignore this parameter. If the count key
is not specified, this key is ignored. duration Integer This parameter
overrides the count key, if specified. This indicates how long the test
should run, given in milliseconds. Some modules will ignore this
parameter.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

module

.. raw:: html

   </td>

.. raw:: html

   <td>

String

.. raw:: html

   </td>

.. raw:: html

   <td>

This parameter specifies the module that will be used in the execution
of the action. Each module has a set of sub-tests or sub-actions that
can be configured based on its specific parameters.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   </table>
pandoc version 3.1.11.1

