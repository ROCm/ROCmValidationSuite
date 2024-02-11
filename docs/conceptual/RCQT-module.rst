.. meta::
  :description: rocm validation suite documentation 
  :keywords: rocm validation suite, ROCm, documentation

.. _rcqt-module:


RCQT Module
-----------

This ‘module’ is actually a set of feature checks that target and
qualify the configuration of the platform. Many of the checks can be
done manually using the operating systems command line tools and general
knowledge about ROCm’s requirements. The purpose of the RCQT modules is
to provide an extensible, OS independent and scriptable interface
capable for performing the configuration checks required for ROCm
support. The checks in this module do not target a specific device
(instead the underlying platform is targeted), and any device or device
id keys specified will be ignored. Iteration keys, i.e. count, wait and
duration, are also ignored. :raw-latex:`\n`:raw-latex:`\n` One RCQT
action can perform only one check at the time. Checks are decoded in
this order::raw-latex:`\n`

-  If ‘package’ key is detected, packaging check will be performed
-  If ‘user’ key is detected, user check will be performed
-  If ‘os_versions’ and ‘kernel_versions’ keys are detected, OS check
   will be performed
-  If ‘soname’, ‘arch’ and ‘ldpath’ keys are detected, linker/loader
   check will be performed
-  If ‘file’ key is detected, file check will be performed

All other keys not pertinent to the detected action are ignored.

Packaging Check
~~~~~~~~~~~~~~~

This feature is used to check installed packages on the system. It
provides checks for installed packages and the currently available
package versions, if applicable.

Packaging Check Specific Keys
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Input keys are described in the table below:

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

package

.. raw:: html

   </td>

.. raw:: html

   <td>

String

.. raw:: html

   </td>

.. raw:: html

   <td>

Specifies the package to check. This key is required.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

version

.. raw:: html

   </td>

.. raw:: html

   <td>

String

.. raw:: html

   </td>

.. raw:: html

   <td>

This is an optional key specifying a package version. If it is provided,
the tool will check if the version is installed, failing otherwise. If
it is not provided any version matching the package name will result in
success.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   </table>

Output
^^^^^^

Output keys are described in the table below:

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

installed

.. raw:: html

   </td>

.. raw:: html

   <td>

Bool

.. raw:: html

   </td>

.. raw:: html

   <td>

If the test has passed, the output will be true. Otherwise it will be
false.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   </table>

The check will emit a result message with the following format:

::

   [RESULT][<timestamp>][<action name>] rcqt packagecheck <package> <installed>

The package name will include the version of the package if the version
key is specified. The installed output value will either be true or
false depending on if the package is installed or not.

Examples
^^^^^^^^

**Example 1:**

In this example, given package does not exist.

::

   actions:
   - name: action_1
     module: rcqt
     package: zip12345

The output for such configuration is:

::

   [RESULT] [500022.877512] [action_1] rcqt packagecheck zip12345 FALSE

**Example 2:**

In this example, version of the given package is incorrect.

::

   actions:
   - name: action_1
     module: rcqt
     package: zip
     version: 3.0-11****

The output for such configuration is:

::

   [RESULT] [500123.480561] [action_1] rcqt packagecheck zip FALSE

**Example 3:**

In this example, given package exists.

::

   actions:
   - name: action_1
     module: rcqt
     package: zip

The output for such configuration is:

::

   [RESULT] [500329.824495] [action_1] rcqt packagecheck zip TRUE

**Example 4:**

In this example, given package exists and its version is correct.

::

   actions:
   - name: action_1
     module: rcqt
     package: zip
     version: 3.0-11

The output for such configuration is:

::

   [RESULT] [500595.859025] [action_1] rcqt packagecheck zip TRUE

User Check
~~~~~~~~~~

This feature checks for the existence of a user and the user’s group
membership.

User Check Specific Keys
^^^^^^^^^^^^^^^^^^^^^^^^

Input keys are described in the table below:

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

user

.. raw:: html

   </td>

.. raw:: html

   <td>

String

.. raw:: html

   </td>

.. raw:: html

   <td>

Specifies the user name to check. This key is required.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

groups

.. raw:: html

   </td>

.. raw:: html

   <td>

Collection of Strings

.. raw:: html

   </td>

.. raw:: html

   <td>

This is an optional key specifying a collection of groups the user
should belong to. The user’s membership in each group will be checked.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   </table>

.. _output-1:

Output
^^^^^^

Output keys are described in the table below:

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

exists

.. raw:: html

   </td>

.. raw:: html

   <td>

Bool

.. raw:: html

   </td>

.. raw:: html

   <td>

This value is true if the user exists.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

members

.. raw:: html

   </td>

.. raw:: html

   <td>

Collection of Bools

.. raw:: html

   </td>

.. raw:: html

   <td>

This value is true if the user is a member of the specified group.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   </table>

The status of the user’s existence is provided in a message with the
following format:

::

   [RESULT][<timestamp>][<action name>] rcqt usercheck <user> <exists>

For each group in the list, a result message with the following format
will be generated:

::

   [RESULT][<timestamp>][<action name>] rcqt usercheck <user> <group> <member>

If the user doesn’t exist no group checks will take place.

.. _examples-1:

Examples
^^^^^^^^

**Example 1:**

In this example, given user does not exist.

::

   actions:
   - name: action_1
     device: all
     module: rcqt
     user: jdoe
     group: sudo,video

The output for such configuration is:

::

   [RESULT] [496559.219160] [action_1] rcqt usercheck jdoe false

Group check is not performed.

**Example 2:**

In this example, group **rvs** does not exist.

::

   actions:
   - name: action_1
     device: all
     module: rcqt
     user: jovanbhdl
     group: rvs,video

The output for such configuration is:

::

   [RESULT] [496984.993394] [action_1] rcqt usercheck jovanbhdl true
   [ERROR ] [496984.993535] [action_1] rcqt usercheck group rvs doesn't exist
   [RESULT] [496984.993578] [action_1] rcqt usercheck jovanbhdl video true

**Example 3:**

In this example, given user exists and belongs to given groups.

::

   actions:
   - name: action_1
     device: all
     module: rcqt
     user: jovanbhdl
     group: sudo,video

The output for such configuration is:

::

   [RESULT] [497361.361045] [action_1] rcqt usercheck jovanbhdl true
   [RESULT] [497361.361168] [action_1] rcqt usercheck jovanbhdl sudo true
   [RESULT] [497361.361209] [action_1] rcqt usercheck jovanbhdl video true

File/device Check
~~~~~~~~~~~~~~~~~

This feature checks for the existence of a file, its owner, group,
permissions and type. The primary purpose of this module is to check
that the device interfaces for the driver and the kfd are available, but
it can also be used to check for the existence of important
configuration files, libraries and executables.

File/device Check Specific Keys
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Input keys are described in the table below:

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

file

.. raw:: html

   </td>

.. raw:: html

   <td>

String

.. raw:: html

   </td>

.. raw:: html

   <td>

The value of this key should satisfy the file name limitations of the
target OS and specifies the file to check. This key is required.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

owner

.. raw:: html

   </td>

.. raw:: html

   <td>

String

.. raw:: html

   </td>

.. raw:: html

   <td>

The expected owner of the file. If this key is specified ownership is
tested.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

group

.. raw:: html

   </td>

.. raw:: html

   <td>

String

.. raw:: html

   </td>

.. raw:: html

   <td>

If this key is specified, group ownership is tested.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

permission

.. raw:: html

   </td>

.. raw:: html

   <td>

Integer

.. raw:: html

   </td>

.. raw:: html

   <td>

If this key is specified, the permissions on the file are tested. The
permissions are expected to match the permission value given.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

type

.. raw:: html

   </td>

.. raw:: html

   <td>

Integer

.. raw:: html

   </td>

.. raw:: html

   <td>

If this key is specified the file type is checked.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

exists

.. raw:: html

   </td>

.. raw:: html

   <td>

Bool

.. raw:: html

   </td>

.. raw:: html

   <td>

If this key is specified and set to false all optional parameters will
be ignored and a check will be made to make sure the file does not
exist. The default value for this key is true.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   </table>

.. _output-2:

Output
^^^^^^

Output keys are described in the table below:

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

owner

.. raw:: html

   </td>

.. raw:: html

   <td>

Bool

.. raw:: html

   </td>

.. raw:: html

   <td>

True if the correct user owns the file.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

group

.. raw:: html

   </td>

.. raw:: html

   <td>

Bool

.. raw:: html

   </td>

.. raw:: html

   <td>

True if the correct group owns the file.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

permission

.. raw:: html

   </td>

.. raw:: html

   <td>

Bool

.. raw:: html

   </td>

.. raw:: html

   <td>

True if the file has the correct permissions.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

type

.. raw:: html

   </td>

.. raw:: html

   <td>

Bool

.. raw:: html

   </td>

.. raw:: html

   <td>

True if the file is of the right type.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

exists

.. raw:: html

   </td>

.. raw:: html

   <td>

Bool

.. raw:: html

   </td>

.. raw:: html

   <td>

True if the file exists and the ‘exists’ config key is true. True if the
file does not exist and the ‘exists’ key if false.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   </table>

If the ‘exists’ key is true a set of messages, one for each stat check,
will be generated with the following format:

::

   [RESULT][<timestamp>][<action name>] rcqt filecheck <config key> <matching output key>

If the ‘exists’ key is false the format of the message will be:

::

   [RESULT][<timestamp>][<action name>] rcqt filecheck <file> DNE <exists>

.. _examples-2:

Examples
^^^^^^^^

**Example 1:**

In this example, config key exists is set to **true** by default and
file really exists so parameters are tested. Permission number 644
equals to rw-r–r– and type number 40 indicates that it is a folder.

rcqt_fc4.conf :

::

   actions:
   - name: action_1
     device: all
     module: rcqt
     file: /work/mvisekrunahdl/ROCmValidationSuite/rcqt.so/src
     owner: mvisekrunahdl
     group: mvisekrunahdl
     permission: 664
     type: 40

Output from running this action:

::

   [RESULT] [240384.678074] [action_1] rcqt filecheck mvisekrunahdl owner:true
   [RESULT] [240384.678214] [action_1] rcqt filecheck mvisekrunahdl group:true
   [RESULT] [240384.678250] [action_1] rcqt filecheck 664 permission:true
   [RESULT] [240384.678275] [action_1] rcqt filecheck 100 type:true

**Example 2:**

In this example, config key exists is set to false, but file actually
exists so parameters are not tested.

rcqt_fc1.conf:

::

   actions:
   - name: action_1
     device: all
     module: rcqt
     file: /work/mvisekrunahdl/ROCmValidationSuite/src
     owner: root
     permission: 644
     type: 40
     exists: false

The output for such configuration is:

::

   [RESULT] [240188.150386] [action_1] rcqt filecheck /work/mvisekrunahdl/ROCmValidationSuite/src DNE false

**Example 3:**

In this example, config key **exists** is true by default and file
really exists. Config key **group, permission** and **type** are not
specified so only ownership is tested.

rcqt_fc2.conf:

::

   actions:
   - name: action_1
     device: all
     module: rcqt
     file: /work/mvisekrunahdl/build/test.txt
     owner: root

The output for such configuration is:

::

   [RESULT] [240253.957738] [action_1] rcqt filecheck root owner:true

**Example 4:**

In this example, config key **exists** is true by default, but given
file does not exist.

rcqt_fc3.conf:

::

   actions:
   - name: action_1
     device: all
     module: rcqt
     file: /work/mvisekrunahdl/ROCmValidationSuite/rcqt.so/src/tst
     owner: mvisekrunahdl
     group: mvisekrunahdl
     permission: 664
     type: 100

The output for such configuration is:

::

   [ERROR ] [240277.355553] [action_1] rcqt File is not found

Kernel compatibility Check
~~~~~~~~~~~~~~~~~~~~~~~~~~

The rcqt-kernelcheck module determines the version of the operating
system and the kernel installed on the platform and compares the values
against the list of supported values.

Kernel compatibility Check Specific Keys
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Input keys are described in the table below:

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

os_versions

.. raw:: html

   </td>

.. raw:: html

   <td>

Collection of Strings

.. raw:: html

   </td>

.. raw:: html

   <td>

A collection of strings corresponding to operating systems names, i.e.
{“Ubuntu 16.04.3 LTS”, “Centos 7.4”, etc.}

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

kernel_versions

.. raw:: html

   </td>

.. raw:: html

   <td>

Collection of Strings

.. raw:: html

   </td>

.. raw:: html

   <td>

A collection of strings corresponding to kernel version names, i.e.
{“4.4.0-116-generic”, “4.13.0-36-generic”, etc.}

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   </table>

.. _output-3:

Output
^^^^^^

Output keys are described in the table below:

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

os

.. raw:: html

   </td>

.. raw:: html

   <td>

String

.. raw:: html

   </td>

.. raw:: html

   <td>

The actual OS installed on the system.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

kernel

.. raw:: html

   </td>

.. raw:: html

   <td>

String

.. raw:: html

   </td>

.. raw:: html

   <td>

The actual kernel version installed on the system.

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

Bool

.. raw:: html

   </td>

.. raw:: html

   <td>

True if the actual os version and kernel version match any value
provided in the collection.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   </table>

If the detected versions of the operating system and the kernel version
match any of the supported values the pass output key will be true.
Otherwise it will be false. The result message will contain the actual
os version and the kernel version regardless of where the check passed
or failed.

::

   [RESULT][<timestamp>][<action name>] rcqt kernelcheck <os version> <kernel version> <pass>

.. _examples-3:

Examples
^^^^^^^^

**Example 1:**

In this example, given kernel version is incorrect.

::

   actions:
   - name: action_1
     device: all
     module: rcqt
     os_version: Ubuntu 16.04.5 LTS
     kernel_version: 4.4.0-116-generic-wrong

The output for such configuration is:

::

   [RESULT] [498398.774182] [action_1] rcqt kernelcheck Ubuntu 16.04.5 LTS 4.18.0-rc1-kfd-compute-roc-master-8874 fail

**Example 2**

In this example, given os version and kernel verison are the correct
ones.

::

   actions:
   - name: action_1
     device: all
     module: rcqt
     os_version: Ubuntu 16.04.5 LTS
     kernel_version: 4.18.0-rc1-kfd-compute-roc-master-8874

The output for such configuration is:

::

   [RESULT] [515924.695932] [action_1] rcqt kernelcheck Ubuntu 16.04.5 LTS 4.18.0-rc1-kfd-compute-roc-master-8874 pass

Linker/Loader Check
~~~~~~~~~~~~~~~~~~~

This feature checks that a search by the linker/loader for a library
finds the correct version in the correct location. The check should
include a SONAME version of the library, the expected location and the
architecture of the library.

Linker/Loader Check Specific Keys
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Input keys are described in the table below:

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

soname

.. raw:: html

   </td>

.. raw:: html

   <td>

String

.. raw:: html

   </td>

.. raw:: html

   <td>

This is the SONAME of the library for the check. An SONAME library
contains the major version of the library in question.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

arch

.. raw:: html

   </td>

.. raw:: html

   <td>

String

.. raw:: html

   </td>

.. raw:: html

   <td>

This value qualifies the architecture expected for the library.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

ldpath

.. raw:: html

   </td>

.. raw:: html

   <td>

String

.. raw:: html

   </td>

.. raw:: html

   <td>

This is the fully qualified path where the library is expected to be
located.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   </table>

.. _output-4:

Output
^^^^^^

Output keys are described in the table below:

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

arch

.. raw:: html

   </td>

.. raw:: html

   <td>

String

.. raw:: html

   </td>

.. raw:: html

   <td>

The actual architecture found for the file, or NA if it wasn’t found.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

path

.. raw:: html

   </td>

.. raw:: html

   <td>

String

.. raw:: html

   </td>

.. raw:: html

   <td>

The actual path the linker is looking for the file at, or “not found” if
the file isn’t found.

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

Bool

.. raw:: html

   </td>

.. raw:: html

   <td>

True if the linker/loader is looking for the file in the correct place
with the correctly specified architecture.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   </table>

If the linker/loader search path looks for the soname version of the
library, qualified by arch, at the directory specified the test will
pass. Otherwise it will fail. The output message has the following
format:

::

   [RESULT][<timestamp>][<action name>] rcqt ldconfigcheck <soname> <arch> <path> <pass>

.. _examples-4:

Examples
^^^^^^^^

**Example 1:**

Consider this action:

::

   actions:
   - name: action_1
     device: all
     module: rcqt
     soname: librcqt.so.0.0.3fail
     arch: i386:x86-64
     ldpath: /work/jovanbhdl/build/bin

The test will fail because the file given is not found on the specified
path:

::

   [RESULT] [587042.789384] [action_1] rcqt ldconfigcheck librcqt.so.0.0.3fail i386:x86-64 /work/jovanbhdl/build/bin false

**Example 2:**

Consider this action:

::

   actions:
   - name: action_1
     device: all
     module: rcqt
     soname: librcqt.so.0.0.16
     arch: i386:x86-64
     ldpath: /work/jovanbhdl/build/bin

The test will pass and will output the message:

::

   [RESULT] [587047.395787] [action_1] rcqt ldconfigcheck librcqt.so.0.0.16 i386:x86-64 /work/jovanbhdl/build/bin true
