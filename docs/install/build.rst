.. meta::
  :description: Build ROCm Validation Suite from source
  :keywords: install, rocm validation suite, rvs, RVS, AMD, ROCm, build, source


***************************************
Build ROCm Validation Suite from source
***************************************

Use the following steps to build and install RVS from source.

Prerequisites
------------------

RVS has been tested on all ROCm-supported Linux environments except for RHEL 9.4. See `Supported operating systems <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-operating-systems>`_ for the complete list of ROCm-supported Linux environments.

.. Note::

    This topic provides commands for the primary Linux distribution families. These commands are also applicable to other operating systems derived from the same families.

Ensure you review the following prerequisites carefully for each operating system before compiling or installing the RVS package.

.. tab-set::
    .. tab-item:: Ubuntu
        :sync: Ubuntu

        .. code-block:: shell

                sudo apt-get -y update && sudo apt-get install -y libpci3 libpci-dev doxygen unzip cmake git libyaml-cpp-dev

    .. tab-item:: RHEL
        :sync: RHEL

        .. code-block:: shell

                sudo yum install -y cmake3 doxygen rpm rpm-build git gcc-c++ yaml-cpp-devel pciutils-devel

    .. tab-item:: SUSE
        :sync: SUSE

        .. code-block:: shell

                sudo zypper  install -y cmake doxygen pciutils-devel libpci3 rpm git rpm-build gcc-c++ yaml-cpp-devel


Install the ROCm Core SDK
-------------------------

RVS is a ROCm Extra requiring the ROCm Core SDK to be installed.

For instructions, see `Install AMD ROCm
<https://rocm.amd.com/en/7.13.0-preview/install/rocm.html?fam=all&i=pkgman>`__. Use the
selector panel on that page to view instructions appropriate for your system
environment.


Building from source
---------------------

This section explains how to get and compile the current development stream of RVS.

1. Clone the repository.

.. code-block::

    git clone https://github.com/ROCm/ROCmValidationSuite.git

2. Configure the build system for RVS.

.. code-block::

    cd ROCmValidationSuite
    cmake -B ./build -DROCM_PATH=<rocm_installed_path> -DCMAKE_INSTALL_PREFIX=<rocm_installed_path> -DCPACK_PACKAGING_INSTALL_PREFIX=<rocm_installed_path>

For example, if ROCm 5.5 was installed, run the following command:

.. code-block::

    cmake -B ./build -DROCM_PATH=/opt/rocm-5.5.0 -DCMAKE_INSTALL_PREFIX=/opt/rocm-5.5.0 -DCPACK_PACKAGING_INSTALL_PREFIX=/opt/rocm-5.5.0

3. Build the binary.

.. code-block::

    make -C ./build

4. Build the package.

.. code-block::

    cd ./build
    make package

.. Note::

    Depending on your OS, only DEB or RPM package will be built.

.. Note::

    You can ignore errors about unrelated configurations.

5. Install the built package.

.. tab-set::
    .. tab-item:: Ubuntu
        :sync: Ubuntu

        .. code-block::

            sudo dpkg -i rocm-validation-suite*.deb

    .. tab-item:: RHEL
        :sync: RHEL

        .. code-block:: shell

                sudo rpm -i --replacefiles --nodeps rocm-validation-suite*.rpm

    .. tab-item:: SUSE
        :sync: SUSE

        .. code-block:: shell

                sudo rpm -i --replacefiles --nodeps rocm-validation-suite*.rpm

.. note::

    RVS is packaged as part of the ROCm release starting from 3.0. You can install the pre-compiled package as indicated below. Ensure prerequisites, ROCm stack, rocblas and rocm-smi-lib64 are already installed.

6. Install the package included with the ROCm release.

.. tab-set::
    .. tab-item:: Ubuntu
        :sync: Ubuntu

        .. code-block::

            sudo apt install rocm-validation-suite


    .. tab-item:: RHEL
        :sync: RHEL

        .. code-block:: shell

                sudo yum install rocm-validation-suite

    .. tab-item:: SUSE
        :sync: SUSE

        .. code-block:: shell

                sudo zypper install rocm-validation-suite


Reporting
-----------

Test results, errors, and verbose logs are printed as terminal output. To enable JSON logging, use the ``-j`` option. The JSON output file is stored in the ``/var/tmp`` folder and the file name will be printed.

You can build RVS from the source code base or by installing from a pre-built package. See the preceding sections for more details.

Running RVS
------------

Run the version built from source code
++++++++++++++++++++++++++++++++++++++

.. code-block::

    cd <source folder>/build/bin

    Command examples
    ./rvs --help ; Lists all options to run RVS test suite
    ./rvs -g ; Lists supported GPUs available in the machine
    ./rvs -d 3 ; Run set of RVS default sanity tests (in rvs.conf) with verbose level 3
    ./rvs -c conf/gst_single.conf ; Run GST module default test configuration

Run the version pre-compiled and packaged with the ROCm release
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. code-block::

    cd /opt/rocm/bin

    Command examples
    ./rvs --help ; Lists all options to run RVS test suite
    ./rvs -g ; Lists supported GPUs available in the machine
    ./rvs -d 3 ; Run set of RVS sanity tests (in rvs.conf) with verbose level 3
    ./rvs -c ../share/rocm-validation-suite/conf/gst_single.conf ; Run GST default test configuration

To run GPU-specific test configurations, use the configuration files in the GPU folders under ``/opt/rocm/share/rocm-validation-suite/conf``.

.. code-block::

    ./rvs -c ../share/rocm-validation-suite/conf/MI300X/gst_single.conf ; Run MI300X specific GST test configuration
    ./rvs -c ../share/rocm-validation-suite/conf/nv32/gst_single.conf ; Run Navi 32 specific GST test configuration

.. Note::

    Always use GPU-specific configurations over the default test configurations.

Building documentation
------------------------

Run the following commands to build documentation locally.

1. Create a Python virtual environment and install documentation dependencies.

   .. code-block:: bash

      python3.12 -m venv docs/.venv
      source docs/.venv/bin/activate

      pip install -r docs/sphinx/requirements.txt

2. Build the documentation using Sphinx.

   python -m sphinx docs docs/_build -E -j auto

3. Open ``docs/_build/index.html`` in your web browser to view the
   documentation.
