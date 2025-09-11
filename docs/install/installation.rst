.. meta::
  :description: Install ROCm Validation Suite
  :keywords: install, rocm validation suite, rvs, RVS, AMD, ROCm


**********************************
Installing ROCm Validation Suite
**********************************
    
You can obtain ROCm Validation Suite (RVS) by building it from:

* the source code base 

* a prebuilt package

Building from source code
---------------------------

RVS is an open-source solution. For more details, refer to the `ROCm Validation Suite GitHub repository. <https://github.com/ROCm/ROCmValidationSuite>`_


Package manager installation
------------------------------
                                   
Based on the OS, use the appropriate package manager to install the RVS package.

For more details, refer to the `ROCm Validation Suite GitHub repository. <https://github.com/ROCm/ROCmValidationSuite>`_

RVS package components are installed in ``/opt/rocm``. The package contains:

- executable binary, located in ``_install-base_/bin/rvs``.
- public shared libraries, located in ``_install-base_/lib``.
- module specific shared libraries, located in ``_install-base_/lib/rvs``.
- default configuration files, located in ``_install-base_/share/rocm-validation-suite/conf``.
- GPU specific configuration files, located in ``_install-base_/share/rocm-validation-suite/conf/<GPU folder>``.
- testscripts, located in ``_install-base_/share/rocm-validation-suite/testscripts``.
- user guide, located in ``_install-base_/share/rocm-validation-suite/userguide``.
- man page, located in ``_install-base_/share/man``.

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
         
          .. code-block:: shell                    
                    
                    sudo yum install -y cmake3 doxygen rpm rpm-build git gcc-c++ yaml-cpp-devel
                      
                    wget http://mirror.centos.org/centos/7/os/x86_64/Packages/pciutils-devel-3.5.1-3.el7.x86_64.rpm
                      
                    sudo rpm -ivh pciutils-devel-3.5.1-3.el7.x86_64.rpm

            
    .. tab-item:: SLES
        
        .. code-block:: shell
                        
                sudo zypper  install -y cmake doxygen pciutils-devel libpci3 rpm git rpm-build gcc-c++ yaml-cpp-devel                       


Install ROCm stack, rocBLAS, and ROCm-SMI-lib
-----------------------------------------------

1. Install the ROCm software stack for Ubuntu, SLES or RHEL. Refer to the `ROCm installation guide <https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html>`_ for more details. 

2. Install rocBLAS and rocm-smi-lib.

.. tab-set::
    .. tab-item:: Ubuntu
        :sync: Ubuntu
    
        .. code-block:: shell

            sudo apt-get install rocblas rocm-smi-lib

    .. tab-item:: RHEL
        :sync: RHEL

        .. code-block:: shell  

            sudo yum install --nogpgcheck rocblas rocm-smi-lib

    .. tab-item:: SUSE
        :sync: SUSE

        .. code-block:: shell  

            sudo zypper install rocblas rocm-smi-lib

If rocm-smi-lib is already installed, but ``/opt/rocm/lib/librocm_smi64.so`` doesn't exist, run the following command:

.. tab-set::
    .. tab-item:: Ubuntu
          :sync: Ubuntu
       
          .. code-block:: shell  

              sudo dpkg -r rocm-smi-lib && sudo apt install rocm-smi-lib


    .. tab-item:: RHEL
          :sync: RHEL

          .. code-block:: shell  

              sudo rpm -e  rocm-smi-lib && sudo yum install  rocm-smi-lib

    .. tab-item:: SUSE
         :sync: SUSE

         .. code-block:: shell  

             sudo rpm -e  rocm-smi-lib && sudo zypper install  rocm-smi-lib


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

    .. tab-item:: RHEL and SUSE

        .. code-block:: shell  

                sudo rpm -i --replacefiles --nodeps rocm-validation-suite*.rpm


.. Note::

    RVS is packaged as part of the ROCm release starting from 3.0. You can install the pre-compiled package as indicated below. Ensure prerequisites, ROCm stack, rocblas and rocm-smi-lib64 are already installed.

6. Install the package included with the ROCm release.

.. tab-set::
    .. tab-item:: Ubuntu
        :sync: Ubuntu

        .. code-block:: 

            sudo apt install rocm-validation-suite


    .. tab-item:: RHEL

        .. code-block:: shell  

                sudo yum install rocm-validation-suite

    .. tab-item:: SUSE

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

.. code-block::

        cd docs     
        pip3 install -r .sphinx/requirements.txt        
        python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html





