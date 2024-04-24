.. meta::
  :description: Install ROCm Validation Suite
  :keywords: install, rocm validation suite, rvs, RVS, AMD, ROCm


**********************************
Installing ROCm Validation Suite
**********************************
    
ROCm Validation Suite (RVS) can be obtained by building it from source code base or by installing from pre-built package.

Building from source code
---------------------------

RVS has been developed as open source solution. Its source code and belonging documentation can be found at AMD's GitHub page.


Package manager installation
------------------------------
                                   
Based on the OS, use the appropriate package manager to install the rocm-validation-suite package.

For more details, refer to `ROCm Validation Suite GitHub site <https://github.com/ROCm/ROCmValidationSuite>`_

RVS package components are installed in `/opt/rocm`. The package contains:

- executable binary (located in _install-base_/bin/rvs)
- public shared libraries (located in _install-base_/lib)
- module specific shared libraries (located in _install-base_/lib/rvs)
- configuration files (located in _install-base_/share/rocm-validation-suite/conf)
- testscripts (located in _install-base_/share/rocm-validation-suite/testscripts)
- user guide (located in _install-base_/share/rocm-validation-suite/userguide)
- man page (located in _install-base_/share/man)


Prerequisites
------------------

Ensure you review the following prerequisites carefully for each operating system before compiling/installing the ROCm Validation Suite (RVS) package.

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

                sudo SUSEConnect -p sle-module-desktop-applications/15.1/x86_64
                        
                sudo SUSEConnect --product sle-module-development-tools/15.1/x86_64
                        
                sudo zypper  install -y cmake doxygen pciutils-devel libpci3 rpm git rpm-build gcc-c++ yaml-cpp-devel


    .. tab-item:: CentOS
         
          .. code-block:: shell   

                    sudo yum install -y cmake3 doxygen pciutils-devel rpm rpm-build git gcc-c++ yaml-cpp-devel                        

                    

Install ROCm stack, rocBLAS, and ROCm-SMI-lib
-----------------------------------------------

1. Install the ROCm ssoftware tack for Ubuntu/CentOS/SLES/RHEL. Refer to the `ROCm installation guide <https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html) for more details>`_

.. Note::

  The rocm_smi64 package has been renamed to rocm-smi-lib64 from >= ROCm3.0. If you are using ROCm release < 3.0 , install the package as "rocm_smi64". The `rocm-smi-lib64` package has been renamed to rocm-smi-lib from >= ROCm4.1.

2. Install rocBLAS and rocm-smi-lib.

.. tab-set::
    .. tab-item:: Ubuntu
      
          .. code-block:: shell

              sudo apt-get install rocblas rocm-smi-lib

    .. tab-item:: CentOS and RHEL
         
          .. code-block:: shell  

              sudo yum install --nogpgcheck rocblas rocm-smi-lib

    .. tab-item:: SUSE
         
          .. code-block:: shell  

              sudo zypper install rocblas rocm-smi-lib

.. Note:

If rocm-smi-lib is already installed but /opt/rocm/lib/librocm_smi64.so doesn't exist, perform the following steps:

.. tab-set::
    .. tab-item:: Ubuntu
       
         .. code-block:: shell  

              sudo dpkg -r rocm-smi-lib && sudo apt install rocm-smi-lib


    .. tab-item:: CentOS and RHEL

         .. code-block:: shell  

              sudo rpm -e  rocm-smi-lib && sudo yum install  rocm-smi-lib

    .. tab-item:: SUSE
         
          .. code-block:: shell  

              sudo rpm -e  rocm-smi-lib && sudo zypper install  rocm-smi-lib

Building from source
---------------------

This section explains how to get and compile current development stream of RVS.

1. Clone the repository.

.. code-block::

    git clone https://github.com/ROCm/ROCmValidationSuite.git

2. Configure. 

.. code-block::

    cd ROCmValidationSuite
    cmake -B ./build -DROCM_PATH=<rocm_installed_path> -DCMAKE_INSTALL_PREFIX=<rocm_installed_path> -DCPACK_PACKAGING_INSTALL_PREFIX=<rocm_installed_path>

For example, if ROCm 5.5 was installed, use the following instruction,

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

Based on your OS, only DEB or RPM package will be built. You may ignore an error for unrelated configurations.

5. Install the built package.

.. tab-set::
    .. tab-item:: Ubuntu       
          .. code-block:: 

              sudo dpkg -i rocm-validation-suite*.deb

   .. tab-item:: CentOS, RHEL, and SUSE
         .. code-block:: shell  

              sudo rpm -i --replacefiles --nodeps rocm-validation-suite*.rpm

**Note:**
RVS is getting packaged as part of ROCm release starting from 3.0. You can install pre-compiled package as below.
Please make sure Prerequisites, ROCm stack, rocblas and rocm-smi-lib64 are already installed

6. Install package packaged with ROCm release.




Reporting
-----------

Test results, errors and verbose logs are printed as terminal output. To enable json logging use "-j" command line option. The json output file is stored in /var/tmp folder and the name of the file will be printed.

RVS can be obtained by building it from source code base or by installing from pre-built package.


Running RVS
------------

Run version built from source code
+++++++++++++++++++++++++++++++++++

.. code-block::

    cd <source folder>/build/bin

    Command examples
    ./rvs --help ; Lists all options to run RVS test suite
    ./rvs -g ; Lists supported GPUs available in the machine
    ./rvs -d 3 ; Run set of RVS sanity tests (in rvs.conf) with verbose level 3
    ./rvs -c conf/gst_single.conf ; Run GST module tests

Run version pre-complied and packaged with ROCm release
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. code-block::

    cd /opt/rocm/bin

    Command examples
    ./rvs --help ; Lists all options to run RVS test suite
    ./rvs -g ; Lists supported GPUs available in the machine
    ./rvs -d 3 ; Run set of RVS sanity tests (in rvs.conf) with verbose level 3
    ./rvs -c conf/gst_single.conf ; Run GST module tests

Similarly, all RVS module tests can be run using scripts present in folder "/opt/rocm/share/rocm-validation-suite/testscripts/".

Building documentation
------------------------

Run the steps below to build documentation locally.

.. code-block::

        cd docs
        
        pip3 install -r .sphinx/requirements.txt
        
        python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html





