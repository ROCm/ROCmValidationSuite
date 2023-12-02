# ROCmValidationSuite
The ROCm Validation Suite (RVS) is a system validation and diagnostics tool for monitoring, stress testing, detecting and troubleshooting issues that affects the functionality and performance of AMD GPU(s) operating in a high-performance/AI/ML computing environment. RVS is enabled using the ROCm software stack on a compatible software and hardware platform.

The RVS is a collection of tests, benchmarks and qualification tools each targeting a specific sub-system of the ROCm platform. All of the tools are implemented in software and share a common command line interface. Each set of tests are implemented in a “module” which is a library encapsulating the functionality specific to the tool. The CLI can specify the directory containing modules to use when searching for libraries to load. Each module may have a set of options that it defines and a configuration file that supports its execution.

Different modules and its description refer [link](./FEATURES.md).

Module configuration files description and examples refer [link](./docs/ug1main.md).

## Prerequisites 
Please do this before compilation/installing compiled package.

Ubuntu :

    sudo apt-get -y update && sudo apt-get install -y libpci3 libpci-dev doxygen unzip cmake git libyaml-cpp-dev

CentOS :

    sudo yum install -y cmake3 doxygen pciutils-devel rpm rpm-build git gcc-c++ yaml-cpp-devel
 
RHEL :

    sudo yum install -y cmake3 doxygen rpm rpm-build git gcc-c++ yaml-cpp-devel
        
    wget http://mirror.centos.org/centos/7/os/x86_64/Packages/pciutils-devel-3.5.1-3.el7.x86_64.rpm
        
    sudo rpm -ivh pciutils-devel-3.5.1-3.el7.x86_64.rpm
		
SLES :

    sudo SUSEConnect -p sle-module-desktop-applications/15.1/x86_64
       
    sudo SUSEConnect --product sle-module-development-tools/15.1/x86_64
       
    sudo zypper  install -y cmake doxygen pciutils-devel libpci3 rpm git rpm-build gcc-c++ yaml-cpp-devel

## Install ROCm stack, rocblas and rocm-smi-lib
Install ROCm stack for Ubuntu/CentOS/SLES/RHEL, Refer https://github.com/RadeonOpenCompute/ROCm

_**Note:**_

rocm_smi64 package has been renamed to rocm-smi-lib64 from >= ROCm3.0. If you are using ROCm release < 3.0 , install the package as "rocm_smi64".
rocm-smi-lib64 package has been renamed to rocm-smi-lib from >= ROCm4.1.
 
Install rocBLAS and rocm-smi-lib : 

Ubuntu :

    sudo apt-get install rocblas rocm-smi-lib
   
CentOS & RHEL :

    sudo yum install --nogpgcheck rocblas rocm-smi-lib
   
SUSE :

    sudo zypper install rocblas rocm-smi-lib

_**Note:**_
If  rocm-smi-lib is already installed but "/opt/rocm/rocm_smi/ path doesn't exist. Do below:

Ubuntu :

    sudo dpkg -r rocm-smi-lib && sudo apt install rocm-smi-lib

CentOS & RHEL :

    sudo rpm -e  rocm-smi-lib && sudo yum install  rocm-smi-lib

SUSE :

    sudo rpm -e  rocm-smi-lib && sudo zypper install  rocm-smi-lib

## Building from Source
This section explains how to get and compile current development stream of RVS.

### Clone repository

    git clone https://github.com/ROCm-Developer-Tools/ROCmValidationSuite.git

### Configure:

    cd ROCmValidationSuite
    cmake -B ./build -DROCM_PATH=<rocm_installed_path> -DCMAKE_INSTALL_PREFIX=<rocm_installed_path> -DCPACK_PACKAGING_INSTALL_PREFIX=<rocm_installed_path>

    e.g. If ROCm 5.5 was installed,
    cmake -B ./build -DROCM_PATH=/opt/rocm-5.5.0 -DCMAKE_INSTALL_PREFIX=/opt/rocm-5.5.0 -DCPACK_PACKAGING_INSTALL_PREFIX=/opt/rocm-5.5.0

### Build binary:

    make -C ./build

### Build package:

    cd ./build
    make package

**Note:**_ based on your OS, only DEB or RPM package will be built. You may
ignore an error for the unrelated configuration

### Install built package:

Ubuntu :

    sudo dpkg -i rocm-validation-suite*.deb

CentOS & RHEL & SUSE :

    sudo rpm -i --replacefiles --nodeps rocm-validation-suite*.rpm

**Note:**
RVS is getting packaged as part of ROCm release starting from 3.0. You can install pre-compiled package as below.
Please make sure Prerequisites, ROCm stack, rocblas and rocm-smi-lib64 are already installed

### Install package packaged with ROCm release:

Ubuntu :

    sudo apt install rocm-validation-suite

CentOS & RHEL :

    sudo yum install rocm-validation-suite

SUSE :

    sudo zypper install rocm-validation-suite

## Executing RVS

### Execute version built from source code:

    cd ./build/bin
    ./rvs --help ; Lists all options to run RVS test suite
    ./rvs -g ; Lists supported GPUs available in the machine
    ./rvs -d 3 ; Executes set of RVS sanity tests (in rvs.conf) with verbose level 3
    ./rvs -d 3 ; Executes the complete RVS test suite
    ./rvs -c conf/gst_single.conf ; Executes GST test 

### Running version pre-complied and packaged with ROCm release

    /opt/rocm/rvs/rvs -d 3 ; Executes the complete RVS test suite

Similarly, you can run all tests as mentioned in "rvsqa.new.sh" script, present at "testscripts/rvsqa.new.sh"

## Executing RVS

### Execute version built from source code

    cd <source folder>/build/bin

    Command examples
    ./rvs --help ; Lists all options to run RVS test suite
    ./rvs -g ; Lists supported GPUs available in the machine
    ./rvs -d 3 ; Executes set of RVS sanity tests (rvs.conf)
    ./rvs -c conf/gst_single.conf ; Executes GST module tests

### Execute version pre-complied and packaged with ROCm release

    cd /opt/rocm/bin

    Command examples
    ./rvs --help ; Lists all options to run RVS test suite
    ./rvs -g ; Lists supported GPUs available in the machine
    ./rvs -d 3 ; Executes set of RVS sanity tests (in rvs.conf) with verbose level 3
    ./rvs -c conf/gst_single.conf ; Executes GST module tests

## Reporting

Test results, errors and verbose logs are printed as terminal output. To enable json logging use "-j" command line option.
The json output file is stored in /var/log folder and the name of the file will be printed.

