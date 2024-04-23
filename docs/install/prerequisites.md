## Prerequisites

Follow the instructions below before compilation/installing compiled package.

### Ubuntu:

    sudo apt-get -y update && sudo apt-get install -y libpci3 libpci-dev doxygen unzip cmake git libyaml-cpp-dev

### CentOS:

    sudo yum install -y cmake3 doxygen pciutils-devel rpm rpm-build git gcc-c++ yaml-cpp-devel

### RHEL:

    sudo yum install -y cmake3 doxygen rpm rpm-build git gcc-c++ yaml-cpp-devel

    wget http://mirror.centos.org/centos/7/os/x86_64/Packages/pciutils-devel-3.5.1-3.el7.x86_64.rpm

    sudo rpm -ivh pciutils-devel-3.5.1-3.el7.x86_64.rpm

### SLES:

    sudo SUSEConnect -p sle-module-desktop-applications/15.1/x86_64

    sudo SUSEConnect --product sle-module-development-tools/15.1/x86_64

    sudo zypper  install -y cmake doxygen pciutils-devel libpci3 rpm git rpm-build gcc-c++ yaml-cpp-devel

## Install ROCm stack, rocblas and rocm-smi-lib
Install ROCm stack for Ubuntu/CentOS/SLES/RHEL. Refer to
 [ROCm installation guide](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html) for more details.

**Note**

rocm_smi64 package has been renamed to rocm-smi-lib64 from >= ROCm3.0. If you are using ROCm release < 3.0 , install the package as "rocm_smi64".
rocm-smi-lib64 package has been renamed to rocm-smi-lib from >= ROCm4.1.

Install rocBLAS and rocm-smi-lib :

### Ubuntu:

    sudo apt-get install rocblas rocm-smi-lib

### CentOS & RHEL:

    sudo yum install --nogpgcheck rocblas rocm-smi-lib

### SUSE:

    sudo zypper install rocblas rocm-smi-lib

**Note**

If rocm-smi-lib is already installed but /opt/rocm/lib/librocm_smi64.so doesn't exist. Do below:

### Ubuntu:

    sudo dpkg -r rocm-smi-lib && sudo apt install rocm-smi-lib

### CentOS & RHEL:

    sudo rpm -e  rocm-smi-lib && sudo yum install  rocm-smi-lib

### SUSE:

    sudo rpm -e  rocm-smi-lib && sudo zypper install  rocm-smi-lib

## Building from Source
This section explains how to get and compile current development stream of RVS.

### Clone repository

    git clone https://github.com/ROCm/ROCmValidationSuite.git

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

**Note**

Based on your OS, only DEB or RPM package will be built. You may ignore an error for the unrelated configuration.

### Install built package:

#### Ubuntu:

    sudo dpkg -i rocm-validation-suite*.deb

#### CentOS, RHEL, and SUSE:

    sudo rpm -i --replacefiles --nodeps rocm-validation-suite*.rpm

**Note**

RVS is getting packaged as part of ROCm release starting from 3.0. You can install pre-compiled package as below.
Please make sure Prerequisites, ROCm stack, rocblas and rocm-smi-lib64 are already installed

### Install package packaged with ROCm release:

#### Ubuntu:

    sudo apt install rocm-validation-suite

#### CentOS & RHEL:

    sudo yum install rocm-validation-suite

#### SUSE:

    sudo zypper install rocm-validation-suite

## Running RVS

### Run version built from source code

    cd <source folder>/build/bin

    Command examples
    ./rvs --help ; Lists all options to run RVS test suite
    ./rvs -g ; Lists supported GPUs available in the machine
    ./rvs -d 3 ; Run set of RVS sanity tests (in rvs.conf) with verbose level 3
    ./rvs -c conf/gst_single.conf ; Run GST module tests

### Run version pre-complied and packaged with ROCm release

    cd /opt/rocm/bin

    Command examples
    ./rvs --help ; Lists all options to run RVS test suite
    ./rvs -g ; Lists supported GPUs available in the machine
    ./rvs -d 3 ; Run set of RVS sanity tests (in rvs.conf) with verbose level 3
    ./rvs -c conf/gst_single.conf ; Run GST module tests

Similarly, all RVS module tests can be run using scripts present in folder "/opt/rocm/share/rocm-validation-suite/testscripts/".

## Reporting

Test results, errors and verbose logs are printed as terminal output. To enable json logging use "-j" command line option.
The json output file is stored in /var/tmp folder and the name of the file will be printed.
