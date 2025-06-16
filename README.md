# ROCmValidationSuite
The ROCm Validation Suite (RVS) is a system validation and diagnostics tool for monitoring, stress testing, detecting and troubleshooting issues that affects the functionality and performance of AMD GPU(s) operating in a high-performance/AI/ML computing environment. RVS is enabled using the ROCm software stack on a compatible software and hardware platform.

RVS is a collection of tests, benchmarks and qualification tools each targeting a specific sub-system of the ROCm platform. All of the tools are implemented in software and share a common command line interface. Each set of tests are implemented in a “module” which is a library encapsulating the functionality specific to the tool. The CLI can specify the directory containing modules to use when searching for libraries to load. Each module may have a set of options that it defines and a configuration file that supports its execution.

For different RVS modules and their description, refer to [the documentation on features](./FEATURES.md).

For module configuration files description and examples, refer to [the user guide](./docs/ug1main.md).

## Prerequisites
Please do this before compilation/installing compiled package.

Ubuntu :

    sudo apt-get -y update && sudo apt-get install -y libpci3 libpci-dev doxygen unzip cmake git libyaml-cpp-dev

CentOS :

    sudo yum install -y cmake3 doxygen pciutils-devel rpm rpm-build git gcc-c++ yaml-cpp-devel

RHEL :

    sudo yum install -y cmake3 doxygen rpm rpm-build git gcc-c++ yaml-cpp-devel pciutils-devel

SLES :

    sudo zypper install -y cmake doxygen pciutils-devel libpci3 rpm git rpm-build gcc-c++ yaml-cpp-devel

## Install ROCm stack, rocblas and rocm-smi-lib
Install ROCm stack for Ubuntu/CentOS/SLES/RHEL. Refer to
 [ROCm installation guide](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html) for more details.

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
If rocm-smi-lib is already installed but /opt/rocm/lib/librocm_smi64.so doesn't exist. Do below:

Ubuntu :

    sudo dpkg -r rocm-smi-lib && sudo apt install rocm-smi-lib

CentOS & RHEL :

    sudo rpm -e  rocm-smi-lib && sudo yum install  rocm-smi-lib

SUSE :

    sudo rpm -e  rocm-smi-lib && sudo zypper install  rocm-smi-lib

## Building from Source
This section explains how to get and compile current development stream of RVS.

### Clone repository

    git clone https://github.com/ROCm/ROCmValidationSuite.git

**Note:** The above command clones the master branch. If you're using a specific ROCm release, it's recommended to use the corresponding RVS version from the same release branch to ensure compatibility.

   e.g. If ROCm 6.4 is installed, clone the RVS repository from the 6.4 release branch.
```
git clone https://github.com/ROCm/ROCmValidationSuite.git -b release/rocm-rel-6.4
```
### Configure:

```
cd ROCmValidationSuite
cmake -B ./build -DROCM_PATH=<rocm_installed_path> -DCMAKE_INSTALL_PREFIX=<rocm_installed_path> -DCPACK_PACKAGING_INSTALL_PREFIX=<rocm_installed_path>
```

e.g. If ROCm 6.4 is installed, you can configure the build using one of the following cmake commands, depending on whether you're using the full versioned path or a symbolic link:

Option 1: Using the full versioned path
```
cmake -B ./build -DROCM_PATH=/opt/rocm-6.4.0 -DCMAKE_INSTALL_PREFIX=/opt/rocm-6.4.0 -DCPACK_PACKAGING_INSTALL_PREFIX=/opt/rocm-6.4.0
```

Option 2: Using the symbolic link
```
cmake -B ./build -DROCM_PATH=/opt/rocm -DCMAKE_INSTALL_PREFIX=/opt/rocm -DCPACK_PACKAGING_INSTALL_PREFIX=/opt/rocm
```

### Build binary:

    make -C ./build

**Note:** Use the -j option with the build command to enable parallel compilation, which can significantly speed up the build process.

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

## Running RVS

### Run version built from source code

    cd <source folder>/build/bin

    Command examples
    ./rvs --help ; Lists all options to run RVS test suite
    ./rvs -g ; Lists supported GPUs available in the machine
    ./rvs -c conf/gst_single.conf ; Run GST module default test configuration

### Run version pre-compiled and packaged with ROCm release

    cd /opt/rocm/bin

    Command examples
    ./rvs --help ; Lists all options to run RVS test suite
    ./rvs -g ; Lists supported GPUs available in the machine
    ./rvs -c ../share/rocm-validation-suite/conf/gst_single.conf ; Run GST default test configuration

To run GPU specific test configuration, use configuration files from GPU folders in "/opt/rocm/share/rocm-validation-suite/conf"

    ./rvs -c ../share/rocm-validation-suite/conf/MI300X/gst_single.conf ; Run MI300X specific GST test configuration
    ./rvs -c ../share/rocm-validation-suite/conf/nv32/gst_single.conf ; Run Navi 32 specific GST test configuration

Note: If present, always use GPU specific configurations instead of default test configurations.

## Reporting

Test results, errors and verbose logs are printed as terminal output. To enable json logging use "-j" command line option.
The json output file path can be specified after "-j" option. If not specified, a file is created in /var/tmp folder and the name of the file will be printed to stdout. The json file schemas are documented at [schemas](./docs/schemas)
