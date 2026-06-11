# ROCmValidationSuite
The ROCm Validation Suite (RVS) is a system validation and diagnostics tool for monitoring, stress testing, detecting and troubleshooting issues that affect the functionality and performance of AMD GPU(s) operating in a high-performance/AI/ML computing environment. RVS is enabled using the ROCm software stack on a compatible software and hardware platform.

RVS is a collection of tests, benchmarks and qualification tools each targeting a specific sub-system of the ROCm platform. All of the tools are implemented in software and share a common command-line interface. Each set of tests is implemented in a “module” which is a library encapsulating the functionality specific to the tool. The CLI can specify the directory containing modules to use when searching for libraries to load. Each module may have a set of options that it defines and a configuration file that supports its execution.

For different RVS modules and their description, refer to [the documentation on features](./FEATURES.md).

For module configuration files description and examples, refer to [the user guide](./docs/ug1main.md).

## Prerequisites
Please do this before compilation/installing compiled package.

Ubuntu :

```
sudo apt-get -y update && sudo apt-get install -y libpci3 libpci-dev doxygen unzip cmake git libyaml-cpp-dev libnuma-dev
```

**Note:** RVS requires CMake >= 3.25. Ubuntu 22.04 and earlier ship with an older version (3.22). If you are on Ubuntu 22.04 or earlier, install a newer CMake from Kitware's official APT repository before proceeding:

```
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc | sudo apt-key add -
sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ jammy main'
sudo apt-get update && sudo apt-get install cmake
```

CentOS :

```
sudo yum install -y cmake3 doxygen pciutils-devel rpm rpm-build git gcc-c++ yaml-cpp-devel yaml-cpp-static numactl-devel
```

RHEL :

```
sudo yum install -y cmake3 doxygen rpm rpm-build git gcc-c++ yaml-cpp-devel yaml-cpp-static pciutils-devel numactl-devel
```

SLES :

```
sudo zypper install -y cmake doxygen pciutils-devel libpci3 rpm git rpm-build gcc-c++ yaml-cpp-devel numactl-devel
```

## Install ROCm stack, rocBLAS, and SMI library

Install ROCm stack for Ubuntu/CentOS/SLES/RHEL. Refer to
 [ROCm installation guide](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html) for more details.

Install rocBLAS and the SMI library. For ROCm 6.4 and earlier, install `rocm-smi-lib`. For ROCm 7.0 and later, install `amd-smi-lib`.

**ROCm 6.4 and earlier:**

Ubuntu :

```
sudo apt-get install rocblas rocm-smi-lib
```

CentOS & RHEL :

```
sudo yum install --nogpgcheck rocblas rocm-smi-lib
```

SUSE :

```
sudo zypper install rocblas rocm-smi-lib
```

**ROCm 7.0 and later:**

Ubuntu :

```
sudo apt-get install rocblas amd-smi-lib
```

CentOS & RHEL :

```
sudo yum install --nogpgcheck rocblas amd-smi-lib
```

SUSE :

```
sudo zypper install rocblas amd-smi-lib
```

**Note:**
If `rocm-smi-lib` is already installed but `/opt/rocm/lib/librocm_smi64.so` does not exist, reinstall it as follows:

Ubuntu :

```
sudo dpkg -r rocm-smi-lib && sudo apt install rocm-smi-lib
```

CentOS & RHEL :

```
sudo rpm -e rocm-smi-lib && sudo yum install rocm-smi-lib
```

SUSE :

```
sudo rpm -e rocm-smi-lib && sudo zypper install rocm-smi-lib
```

## Building from Source

This section explains how to get and compile current development stream of RVS.

### Clone repository

```
git clone https://github.com/ROCm/ROCmValidationSuite.git
```

**Note:**
The above command clones the master branch. If you're using a specific ROCm release, it's recommended to use the corresponding RVS version from the same release branch to ensure compatibility.

If ROCm 6.4 is installed, clone the RVS repository from the 6.4 release branch by running:

```
git clone https://github.com/ROCm/ROCmValidationSuite.git -b release/rocm-rel-6.4
```

### Configure

```
cd ROCmValidationSuite
cmake -B ./build -DROCM_PATH=<rocm_installed_path> -DCMAKE_INSTALL_PREFIX=<rocm_installed_path> -DCPACK_PACKAGING_INSTALL_PREFIX=<rocm_installed_path>
```

If ROCm 6.4 is installed, you can configure the build using one of the following cmake commands, depending on whether you're using the full versioned path or a symbolic link:

Option 1: Using the full versioned path

```
cmake -B ./build -DROCM_PATH=/opt/rocm-6.4.0 -DCMAKE_INSTALL_PREFIX=/opt/rocm-6.4.0 -DCPACK_PACKAGING_INSTALL_PREFIX=/opt/rocm-6.4.0
```


Option 2: Using the symbolic link

```
cmake -B ./build -DROCM_PATH=/opt/rocm -DCMAKE_INSTALL_PREFIX=/opt/rocm -DCPACK_PACKAGING_INSTALL_PREFIX=/opt/rocm
```

### Build binary

```
make -C ./build -j $(nproc)
```

**Note:**
`$(nproc)` automatically uses all available CPU cores for parallel compilation, which can significantly speed up the build process. You can replace it with a specific number (e.g., `-j 8`) to limit core usage.

### Build package

```
cd ./build
make package
```

**Note:** Based on your OS, only DEB or RPM package will be built. You may
ignore an error for the unrelated configuration

### Install built package

Ubuntu :

```
sudo dpkg -i rocm-validation-suite*.deb
```

CentOS & RHEL & SUSE :

```
sudo rpm -i --replacefiles --nodeps rocm-validation-suite*.rpm
```

**Note:**
RVS is getting packaged as part of ROCm release starting from 3.0. You can install the pre-compiled package as below.
Please make sure Prerequisites, ROCm stack, rocblas and rocm-smi-lib64 are already installed

### Install package packaged with ROCm release

Ubuntu :

```
sudo apt install rocm-validation-suite
```

CentOS & RHEL :

```
sudo yum install rocm-validation-suite
```

SUSE :

```
sudo zypper install rocm-validation-suite
```

## Install RVS from Tarball - for TheRock based ROCm installation

Follow these steps to install RVS from a prebuilt tarball on top of an existing ROCm Core SDK installation using the TheRock build system.

### Download the RVS tarball

```
wget https://repo.amd.com/rocm/rvs/tarball/amdrocm7-rvs-1.4.21-288-Linux.tar.gz
```

### Extract the tarball

Set `ROCM_PATH` to your ROCm Core SDK location. For a package manager installation, the default is `/opt/rocm`:

```
export ROCM_PATH=/opt/rocm
sudo mkdir -p $ROCM_PATH/extras-7
sudo tar -xzf amdrocm7-rvs-1.4.21-288-Linux.tar.gz -C $ROCM_PATH/extras-7
```

### Set up your environment

**User setup** 

Add the following to `~/.bashrc`, then run `source ~/.bashrc`:

```
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/extras-7/bin:$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/extras-7/lib:$ROCM_PATH/lib:$ROCM_PATH/lib/llvm/lib:$LD_LIBRARY_PATH
```

**System-wide setup:**

```
sudo tee /etc/profile.d/set-rocm-env.sh << EOF
export ROCM_PATH=/opt/rocm
export PATH=\$ROCM_PATH/extras-7/bin:\$ROCM_PATH/bin:\$PATH
export LD_LIBRARY_PATH=\$ROCM_PATH/extras-7/lib:\$ROCM_PATH/lib:\$ROCM_PATH/lib/llvm/lib:\$LD_LIBRARY_PATH
EOF
sudo chmod +x /etc/profile.d/set-rocm-env.sh
source /etc/profile.d/set-rocm-env.sh
```

### Verify the installation

```
rvs -h
```

## Running RVS

### Run version built from source code

```
cd <source folder>/build/bin
```

Command examples:

```
./rvs --help  # Lists all options to run RVS test suite
./rvs -g      # Lists supported GPUs available in the machine
./rvs -c conf/gst_single.conf  # Run GST module default test configuration
./rvs -m gst  # Run GST module using platform-detected config
./rvs -r 3    # Run RVS level 3 (range: 1–5, 5 = highest) tests for the platform-detected
```

### Run version pre-compiled and packaged with ROCm release

```
cd /opt/rocm/bin
```

Command examples:

```
./rvs --help  # Lists all options to run RVS test suite
./rvs -g      # Lists supported GPUs available in the machine
./rvs -c ../share/rocm-validation-suite/conf/gst_single.conf  # Run GST default test configuration
./rvs -m gst  # Run GST module using platform-detected config
./rvs -r 3    # Run RVS level 3 (range: 1–5, 5 = highest) tests for the platform-detected
```

To run GPU specific test configuration, use configuration files from GPU folders in "/opt/rocm/share/rocm-validation-suite/conf"

```
./rvs -c ../share/rocm-validation-suite/conf/MI300X/gst_single.conf  # Run MI300X specific GST test configuration
./rvs -c ../share/rocm-validation-suite/conf/nv32/gst_single.conf  # Run Navi 32 specific GST test configuration
```

**Note:**
If present, always use GPU specific configurations instead of default test configurations.

## Reporting

Test results, errors and verbose logs are printed as terminal output. To enable JSON logging, use the `-j` command-line option.
The JSON output file path can be specified after the `-j` option. If not specified, a file is created in the `/var/tmp` folder and the name of the file will be printed to stdout. The JSON file schemas are documented at [schemas](./docs/schemas)
