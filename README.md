# ROCmValidationSuite

The ROCm Validation Suite (RVS) is a system validation and diagnostics tool for monitoring, stress
testing, detecting, and troubleshooting issues that affect the functionality and performance of AMD
GPUs operating in a high-performance computing environment.

RVS consists of tests, benchmarks, and qualification tools that each target a specific sub-system of the
ROCm platform. All of the tools share a common command line interface (CLI) and are implemented in
a module. Each module defines a set of options and contains a configuration file that supports
running the module.

For a list of available modules, refer to the
[RVS modules](https://rocm.docs.amd.com/projects/ROCmValidationSuite/en/latest/features.html)
page.

Documentation for ROCm Validation Suite (RVS) is available at
[https://rocm.docs.amd.com/projects/ROCmValidationSuite/en/latest/](https://rocm.docs.amd.com/projects/ROCmValidationSuite/en/latest/).

## Prerequisites

1. Before compiling and installing the RVS package, you must run the following code:

   * Ubuntu:

     ```bash
     sudo apt-get -y update && sudo apt-get install -y libpci3 libpci-dev doxygen unzip cmake git libyaml-cpp-dev
     ```

   * CentOS:

     ```bash
     sudo yum install -y cmake3 doxygen pciutils-devel rpm rpm-build git gcc-c++ yaml-cpp-devel
     ```

   * RHEL:

     ```bash
     sudo yum install -y cmake3 doxygen rpm rpm-build git gcc-c++ yaml-cpp-devel \
     wget http://mirror.centos.org/centos/7/os/x86_64/Packages/pciutils-devel-3.5.1-3.el7.x86_64.rpm \
     sudo rpm -ivh pciutils-devel-3.5.1-3.el7.x86_64.rpm
     ```

   * SLES:

     ```bash
     sudo SUSEConnect -p sle-module-desktop-applications/15.1/x86_64 \
     sudo SUSEConnect --product sle-module-development-tools/15.1/x86_64 \
     sudo zypper  install -y cmake doxygen pciutils-devel libpci3 rpm git rpm-build gcc-c++ yaml-cpp-devel
     ```

2. Install ROCm. Refer to the
  [ROCm repository](https://github.com/RadeonOpenCompute/ROCm) for instructions.

3. Install rocBLAS and rocm-smi-lib.

   * Ubuntu:

     ```bash
     sudo apt-get install rocblas rocm-smi-lib
     ```

   * CentOS and RHEL:

     ```bash
     sudo yum install --nogpgcheck rocblas rocm-smi-lib
     ```

   * SUSE:

     ```bash
     sudo zypper install rocblas rocm-smi-lib
     ```

    If the `rocm-smi-lib` is already installed but the `/opt/rocm/rocm_smi/` path doesn't exist, run the
  following code:

   * Ubuntu:

      ```bash
      sudo dpkg -r rocm-smi-lib && sudo apt install rocm-smi-lib
      ```

   * CentOS and RHEL:

      ```bash
      sudo rpm -e  rocm-smi-lib && sudo yum install  rocm-smi-lib
      ```

   * SUSE:

      ```bash
      sudo rpm -e  rocm-smi-lib && sudo zypper install  rocm-smi-lib
      ```

## Build and install

You can install RVS from source or from the ROCm install package.

### Install from source

To download and compile the current development stream of RVS, follow these steps:

1. Clone the repository.

    ```bash
    git clone https://github.com/ROCm-Developer-Tools/ROCmValidationSuite.git
    ```

2. Configure the installation for your ROCm version.

    ```bash
    cd ROCmValidationSuite \
    cmake -B ./build -DROCM_PATH=<rocm_installed_path> -DCMAKE_INSTALL_PREFIX=<rocm_installed_path> -DCPACK_PACKAGING_INSTALL_PREFIX=<rocm_installed_path>
    ```

3. Build the binary.

    ```bash
    make -C ./build
    ```

4. Build the package. Depending on your operating system, either a DEB or an RPM package is built.
  You can ignore an error for the irrelevant configuration.

    ```bash
    cd ./build
    make package
    ```

5. Install the built package.

* Ubuntu:

    ```bash
    sudo dpkg -i rocm-validation-suite*.deb
    ```

* CentOS, RHEL, and SUSE:

    ```bash
    sudo rpm -i --replacefiles --nodeps rocm-validation-suite*.rpm
    ```

### Using the install package

* Ubuntu:

    ```bash
    sudo apt install rocm-validation-suite
    ```

* CentOS and RHEL:

    ```bash
    sudo yum install rocm-validation-suite
    ```

* SUSE:

    ```bash
    sudo zypper install rocm-validation-suite
    ```

## Running RVS

To run RVS, use one of the following options:

* Running the version built from source code:

    ```bash
    cd ./build/bin
    ./rvs --help ; Lists all options to run RVS test suite
    ./rvs -g ; Lists supported GPUs available in the machine
    ./rvs -d 3 ; Executes the complete RVS test suite
    ./rvs -c conf/gst_single.conf ; Executes GST test only
    ```

* Running the version pre-complied and packaged with the ROCm release:

    ```bash
    /opt/rocm/rvs/rvs -d 3 ; Executes the complete RVS test suite
    ```

Similarly, you can run all tests as described in the `rvsqa.new.sh` script, which is located at
`testscripts/rvsqa.new.sh`.

## Regression

Simple regression is implemented. To learn more, refer to the
[Regression](https://rocm.docs.amd.com/projects/ROCmValidationSuite/en/latest/regression.html)
page.

## Reporting

Test-based reporting is enabled. JSON-based reporting is added to the `gst` and `iet` modules. To
enable JSON logging use the `-j` command line option:

```bash
./rvs -c conf/gst_sinle.conf -d 3 -j
```

The JSON location is `/var/log folder` and the name of the file is printed in the `stdout`.

The output structure is:

```bash
{
{"module-name":{
  "action-name":[

{
    "target" : "<flops/power>"
  },
{
    "dtype" : "optype"
  },
{
    "gpu_id" : "63217",
    "GFLOPS" : "11433.352136"
  },
{
    "gpu_id" : "63217",
    "GFLOPS" : "11436.291718"
  },
....]
}
}
....
}
```

Here is a gst example:

```bash
{"gst":{
  "gpustress-9000-sgemm-false":[

{
    "target" : "9000.000000"
  },
{
    "dtype" : "sgemm"
  },
{
    "gpu_id" : "63217",
    "GFLOPS" : "11433.352136"
  },
{
    "gpu_id" : "63217",
    "GFLOPS" : "11436.291718"
  }]
  }
  }
  {"gst":{
  "gpustress-8000-sgemm-true":[
,
{
    "target" : "8000.000000"
  },
{
    "dtype" : "sgemm"
  },
{
    "gpu_id" : "63217",
    "GFLOPS" : "11657.886019"
  },
{
    "gpu_id" : "63217",
    "GFLOPS" : "11675.718793"
  },
{
    "gpu_id" : "63217",
    "GFLOPS" : "11687.461158"
  } ]
  }
 }
```
