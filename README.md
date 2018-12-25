# ROCmValidationSuite
The ROCm Validation Suite (RVS) is a system administrator’s and cluster manager's tool for detecting and troubleshooting common problems affecting AMD GPU(s) running in a high-performance computing environment, enabled using the ROCm software stack on a compatible platform.

The RVS is a collection of tests, benchmarks and qualification tools each targeting a specific sub-system of the ROCm platform. All of the tools are implemented in software and share a common command line interface. Each set of tests are implemented in a “module” which is a library encapsulating the functionality specific to the tool. The CLI can specify the directory containing modules to use when searching for libraries to load. Each module may have a set of options that it defines and a configuration file that supports its execution.

The function of each module see this [link](./FEATURES.md).

## Prerequisites

In order to build RVS from source please install prerequisites by following
this [link](./PREREQUISITES.md).

## Building from Source
This section explains how to get and compile current development stream of RVS.

### Clone repository

    cd /your/work/bench/folder
    git clone https://github.com/ROCm-Developer-Tools/ROCmValidationSuite.git

### Configure and build RVS:

    cd ROCmValidationSuite

    cmake ./ -B../build
    make -C ../build

_**Note:**_

- To use `rocm_smi_lib64` library already installed on your system add `-DRVS_ROCMSMI=0`
to the cmake command

- To build RVS with local copy of rocBLAS add `-DRVS_ROCBLAS=1` to the cmake command

- To build RVS without tests insert  `-DRVS_BUILD_TESTS:BOOL=FALSE` define

Example:
```
    cmake -DRVS_ROCMSMI=0 -DRVS_ROCBLAS=1 ./ -B../build
```
_**Note:**_ for more details on how to speed up your build and install rocBLAS
localy, follow link [Building local rocBLAS](https://github.com/ROCm-Developer-Tools/ROCmValidationSuite/wiki/Building-local-rocBLAS)

### Build package:

     cd ./build
     make package

**Note:**_ based on your OS, only DEB or RPM package will be built. You may
ignore an error for the unrelated configuration


## Running RVS

### Running version built from source code:

    cd ../build/bin
    sudo ./rvs -d 3

### Running without install

In general, it is possible to run RVS by simply coping all relevant files from
`build/bin` folder onto another location (e.g., Docker image). Please note that
if RVS was built using local copy of `rocm_smi_lib`  you will also need to copy
`rocm_smi_lib64.so`

In that case, you may run RVS using this command:

    sudo LD_LIBRARY_PATH=<_rocm_smi_lib_path> ./rvs ...

_**Note:**_ it is important to specify path to `rocm_smi_lib64.so` until this
library is fully included into ROCm distribution.

### Install package:

    sudo dpkg -i rocm-validation-suite.0.0.23
    sudo ldconfig

_**Note:**_ it is important to run `ldconfig` after install in order to refresh
dynamic linker cache.


For CentOS specific instructions see this [link](./CentOS.md).


## Regression

Simple regression has been implemented. You may find more about it
on this [link](./REGRESSION.md).
