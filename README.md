# ROCmValidationSuite
The ROCm Validation Suite (RVS) is a system administrator’s and cluster manager's tool for detecting and troubleshooting common problems affecting AMD GPU(s) running in a high-performance computing environment, enabled using the ROCm software stack on a compatible platform.

The RVS is a collection of tests, benchmarks and qualification tools each targeting a specific sub-system of the ROCm platform. All of the tools are implemented in software and share a common command line interface. Each set of tests are implemented in a “module” which is a library encapsulating the functionality specific to the tool. The CLI can specify the directory containing modules to use when searching for libraries to load. Each module may have a set of options that it defines and a configuration file that supports its execution.

The function of each module see this [link](./FEATURES.md).

## Prerequisites

In order to build RVS from source please install prerequisites by following
this [link](./PREREQUISITES.md).

## Building from Source
This section explains how to get and compile current development stream of RVS.

Clone repository:

    cd /your/work/bench/folder
    export WB=$PWD
    git clone https://github.com/ROCm-Developer-Tools/ROCmValidationSuite.git
    
Init environment variables and submodule:

    export RVS=$WB/ROCmValidationSuite
    cd $RVS
    git submodule init
    git submodule update
    
Compile yaml-cpp (this needs to be done only once after cloning):

    cd $RVS
    cmake ./yaml-cpp -B../build/yaml-cpp
    make -C ../build/yaml-cpp


Compile rocm_smi_lib (this needs to be done only once after cloning).
_**Note:**_ current master branch uses pre 1.9.x `rocm_smi_lib`:

    cd $RVS/rocm_smi_lib/
    git checkout 87e7a6fad4ab976be25b7a620d8e1b411b633f0b
    cd $RVS
    cmake ./rocm_smi_lib -DROCM_SMI_BLD_BITS=64 -B../build/rocm_smi_lib
    make -C ../build/rocm_smi_lib

_**Note:**_ `rocm_smi_lib` handling will need to be changed once `rocm_smi_lib`
is included into ROCm package

Compile rocBLAS (this needs to be done only once after cloning):

    cd $RVS
    cd ../build
    git clone https://github.com/ROCmSoftwarePlatform/rocBLAS.git
    cd rocBLAS
    ./install.sh -d

Note:
- in case you want to install rocBLAS after compiling then you can use the _-i_ option (e.g.: _./install.sh -i_). In this case, _ROCBLAS_INC_DIR_ and _ROCBLAS_LIB_DIR_ have to be updated based on your rocBLAS installation location (e.g.: _/opt/rocm/rocblas/.._)
- if the latest version of rocBLAS is already installed on your system you may skip this step but you need to update the _ROCBLAS_INC_DIR_ and _ROCBLAS_LIB_DIR_ based on your rocBLAS installation location
- if rocBLAS dependencies are already satisfied then you can skip the _-d_ option
- in case _./install.sh -d_ fails please try without _-d_

Compile RVS:

    cd $RVS

    cmake -DCMAKE_BUILD_TYPE=Debug ./ -B../build
    make -C ../build

Build package:

    cd $WB/build
    make package

_**Note:**_ based on your OS, only DEB or RPM package will be built. You may
ignore an error for the unrelated configuration


## Running RVS

### Running version built from source code:

    cd ../build/bin
    sudo ./rvs -d 3

### Running without install

In general, it is possible to run RVS by simply coping all relevant files
onto another location (e.g., Docker image). Please note that in addition to
files in `bin/` folder you will also need to copy `rocm_smi_lib64.so`

In that case, you may run RVS using this command:

    sudo LD_LIBRARY_PATH=<_rocm_smi_lib_path> ./rvs ...

_**Note:**_ it is important to specify path to `rocm_smi_lib64.so` until this
library is fully included into ROCm distribution.

### Install package:

    sudo dpkg -i rocm-validation-suite.0.0.17
    sudo ldconfig

_**Note:**_ it is important to run `ldconfig` after install in order to refresh
dynamic linker cache.


For CentOS specific instructions see this [link](./CentOS.md).


## Regression

Simple regression has been implemented. You may find more about it
on this [link](./REGRESSION.md).
