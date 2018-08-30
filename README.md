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

Compile rocm_smi_lib (this needs to be done only once after cloning):

    cd $RVS
    cmake ./rocm_smi_lib -DROCM_SMI_BLD_BITS=64 -B../build/rocm_smi_lib
    make -C ../build/rocm_smi_lib

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

    cmake ./ -B../build    
    make -C ../build

Run:

    sudo bin/rvs -d 3

Build package:

    cd $WB/build
    make package

_Note:_ based on your OS, only DEB or RPM package will be buile.

For CentOS specific instructions see this [link](./CentOS.md).

