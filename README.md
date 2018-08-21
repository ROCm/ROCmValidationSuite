# ROCmValidationSuite
The ROCm Validation Suite (RVS) is a system administrator’s and cluster manager's tool for detecting and troubleshooting common problems affecting AMD GPU(s) running in a high-performance computing environment, enabled using the ROCm software stack on a compatible platform.

The RVS is a collection of tests, benchmarks and qualification tools each targeting a specific sub-system of the ROCm platform. All of the tools are implemented in software and share a common command line interface. Each set of tests are implemented in a “module” which is a library encapsulating the functionality specific to the tool. The CLI can specify the directory containing modules to use when searching for libraries to load. Each module may have a set of options that it defines and a configuration file that supports its execution.

The function of each module see this [link](./FEATURES.md).

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
    cd ../build/yaml-cpp
    make

Compile rocm_smi_lib (this needs to be done only once after cloning):

    cd $RVS
    cmake ./rocm_smi_lib -DROCM_SMI_BLD_BITS=64 -B../build/rocm_smi_lib
    cd ../build/rocm_smi_lib
    make

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
    # Contains header files exported by ROC Runtime
    export ROCR_INC_DIR=/opt/rocm/include/

    # Contains library files exported by ROC Runtime
    export ROCR_LIB_DIR=/opt/rocm/lib/

    # Contains header files exported by ROC Thunk
    export ROCT_INC_DIR=/opt/rocm/include/libhsakmt/

    # Contains library files exported by ROC Thunk
    export ROCT_LIB_DIR=/opt/rocm/lib/
    
    # Contains header files exported by ROC Runtime
    export HIP_INC_DIR=/opt/rocm/hip/include/hip/
    
    # Contains header files exported by rocBLAS
    export ROCBLAS_INC_DIR=$RVS/../build/rocBLAS/build/release/rocblas-install/include/
    
    # Contains library files exported by rocBLAS
    export ROCBLAS_LIB_DIR=$RVS/../build/rocBLAS/build/release/rocblas-install/lib/
    
    # Contains header files exported by rocm_smi
    export ROCM_SMI_INC_DIR=$RVS/rocm_smi_lib/include

    # Contains library files exported by rocm_smi
    export ROCM_SMI_LIB_DIR=$RVS/../build/rocm_smi_lib    

    cmake -DROCR_INC_DIR=$ROCR_INC_DIR -DROCR_LIB_DIR=$ROCR_LIB_DIR  -DROCBLAS_INC_DIR=$ROCBLAS_INC_DIR -DROCBLAS_LIB_DIR=$ROCBLAS_LIB_DIR -DHIP_INC_DIR=$HIP_INC_DIR  -DROCM_SMI_INC_DIR=$ROCM_SMI_INC_DIR -DROCM_SMI_LIB_DIR=$ROCM_SMI_LIB_DIR  ./ -B../build    
    cd ../build
    make

Run:

    sudo bin/rvs -d 3

Build package:

    cd $WB/build
    make package

_Note:_ based on your OS, only DEB or RPM package will be buile.

For CentOS specific instructions see this [link](./CentOS.md).

