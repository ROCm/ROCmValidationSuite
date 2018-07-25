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
    cmake ./rocm_smi_lib -B../build/rocm_smi_lib
    cd ../build/rocm_smi_lib
    make

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


    cmake -DROCR_INC_DIR=$ROCR_INC_DIR -DROCR_LIB_DIR=$ROCR_LIB_DIR ./ -B../build
    cd ../build
    make

Run:

    sudo bin/rvs -d 3

Build package:

    cd $WB/build
    make package

_Note:_ based on your OS, only DEB or RPM package will be buile.

For CentOS specific instructions see this [link](./CentOS.md).

