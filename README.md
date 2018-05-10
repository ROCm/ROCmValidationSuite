# ROCmValidationSuite
The ROCm Validation Suite (RVS) is a system administrator’s and cluster manager's tool for detecting and troubleshooting common problems affecting AMD GPU(s) running in a high-performance computing environment, enabled using the ROCm software stack on a compatible platform.

The RVS is a collection of tests, benchmarks and qualification tools each targeting a specific sub-system of the ROCm platform. All of the tools are implemented in software and share a common command line interface. Each set of tests are implemented in a “module” which is a library encapsulating the functionality specific to the tool. The CLI can specify the directory containing modules to use when searching for libraries to load. Each module may have a set of options that it defines and a configuration file that supports its execution.

The function of each module see this [link](./FEATURES.md).

# RVS "Hello World"
This section explains how to get and compile current development stream of RVS.

Clone repository:

    cd /your/work/bench/folder
    export WB=$PWD
    git clone https://github.com/ROCm-Developer-Tools/ROCmValidationSuite.git
    git submodule init
    git submodule update
    
Init environment variables:

    export RVS=$WB/ROCmValidationSuite
    export LD_LIBRARY_PATH=$RVS/gpup.so
    
Compile: 

    cd $RVS
    cmake .
    make

Run:

    cd $RVS/rvs
    ./rvs