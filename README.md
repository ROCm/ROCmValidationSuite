# ROCmValidationSuite
The ROCm Validation Suite (RVS) is a system administrator’s and cluster manager's tool for detecting and troubleshooting common problems affecting AMD GPU(s) running in a high-performance computing environment, enabled using the ROCm software stack on a compatible platform.

The RVS is a collection of tests, benchmarks and qualification tools each targeting a specific sub-system of the ROCm platform. All of the tools are implemented in software and share a common command line interface. Each set of tests are implemented in a “module” which is a library encapsulating the functionality specific to the tool. The CLI can specify the directory containing modules to use when searching for libraries to load. Each module may have a set of options that it defines and a configuration file that supports its execution.

The function of each module see this [link](./FEATURES.md).

## Prerequisites

Ubuntu : 
      
        sudo apt-get -y update && sudo apt-get install -y libpci3 libpci-dev doxygen unzip cmake git

 CentOS : 
        
        sudo yum install -y cmake3 doxygen pciutils-devel rpm rpm-build git gcc-c++ 
 
 RHEL : 
        
        sudo yum install -y cmake3 doxygen rpm rpm-build git gcc-c++ 
        
        wget http://mirror.centos.org/centos/7/os/x86_64/Packages/pciutils-devel-3.5.1-3.el7.x86_64.rpm
        
        sudo rpm -ivh pciutils-devel-3.5.1-3.el7.x86_64.rpm
		
 SLES :  
		    
        sudo SUSEConnect -p sle-module-desktop-applications/15.1/x86_64
       
		      sudo SUSEConnect --product sle-module-development-tools/15.1/x86_64
       
		      sudo zypper  install -y cmake doxygen pciutils-devel libpci3 rpm git rpm-build gcc-c++ 

## Install ROCm stack, rocblas and rocm_smi64
Install ROCm stack for Ubuntu/CentOS, Refer https://github.com/RadeonOpenCompute/ROCm
 
Install rocBLAS and rocm_smi64 : 

   Ubuntu : 
   
          sudo apt-get install rocblas rocm_smi64
   
   CentOS & RHEL : 
            
            sudo yum install rocblas rocm_smi64
   
   SUSE : 
         
            sudo zypper install rocblas rocm_smi64

_**Note:**_
If  rocm_smi64 is already installed but "/opt/rocm/rocm_smi/ path doesn't exist. Do below:

Ubuntu : sudo dpkg -r rocm_smi64 && sudo apt install rocm_smi64

CentOS & RHEL : sudo rpm -e rocm_smi64 && sudo yum install rocm_smi64

SUSE : sudo rpm -e rocm_smi64 && sudo zypper install rocm_smi64

## Building from Source
This section explains how to get and compile current development stream of RVS.

### Clone repository
    git clone https://github.com/ROCm-Developer-Tools/ROCmValidationSuite.git

### Configure and build RVS:

    cd ROCmValidationSuite
 If OS is Ubuntu and SLES, use cmake
    
     cmake ./ -B./build
     
     make -C ./build
     
If OS is CentOS and RHEL, use cmake3

    cmake3 ./ -B./build
 
    make -C ./build

### Build package:

     cd ./build
     
     make package

**Note:**_ based on your OS, only DEB or RPM package will be built. You may
ignore an error for the unrelated configuration

### Install package:

    Ubuntu : sudo dpkg -i rocm-validation-suite*.deb
    CentOS & RHEL & SUSE : sudo rpm -i --replacefiles rocm-validation-suite*.rpm

## Running RVS

### Running version built from source code:

    cd ./build/bin
    sudo ./rvs -d 3
    sudo ./rvsqa.new.sh  ; It will run complete rvs test suite


## Regression

Simple regression has been implemented. You may find more about it
on this [link](./REGRESSION.md).
