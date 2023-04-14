# ROCmValidationSuite

The ROCm Validation Suite (RVS) is a system administrator’s and cluster manager's tool for detecting and troubleshooting common problems affecting AMD GPU(s) running in a high-performance computing environment, enabled using the ROCm software stack on a compatible platform.

The RVS is a collection of tests, benchmarks and qualification tools each targeting a specific sub-system of the ROCm platform. All of the tools are implemented in software and share a common command line interface. Each set of tests are implemented in a “module” which is a library encapsulating the functionality specific to the tool. The CLI can specify the directory containing modules to use when searching for libraries to load. Each module may have a set of options that it defines and a configuration file that supports its execution.

The function of each module see this [link](./features.md).

Examples and about config files [link](./ug1main.md).

## Documentation

Run the steps below to build documentation locally.

```
cd docs

pip3 install -r .sphinx/requirements.txt

python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

## Prerequisites 
Please do this before compilation/installing compiled package.

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

## Install ROCm stack, rocblas and rocm-smi-lib
Install ROCm stack for Ubuntu/CentOS/SLES/RHEL, Refer https://github.com/RadeonOpenCompute/ROCm

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
If  rocm-smi-lib is already installed but "/opt/rocm/rocm_smi/ path doesn't exist. Do below:

Ubuntu : sudo dpkg -r rocm-smi-lib && sudo apt install rocm-smi-lib

CentOS & RHEL : sudo rpm -e  rocm-smi-lib && sudo yum install  rocm-smi-lib

SUSE : sudo rpm -e  rocm-smi-lib && sudo zypper install  rocm-smi-lib

## Building from Source
This section explains how to get and compile current development stream of RVS.

### Clone repository
    git clone https://github.com/ROCm-Developer-Tools/ROCmValidationSuite.git

### Configure and build RVS:

    cd ROCmValidationSuite
 If OS is Ubuntu and SLES, use cmake
    
     cmake  -DROCM_PATH=<rocm_installed_path> -DCMAKE_INSTALL_PREFIX=<rocm_installed_path> -DCMAKE_PACKAGING_INSTALL_PREFIX=<rocm_installed_path> ./ -B./build
     eg/- cmake -DROCM_PATH=/opt/rocm-4.0.0 -DCMAKE_INSTALL_PREFIX=/opt/rocm-4.0.0 -DCMAKE_PACKAGING_INSTALL_PREFIX=/opt/rocm-4.0.0 ./ -B./build
     
     make -C ./build
     
If OS is CentOS and RHEL, use cmake3

    cmake3  -DROCM_PATH=<rocm_installed_path> -DCMAKE_INSTALL_PREFIX=<rocm_installed_path> -DCMAKE_PACKAGING_INSTALL_PREFIX=<rocm_installed_path> ./ -B./build
 
    make -C ./build

### Build package:

     cd ./build
     
     make package

**Note:**_ based on your OS, only DEB or RPM package will be built. You may
ignore an error for the unrelated configuration

### Install package:

    Ubuntu : sudo dpkg -i rocm-validation-suite*.deb
    CentOS & RHEL & SUSE : sudo rpm -i --replacefiles --nodeps rocm-validation-suite*.rpm

**Note:**
RVS is getting packaged as part of ROCm release starting from 3.0. You can install pre-compiled package as below.
Please make sure Prerequisites, ROCm stack, rocblas and rocm-smi-lib64 are already installed

    Ubuntu : sudo apt install rocm-validation-suite
    CentOS & RHEL : sudo yum install rocm-validation-suite
    SUSE : sudo zypper install rocm-validation-suite

## Running RVS

### Running version built from source code:

    cd ./build/bin
    sudo ./rvs -d 3
    sudo ./rvsqa.new.sh ; It will run complete rvs test suite

### Running version pre-complied and packaged with ROCm release
sudo /opt/rocm/rvs/rvs -d 3

Similarly, you can run all tests as mentioned in "rvsqa.new.sh" script, present at "testscripts/rvsqa.new.sh"

## Regression

Simple regression has been implemented. You may find more about it
on this [link](./REGRESSION.md).

## Reporting

Test based reporting is enabled since beginning. 
Added json based reporting to gst and iet modules. To enable json logging use "-j" command line option.
./rvs -c conf/gst_sinle.conf -d 3 -j
the json location will be in /var/log folder and the name of the file will be printed in the stdout.
output structure is as shown below:
```

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
example for gst is:
```

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
}

```
