
@section ugs1 1 Introduciton
The ROCm Validation Suite (RVS) is a system administrator’s and cluster
manager's tool for detecting and troubleshooting common problems affecting AMD
GPU(s) running in a high-performance computing environment, enabled using the
ROCm software stack on a compatible platform.

The RVS is a collection of tests, benchmarks and qualification tools each
targeting a specific sub-system of the ROCm platform. All of the tools are
implemented in software and share a common command line interface. Each set of
tests are implemented in a “module” which is a library encapsulating the f
unctionality specific to the tool. The CLI can specify the directory containing
modules to use when searching for libraries to load. Each module may have a set
of options that it defines and a configuration file that supports its execution.

@section usg2 2 Installing RVS

RVS cab be obtained by building it fro source code base or by installing from
pre-built package.

@subsection ugs21 2.1 Building from Source Code

RVS has been developed as open source solution. Its source code and belonging
documentation can be found at AMD's GitHub page.\n
In order to build RVS from source code please visit
[ROCm Validation Suite GitHub site](https://github.com/ROCm-Developer-Tools/ROCmValidationSuite)
and follow instructions in README file.

@subsection usg22 2.2 Installing from Package
Please download `rocm-validation-suite-$(RVSVER).deb` or `.rpt` file from AMD
site. Install package using your favorite package manager.

RVS components will be installed in `/opt/rocm/rvs`. Package contains:
- executable modules
- user guide (located in in _install-base_/userguide/html)
- man page (located in _install-base_/man)
- configuration examples (located in _install-base_/conf)

If needed, you may remove RVS package using you favorite package manager.


@section usg3 3 Basic Concepts

@subsection usg31 3.1 RVS Architecture

RVS is implemented as a set of modules each implementing particular test
functionality. Modules are invoked from one central place (aka Launcher) which
is responsible for reading input (command line and test configuration file),
loading and running appropriate modules and providing test output. RVS
architecture is built around concept of Linux shared objects, thus
allowing for easy addition of new modules in the future.


@subsection usg31a 3.2 Available Modules

@subsubsection usg31a01 3.2.1 GPU Properties – GPUP
The GPU Properties module queries the configuration of a target device and
returns the device’s static characteristics.\n
These static values can be used to debug issues such as device support,
performance and firmware problems.

@subsubsection usg31a02 3.2.2 GPU Monitor – GM module
The GPU monitor tool is capable of running on one, some or all of the GPU(s)
installed and will report various information at regular intervals. The module
can be configured to halt another RVS modules execution if one of the quantities exceeds a specified boundary value.

@subsubsection usg31a03 3.2.3 PCI Express State Monitor  – PESM module
The PCIe State Monitor tool is used to actively monitor the PCIe interconnect
between the host platform and the GPU. The module will register a “listener” on
a target GPU’s PCIe interconnect, and log a message whenever it detects a state change. The PESM will be able to detect the following state changes:

1.	PCIe link speed changes
2.	GPU power state changes

@subsubsection usg31a04 3.2.4 ROCm Configuration Qualification Tool - RCQT module
The ROCm Configuration Qualification Tool ensures the platform is capable of
running ROCm applications and is configured correctly. It checks the installed
versions of the ROCm components and the platform configuration of the system.
This includes checking that dependencies, corresponding to the associated
operating system and runtime environment, are installed correctly. Other
qualification steps include checking:

1.	The existence of the /dev/kfd device
2.	The /dev/kfd device’s permissions
3.	The existence of all required users and groups that support ROCm
4.	That the user mode components are compatible with the drivers, both the KFD and the amdgpu driver.
5.	The configuration of the runtime linker/loader qualifying that all ROCm libraries are in the correct search path.

@subsubsection usg31a05 3.2.5 PCI Express Qualification Tool – PEQT module
The PCIe Qualification Tool consists is used to qualify the PCIe bus on which
the GPU is connected. The qualification test will be capable of determining the
following characteristics of the PCIe bus interconnect to a GPU:

1.	Support for Gen 3 atomic completers
2.	DMA transfer statistics
3.	PCIe link speed
4.	PCIe link width

@subsubsection usg31a06 3.2.6 SBIOS Mapping Qualification Tool – SMQT module
The GPU SBIOS mapping qualification tool is designed to verify that a
platform’s SBIOS has satisfied the BAR mapping requirements for VDI and Radeon
Instinct products for ROCm support.

Refer to the “ROCm Use of Advanced PCIe Features and Overview of How BAR Memory
is Used In ROCm Enabled System” web page for more information about how BAR
memory is initialized by VDI and Radeon products.

@subsubsection usg31a07 3.2.7 P2P Benchmark and Qualification Tool – PBQT module
The P2P Benchmark and Qualification Tool  is designed to provide the list of all
GPUs that support P2P and characterize the P2P links between peers. In addition
to testing for P2P compatibility, this test will perform a peer-to-peer
throughput test between all P2P pairs for performance evaluation. The P2P
Benchmark and Qualification Tool will allow users to pick a collection of two or
more GPUs on which to run. The user will also be able to select whether or not
they want to run the throughput test on each of the pairs.

Please see the web page “ROCm, a New Era in Open GPU Computing” to find out more
about the P2P solutions available in a ROCm environment.

@subsubsection usg31a08 3.2.8 PCI Express Bandwidth Benchmark – PEBB module
The PCIe Bandwidth Benchmark attempts to saturate the PCIe bus with DMA
transfers between system memory and a target GPU card’s memory. The maximum
bandwidth obtained is reported to help debug low bandwidth issues. The
benchmark should be capable of  targeting one, some or all of the GPUs
installed in a platform, reporting individual benchmark statistics for each.

@subsubsection usg31a09 3.2.9 GPU Stress Test  - GST module
The GPU Stress Test runs a Graphics Stress test or SGEMM/DGEMM
(Single/Double-precision General Matrix Multiplication) workload on one, some or
all GPUs. The GPUs can be of the same or different types. The duration of the
benchmark should be configurable, both in terms of time (how long to run) and
iterations (how many times to run).

The test should be capable driving the power level equivalent to the rated TDP
of the card, or levels below that. The tool must be capable of driving cards at
TDP-50% to TDP-100%, in 10% incremental jumps. This should be controllable by
the user.

@subsubsection usg31a10 3.2.10 Input EDPp Test  - IET module
The Input EDPp Test generates EDP peak power on all input rails. This test is
used to verify if the system PSU is capable of handling the worst case power
spikes of the board.  Peak Current at defined period  =  1 minute moving
average power.


@subsection usg32 3.2 Configuration Files

The RVS tool will allow the user to indicate a configuration file, adhering to
the YAML 1.2 specification, which details the validation tests to run and the
expected results of a test, benchmark or configuration check.

The configuration
file used for an execution is specified using the `--config` option. The default
configuration file used for a run will be `rvs.conf`, which will include default
values for all defined tests, benchmarks and configurations checks, as well as
device specific configuration values. The format of the configuration files
determines the order in which actions are executed, and can provide the number
of times the test will be executed as well.

Configuration file is, in YAML terms, mapping of 'actions' keyword into
sequence of action items. Action items are themselves YAML keyed lists. Each
list consists of several _key:value_ pairs. Some keys may have values which
are keyed lists themselves (nested mappings).

Action item (or action for short) uses keys to define nature of validation test
to be performed. Each action has some common keys -- like 'name', 'module',
'deviceid' -- and test specific keys which depend on the module being used.

An example of RVS configuration file is given here:


    actions:
    - name: action_1
      device: all
      module: gpup
      properties:
        mem_banks_count:
      io_links-properties:
        version_major:
    - name: action_2
      module: gpup
      device: all
      properties:
        mem_banks_count:
    - name: action_3
    ...


@subsection usg33 3.3 Basic Configuration Keys


@subsection usg34 3.4 Command Line Options

@section usg4 4 RCQT Module


**Bold Test**
_Cursive text_

