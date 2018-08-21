
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

1.  PCIe link speed changes
2.  GPU power state changes

@subsubsection usg31a04 3.2.4 ROCm Configuration Qualification Tool - RCQT module
The ROCm Configuration Qualification Tool ensures the platform is capable of
running ROCm applications and is configured correctly. It checks the installed
versions of the ROCm components and the platform configuration of the system.
This includes checking that dependencies, corresponding to the associated
operating system and runtime environment, are installed correctly. Other
qualification steps include checking:

1.  The existence of the /dev/kfd device
2.  The /dev/kfd device’s permissions
3.  The existence of all required users and groups that support ROCm
4.  That the user mode components are compatible with the drivers, both the KFD and the amdgpu driver.
5.  The configuration of the runtime linker/loader qualifying that all ROCm libraries are in the correct search path.

@subsubsection usg31a05 3.2.5 PCI Express Qualification Tool – PEQT module
The PCIe Qualification Tool consists is used to qualify the PCIe bus on which
the GPU is connected. The qualification test will be capable of determining the
following characteristics of the PCIe bus interconnect to a GPU:

1.  Support for Gen 3 atomic completers
2.  DMA transfer statistics
3.  PCIe link speed
4.  PCIe link width

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


@subsection usg33 3.3 Common Configuration Keys

Common configuration keys applicable to most module are summarized in the
table below:\n
<table>
<tr><th>Config Key</th> <th>Type</th><th> Description</th></tr>
<tr><td>name</td><td>String</td><td>The name of the defined action.</td></tr>
<tr><td>device</td><td>Collection of String</td><td>This is a list of device indexes
(gpu ids), or the
keyword “all”. The defined actions will be
executed on the specified device, as long as
the action targets a device specifically (some
are platform actions). If an invalid device id
value or no value is specified the tool will report that the device was not
found and terminate execution, returning an
error regarding the configuration file.</td></tr>

<tr><td>deviceid</td><td>Integer</td><td>This is an optional parameter, but if
specified it restricts the action to a specific device type
corresponding to the deviceid.</td></tr>
<tr><td>parallel</td><td>Bool</td><td>If this key is false, actions will be run
on one device at a time, in the order specified in the device list, or the
natural ordering if the device value is “all”. If this parameter is true,
actions will be run on all specified devices in parallel. If a value isn’t
specified the default value is false.</td></tr>

<tr><td>count</td><td>Integer</td><td>This specifies number of times to execute
the action. If the value is 0, execution will continue indefinitely. If a value
isn’t specified the default will be 1. Some modules will ignore this
parameter.</td></tr>

<tr><td>wait</td><td>Integer</td><td>This indicates how long the test should wait
between executions, in milliseconds. Some
modules will ignore this parameter. If the
count key is not specified, this key is ignored.
duration Integer This parameter overrides the count key, if
specified. This indicates how long the test
should run, given in milliseconds. Some
modules will ignore this parameter.</td></tr>


<tr><td>module</td><td>String</td><td>This parameter specifies the module that
will be used in the execution of the action. Each module has a set of sub-tests
or sub-actions that can be configured based on its specific parameters.</td></tr>
</table>

@subsection usg34 3.4 Command Line Options

Command line options are summarized in the table below:

<table>
<tr><th>Short option</th><th>Long option</th><th> Description</th></tr>
<tr><td>-a</td><td>\-\-appendLog</td><td>When generating a debug logfile,
do not overwrite the contents
of a current log. Used in conjuction with the -d and -l options.
</td></tr>

<tr><td>-c</td><td>\-\-config</td><td>Specify the configuration file to be used.
The default is \<installbase\>/RVS/conf/RVS.conf
</td></tr>

<tr><td></td><td>\-\-configless</td><td>Run RVS in a configless mode.
Executes a "long" test on all supported GPUs.</td></tr>

<tr><td>-d</td><td>\-\-debugLevel</td><td>Specify the debug level for the output
log. The range is 0 to 5 with 5 being the most verbose.
Used in conjunction with the -l flag.</td></tr>

<tr><td>-g</td><td>\-\-listGpus</td><td>List the GPUs available and exit.
This will only list GPUs that are supported by RVS.</td></tr>

<tr><td>-i</td><td>\-\-indexes</td><td>Comma separated list of  devices to run
RVS on. This will override the device values specified in the configuration file
for every action in the configuration file, including the "all" value.</td></tr>

<tr><td>-j</td><td>\-\-json</td><td>Output should use the JSON format.</td></tr>

<tr><td>-l</td><td>\-\-debugLogFile</td><td>Specify the logfile for debug
information. This will produce a log file intended for post-run analysis after
an error.</td></tr>

<tr><td></td><td>\-\-quiet</td><td>No console output given. See logs and return
code for errors.</td></tr>

<tr><td>-m</td><td>\-\-modulepath</td><td>Specify a custom path for the RVS
modules.</td></tr>

<tr><td></td><td>\-\-specifiedtest</td><td>Run a specific test in a configless
mode. Multiple word tests should be in quotes. This action will default to all
devices, unless the \-\-indexes option is specifie.</td></tr>

<tr><td>-t</td><td>\-\-listTests</td><td>List the modules available to be
executed through RVS and exit. This will list only the readily loadable modules
given the current path and library conditions.</td></tr>

<tr><td>-v</td><td>\-\-verbose</td><td>Enable verbose reporting. This is
equivalent to specifying the -d 5 option.</td></tr>

<tr><td></td><td>\-\-version</td><td>Displays the version information and exits.
</td></tr>

<tr><td>-h</td><td>\-\-help</td><td>Display usage information and exit.
</td></tr>

</table>

@section usg4 4 RCQT Module

@section usg5 5 GST Module
@subsection usg51 5.1 Configuration keys
@verbatim
-----------------------------------------------------------------------------------------------------------------
Config key         Type         Description
-----------------------------------------------------------------------------------------------------------------
target_stress      Float        The maximum relative performance the GPU will attempt to achieve in gigaflops.
                                This parameter is required.

copy_matrix        Bool         This parameter indicates if each operation should copy the matrix data to the GPU
                                before executing. The default value is true.

ramp_interval      Integer      This is an time interval, specified in milliseconds, given to the test to reach the 
                                given target_stress gigaflops. The default value is 5000 (5 seconds). This time
                                is counted against the duration of the test. If the target gflops, or stress, is not 
                                achieved in this time frame, the test will fail. If the target stress (gflops) 
                                is achieved the test will attempt to run for the rest of the duration specified 
                                by the action, sustaining the stress load during that time.

tolerance          Float        A value indicating how much the target_stress can fluctuate after the ramp period 
                                for the test to succeed. The default value is 0.1 or 10%.

max_violations     Integer      The number of tolerance violations that can occur after the ramp_interval for 
                                the test to still pass. The default value is 0.

log_interval       Integer      If this key is specified informational messages will be emitted at the given
                                interval providing the current values of all parameters that have a bound set.
                                The units for the log_interval are milliseconds. The default value is 1000 (1 second).

matrix_size        Integer      Size of the matrices of the SGEMM operations. The default value is 5760.
@endverbatim

@subsection usg52 5.2 Output keys
@verbatim
-----------------------------------------------------------------------------------------------------------------
Output key         Type         Description
-----------------------------------------------------------------------------------------------------------------
target_stress      Time Series  The average gflops over the last log interval.
                   Float        

max_gflops         Float        The maximum sustained performance obtained by the GPU during the test.

stress_violations  Integer      The number of gflops readings that violated the tolerance of the test after 
                                the ramp interval.

flops_per_op       Integer      Flops (floating point operations) per operation queued to the GPU queue.
                                One operation is one call to SGEMM/DGEMM.

bytes_copied_      Integer      Calculated number of ops/second necessary to achieve target gigaflops.
per_op
 
try_ops_per_sec    Float        Calculated number of ops/second necessary to achieve target gigaflops.

pass               Bool         If the GPU achieves its desired sustained performance level                         
@endverbatim



@subsection usg53 5.3 Configuration files examples

When running the __GST__ module, users should provide at least an action name, the module name (gst), a list of GPU IDs, the test duration and a target stress value (gigaflops). Thus, the most basic configuration file looks like this: 


    actions:
    - name: action_gst_1
      module: gst
      device: all
      target_stress: 3500
      duration: 8000

For the above configuration file, all the missing configuration keys will have their default
values (e.g.: __copy_matrix=true__, __matrix_size=5760__ etc.). For more information about the default 
values please consult the dedicated sections (__3.3 Common Configuration Keys__ and __5.1 Configuration keys__). 
    
When the __RVS__ tool runs against such a configuration file, it will do the following:
  - run the stress test on all available (and compatible) AMD GPUs, one after the other
  - log a start message containing the GPU ID, the __target_stress__ and the value of the __copy_matrix__<br />
e.g.: __[INFO  ] [164337.932824] action_gst_1 gst 50599 start 3500.000000 copy matrix:true__
  - emit, each __log_interval__ (e.g.: 1000ms), a message containing the gigaflops value that the current GPU achieved<br />
e.g.: __[INFO  ] [164355.111207] action_gst_1 gst 33367 Gflops 3535.670231__
  - log a message as soon as the current GPU reaches the given __target_stress__<br />
e.g.: __[INFO  ] [164350.804843] action_gst_1 gst 33367 target achieved 3500.000000__
  - log a __ramp time exceeded__ message if the GPU was not able to reach the __target_stress__ in the __ramp_interval__ time frame (e.g.: 5000). In such a case, the test will also terminate<br />
e.g.: __[INFO  ] [164013.788870] action_gst_1 gst 3254 ramp time exceeded 5000__
  - log the test result, when the stress test completes. The message contains the test's overall result and some other statistics according to __5.2 Output keys__<br />
e.g.: __[RESULT] [164355.647523] action_gst_1 gst 33367 Gflop: 4066.020766 flops_per_op: 382.205952x1e9 bytes_copied_per_op: 398131200 try_ops_per_sec: 9.157367 pass: TRUE__ 
  - log a __stress violation__ message when the current gigaflops (for the last __log_interval__, e.g.; 1000ms) violates the bounds set by the __tolerance__ configuration key (e.g.: 0.1). Please note that this message is not logged during the __ramp_interval__ time frame<br />
e.g.: __[INFO  ] [164013.788870] action_gst_1 gst 3254 stress violation 2500__


If a mandatory configuration key is missing, the __RVS__ tool will log an error message and terminate the executation of the current module. For example, the following configuration file will cause the __RVS__ to terminate with the following error message:<br /> 
__RVS-GST: action: action_gst_1  key 'target_stress' was not found__

    actions:
    - name: action_gst_1
      module: gst
      device: all      
      duration: 8000

A more complex configuration file looks like this:

    actions:
    - name: action_1
      device: 50599 33367
      module: gst
      parallel: false
      count: 12
      wait: 100
      duration: 7000
      ramp_interval: 3000
      log_interval: 1000
      max_violations: 2
      copy_matrix: false
      target_stress: 5000
      tolerance: 0.07
      matrix_size: 5760    

For this configuration file, the RVS tool:
  - will run the stress test only for the GPUs having the ID 50599 or 33367. To get all the available GPU IDs, run __RVS__ tool with __-g__ option
  - will run the test on the selected GPUs, one after the other
  - will run each test, 12 times
  - will only copy the matrices to the GPUs at the beginning of the test
  - will wait 100ms before each test execution
  - will try to reach 5000 gflops in maximum 3000ms
  - if __target_stress__ (5000) is achieved in the __ramp_interval__ (3000 ms) it will attempt to run the test for the rest of the duration, sustaining the stress load during that time     
  - will allow a 7% __target_stress__ __tolerance__ (each __target_stress__ violation will generate a __stress violation__ message as shown in the first example)
  - will allow only 2 __target_stress__ violations. Exceeding the __max_violations__ will not terminate the test, but the __RVS__ will mark the test result as "fail". 

The output for such a configuration key may look like this:

__[INFO  ] [172061.758830] action_1 gst 50599 start 5000.000000 copy matrix:false__<br />
__[INFO  ] [172063.547668] action_1 gst 50599 Gflops 6471.614725__<br />
__[INFO  ] [172064.577715] action_1 gst 50599 target achieved 5000.000000__<br />
__[INFO  ] [172065.609224] action_1 gst 50599 Gflops 5189.993529__<br />
__[INFO  ] [172066.634360] action_1 gst 50599 Gflops 5220.373979__<br />
__[INFO  ] [172067.659262] action_1 gst 50599 Gflops 5225.472000__<br />
__[INFO  ] [172068.694305] action_1 gst 50599 Gflops 5169.935583__<br />
__[RESULT] [172069.573967] action_1 gst 50599 Gflop: 6471.614725 flops_per_op: 382.205952x1e9 bytes_copied_per_op: 398131200 try_ops_per_sec: 13.081952 pass: TRUE__<br />
__[INFO  ] [172069.574369] action_1 gst 33367 start 5000.000000 copy matrix:false__<br />
__[INFO  ] [172071.409483] action_1 gst 33367 Gflops 6558.348080__<br />
__[INFO  ] [172072.438104] action_1 gst 33367 target achieved 5000.000000__<br />
__[INFO  ] [172073.465033] action_1 gst 33367 Gflops 5215.285895__<br />
__[INFO  ] [172074.501571] action_1 gst 33367 Gflops 5164.945297__<br />
__[INFO  ] [172075.529468] action_1 gst 33367 Gflops 5210.207720__<br />
__[INFO  ] [172076.558102] action_1 gst 33367 Gflops 5205.139424__<br />
__[RESULT] [172077.448182] action_1 gst 33367 Gflop: 6558.348080 flops_per_op: 382.205952x1e9 bytes_copied_per_op: 398131200 try_ops_per_sec: 13.081952 pass: TRUE__<br />

When setting the __parallel__ to false, the __RVS__ will run the stress tests on all selected GPUs in parallel and the output may look like this:

__[INFO  ] [173381.407428] action_1 gst 50599 start 5000.000000 copy matrix:false__<br />
__[INFO  ] [173381.407744] action_1 gst 33367 start 5000.000000 copy matrix:false__<br />
__[INFO  ] [173383.245771] action_1 gst 33367 Gflops 6558.348080__<br />
__[INFO  ] [173383.256935] action_1 gst 50599 Gflops 6484.532120__<br />
__[INFO  ] [173384.274202] action_1 gst 33367 target achieved 5000.000000__<br />
__[INFO  ] [173384.286014] action_1 gst 50599 target achieved 5000.000000__<br />
__[INFO  ] [173385.301038] action_1 gst 33367 Gflops 5215.285895__<br />
__[INFO  ] [173385.315794] action_1 gst 50599 Gflops 5200.080980__<br />
__[INFO  ] [173386.337638] action_1 gst 33367 Gflops 5164.945297__<br />
__[INFO  ] [173386.353274] action_1 gst 50599 Gflops 5159.964636__<br />
__[INFO  ] [173387.365494] action_1 gst 33367 Gflops 5210.207720__<br />
__[INFO  ] [173387.383437] action_1 gst 50599 Gflops 5195.032357__<br />
__[INFO  ] [173388.401250] action_1 gst 33367 Gflops 5169.935583__<br />
__[INFO  ] [173388.421599] action_1 gst 50599 Gflops 5154.993572__<br />
__[RESULT] [173389.282710] action_1 gst 33367 Gflop: 6558.348080 flops_per_op: 382.205952x1e9 bytes_copied_per_op: 398131200 try_ops_per_sec: 13.081952 pass: TRUE__<br />
__[RESULT] [173389.305479] action_1 gst 50599 Gflop: 6484.532120 flops_per_op: 382.205952x1e9 bytes_copied_per_op: 398131200 try_ops_per_sec: 13.081952 pass: TRUE__<br />

It is important that all the configuration keys will be adjusted/fine-tuned according to the actual GPUs and HW platform capabilities. For example, a matrix size of 5760 should fit the VEGA 10 GPUs while 8640 should work with the VEGA 20 GPUs.









 
 



  
