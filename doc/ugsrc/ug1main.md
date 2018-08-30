
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
[ROCm Validation Suite GitHub
site](https://github.com/ROCm-Developer-Tools/ROCmValidationSuite)
and follow instructions in README file.

@subsection usg22 2.2 Installing from Package
Please download `rocm-validation-suite-$(RVSVER).deb` or `.rpt` file from AMD
site. Install package using your favorite package manager.

RVS components is installed in `/opt/rocm/rvs`. Package contains:
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
can be configured to halt another RVS modules execution if one of the quantities
exceeds a specified boundary value.
@subsubsection usg31a03 3.2.3 PCI Express State Monitor  – PESM module
The PCIe State Monitor tool is used to actively monitor the PCIe interconnect
between the host platform and the GPU. The module will register a “listener” on
a target GPU’s PCIe interconnect, and log a message whenever it detects a state
change. The PESM is able to detect the following state changes:

1.  PCIe link speed changes
2.  GPU power state changes

@subsubsection usg31a04 3.2.4 ROCm Configuration Qualification Tool - RCQT
module
The ROCm Configuration Qualification Tool ensures the platform is capable of
running ROCm applications and is configured correctly. It checks the installed
versions of the ROCm components and the platform configuration of the system.
This includes checking that dependencies, corresponding to the associated
operating system and runtime environment, are installed correctly. Other
qualification steps include checking:

1.  The existence of the /dev/kfd device
2.  The /dev/kfd device’s permissions
3.  The existence of all required users and groups that support ROCm
4.  That the user mode components are compatible with the drivers, both the KFD
and the amdgpu driver.
5.  The configuration of the runtime linker/loader qualifying that all ROCm
libraries are in the correct search path.

@subsubsection usg31a05 3.2.5 PCI Express Qualification Tool – PEQT module
The PCIe Qualification Tool consists is used to qualify the PCIe bus on which
the GPU is connected. The qualification test is capable of determining the
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
configuration file used for a run is `rvs.conf`, which will include default
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
<tr><td>device</td><td>Collection of String</td>
<td>This is a list of device indexes (gpu ids), or the keyword “all”. The
defined actions will be executed on the specified device, as long as the action
targets a device specifically (some are platform actions). If an invalid device
id value or no value is specified the tool will report that the device was not
found and terminate execution, returning an error regarding the configuration
file.</td></tr>

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
isn’t specified the default is 1. Some modules will ignore this
parameter.</td></tr>

<tr><td>wait</td><td>Integer</td><td>This indicates how long the test should
wait
between executions, in milliseconds. Some
modules will ignore this parameter. If the
count key is not specified, this key is ignored.
duration Integer This parameter overrides the count key, if
specified. This indicates how long the test
should run, given in milliseconds. Some
modules will ignore this parameter.</td></tr>


<tr><td>module</td><td>String</td><td>This parameter specifies the module that
will be used in the execution of the action. Each module has a set of sub-tests
or sub-actions that can be configured based on its specific
parameters.</td></tr>
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

@section usg4 4 GPUP Module
The GPU properties module provides an interface to easily dump the static
characteristics of a GPU. This information is stored in the sysfs file system
for the kfd, with the following path:

    /sys/class/kfd/kfd/topology/nodes/<node id>

Each of the GPU nodes in the directory is identified with a number,
indicating the device index of the GPU. This module will ignore count, duration
or wait key values.

@subsection usg41 4.1 Module Specific Keys
<table>
<tr><th>Config Key</th> <th>Type</th><th> Description</th></tr>
<tr><td>properties</td><td>Collection of Strings</td>
<td>The properties key specifies what configuration property or properties the
query is interested in. Possible values are:\n
all - collect all settings\n
gpu_id\n
cpu_cores_count\n
simd_count\n
mem_banks_count\n
caches_count\n
io_links_count\n
cpu_core_id_base\n
simd_id_base\n
max_waves_per_simd\n
lds_size_in_kb\n
gds_size_in_kb\n
wave_front_size\n
array_count\n
simd_arrays_per_engine\n
cu_per_simd_array\n
simd_per_cu\n
max_slots_scratch_cu\n
vendor_id\n
device_id\n
location_id\n
drm_render_minor\n
max_engine_clk_fcompute\n
local_mem_size\n
fw_version\n
capability\n
max_engine_clk_ccompute\n
</td></tr>
<tr><td>io_links-properties</td><td>Collection of Strings</td>
<td>The properties key specifies what configuration
property or properties the query is interested in.
Possible values are:\n
all - collect all settings\n
count - the number of io_links\n
type\n
version_major\n
version_minor\n
node_from\n
node_to\n
weight\n
min_latency\n
max_latency\n
min_bandwidth\n
max_bandwidth\n
recommended_transfer_size\n
flags\n
</td></tr>
</table>
@subsection usg42 4.2 Output

Module specific output keys are described in the table below:
<table>
<tr><th>Output Key</th> <th>Type</th><th> Description</th></tr>
<tr><td>properties-values</td><td>Collection of Integers</td>
<td>The collection will contain a positive integer value for each of the valid
properties specified in the properties config key.</td></tr>
<tr><td>io_links-propertiesvalues</td><td>Collection of Integers</td>
<td>The collection will contain a positive integer value for each of the valid
properties specified in the io_links-properties config key.</td></tr>
</table>
Each of the settings specified has a positive integer value. For each
setting requested in the properties key a message with the following format will
be returned:

    [RESULT][<timestamp>][<action name>] gpup <gpu id> <property> <property value>

For each setting in the io_links-properties key a message with the following
format will be returned:

    [RESULT][<timestamp>][<action name>] gpup <gpu id> <io_link id> <property> <property value>

@subsection usg43 4.3 Examples

**Example 1:**

Consider action>

    actions:
    - name: action_1
      device: all
      module: gpup
      properties:
        all:
      io_links-properties:
        all:

Action will display all properties for all compatible GPUs present in the
system. Output for such configuration may be like this:

    [RESULT] [597737.498442] action_1 gpup 3254 cpu_cores_count 0
    [RESULT] [597737.498517] action_1 gpup 3254 simd_count 256
    [RESULT] [597737.498558] action_1 gpup 3254 mem_banks_count 1
    [RESULT] [597737.498598] action_1 gpup 3254 caches_count 96
    [RESULT] [597737.498637] action_1 gpup 3254 io_links_count 1
    [RESULT] [597737.498680] action_1 gpup 3254 cpu_core_id_base 0
    [RESULT] [597737.498725] action_1 gpup 3254 simd_id_base 2147487744
    [RESULT] [597737.498768] action_1 gpup 3254 max_waves_per_simd 10
    [RESULT] [597737.498812] action_1 gpup 3254 lds_size_in_kb 64
    [RESULT] [597737.498856] action_1 gpup 3254 gds_size_in_kb 0
    [RESULT] [597737.498901] action_1 gpup 3254 wave_front_size 64
    [RESULT] [597737.498945] action_1 gpup 3254 array_count 4
    [RESULT] [597737.498990] action_1 gpup 3254 simd_arrays_per_engine 1
    [RESULT] [597737.499035] action_1 gpup 3254 cu_per_simd_array 16
    [RESULT] [597737.499081] action_1 gpup 3254 simd_per_cu 4
    [RESULT] [597737.499128] action_1 gpup 3254 max_slots_scratch_cu 32
    [RESULT] [597737.499175] action_1 gpup 3254 vendor_id 4098
    [RESULT] [597737.499222] action_1 gpup 3254 device_id 26720
    [RESULT] [597737.499270] action_1 gpup 3254 location_id 8960
    [RESULT] [597737.499318] action_1 gpup 3254 drm_render_minor 128
    [RESULT] [597737.499369] action_1 gpup 3254 max_engine_clk_ccompute 2200
    [RESULT] [597737.499419] action_1 gpup 3254 local_mem_size 17163091968
    [RESULT] [597737.499468] action_1 gpup 3254 fw_version 405
    [RESULT] [597737.499518] action_1 gpup 3254 capability 8832
    [RESULT] [597737.499569] action_1 gpup 3254 max_engine_clk_ccompute 2200
    [RESULT] [597737.499633] action_1 gpup 3254 0 count 1
    [RESULT] [597737.499675] action_1 gpup 3254 0 type 2
    [RESULT] [597737.499695] action_1 gpup 3254 0 version_major 0
    [RESULT] [597737.499716] action_1 gpup 3254 0 version_minor 0
    [RESULT] [597737.499736] action_1 gpup 3254 0 node_from 4
    [RESULT] [597737.499763] action_1 gpup 3254 0 node_to 1
    [RESULT] [597737.499783] action_1 gpup 3254 0 weight 20
    [RESULT] [597737.499808] action_1 gpup 3254 0 min_latency 0
    [RESULT] [597737.499830] action_1 gpup 3254 0 max_latency 0
    [RESULT] [597737.499853] action_1 gpup 3254 0 min_bandwidth 0
    [RESULT] [597737.499878] action_1 gpup 3254 0 max_bandwidth 0
    [RESULT] [597737.499902] action_1 gpup 3254 0 recommended_transfer_size 0
    [RESULT] [597737.499927] action_1 gpup 3254 0 flags 1
    [RESULT] [597737.500208] action_1 gpup 50599 cpu_cores_count 0
    [RESULT] [597737.500254] action_1 gpup 50599 simd_count 256
    ...
    [RESULT] [597737.501603] action_1 gpup 50599 0 recommended_transfer_size 0
    [RESULT] [597737.501626] action_1 gpup 50599 0 flags 1
    [RESULT] [597737.501877] action_1 gpup 33367 cpu_cores_count 0
    [RESULT] [597737.501921] action_1 gpup 33367 simd_count 256
    ...
    [RESULT] [597737.503258] action_1 gpup 33367 0 recommended_transfer_size 0
    [RESULT] [597737.503282] action_1 gpup 33367 0 flags 1
    ...

**Example 2:**

Consider action:

    actions:
    - name: action_1
      device: all
      module: gpup
      properties:
        simd_count:
        mem_banks_count:
        io_links_count:
        vendor_id:
        device_id:
        location_id:
        max_engine_clk_ccompute:
      io_links-properties:
        version_major:
        type:
        version_major:
        version_minor:
        node_from:
        node_to:
        recommended_transfer_size:
        flags:

This action explicitely lists some of the properties.
Output for such configuration may be:

    [RESULT] [597868.690637] action_1 gpup 3254 device_id 26720
    [RESULT] [597868.690713] action_1 gpup 3254 io_links_count 1
    [RESULT] [597868.690766] action_1 gpup 3254 location_id 8960
    [RESULT] [597868.690819] action_1 gpup 3254 max_engine_clk_ccompute 2200
    [RESULT] [597868.690862] action_1 gpup 3254 mem_banks_count 1
    [RESULT] [597868.690903] action_1 gpup 3254 simd_count 256
    [RESULT] [597868.690950] action_1 gpup 3254 vendor_id 4098
    [RESULT] [597868.691029] action_1 gpup 3254 0 flags 1
    [RESULT] [597868.691053] action_1 gpup 3254 0 node_from 4
    [RESULT] [597868.691075] action_1 gpup 3254 0 node_to 1
    [RESULT] [597868.691099] action_1 gpup 3254 0 recommended_transfer_size 0
    [RESULT] [597868.691119] action_1 gpup 3254 0 type 2
    [RESULT] [597868.691138] action_1 gpup 3254 0 version_major 0
    [RESULT] [597868.691158] action_1 gpup 3254 0 version_minor 0
    [RESULT] [597868.691425] action_1 gpup 50599 device_id 26720
    [RESULT] [597868.691469] action_1 gpup 50599 io_links_count 1
    [RESULT] [597868.691517] action_1 gpup 50599 location_id 17152
    ...
    [RESULT] [597868.692159] action_1 gpup 33367 device_id 26720
    [RESULT] [597868.692204] action_1 gpup 33367 io_links_count 1
    [RESULT] [597868.692252] action_1 gpup 33367 location_id 25344
    ...
    [RESULT] [597868.692619] action_1 gpup 33367 0 version_minor 0

**Example 3:**

Consider this action:

    actions:
    - name: action_1
      device: all
      module: gpup
      deviceid: 267
      properties:
        all:
      io_links-properties:
        all:

Action lists deviceid 267 which is not present in the system.
Output for such configuration is:

    RVS-GPUP: action: action_1  invalid 'deviceid' key value




@section usg5 5 GM Module
The GPU monitor module can be used monitor and characterize the response of a
GPU to different levels of use. This module is intended to run concurrently with
other actions, and provides a ‘start’ and ‘stop’ configuration key to start the
monitoring and then stop it after testing has completed. The module can also be
configured with bounding box values for interested GPU parameters. If any of the
GPU’s parameters exceed the bounding values on a specific GPU an INFO warning
message will be printed to stdout while the bounding value is still exceeded.

@subsection usg51 5.1 Module Specific Keys

<table>
<tr><th>Config Key</th> <th>Type</th><th> Description</th></tr>
<tr><td>monitor</td><td>Bool</td>
<td>If this this key is set to true, the GM module will start monitoring on
specified devices. If this key is set to false, all other keys are ignored and
monitoring of the specified device will be stopped.</td></tr>
<tr><td>metrics</td>
<td>Collection of Structures, specifying the metric, if there are bounds and the
bound values. The structures have the following format:\n{String, Bool, Integer,
Integer}</td>
<td>The set of metrics to monitor during the monitoring period. Example values
are:\n{‘temp’, ‘true’, max_temp, min_temp}\n {‘clock’, ‘false’, max_clock,
min_clock}\n {‘mem_clock’, ‘true’, max_mem_clock, min_mem_clock}\n {‘fan’,
‘true’, max_fan, min_fan}\n {‘power’, ‘true’, max_power, min_power}\n The set of
upper bounds for each metric are specified as an integer. The units and values
for each metric are:\n temp - degrees Celsius\n clock - MHz \n mem_clock - MHz
\n fan - Integer between 0 and 255 \n power - Power in Watts</td></tr>
<tr><td>sample_interval</td><td>Integer</td>
<td>If this key is specified metrics will be sampled at the given rate. The
units for the sample_interval are milliseconds. The default value is 1000.
</td></tr>
<tr><td>log_interval</td><td>Integer</td>
<td>If this key is specified informational messages will be emitted at the given
interval, providing the current values of all parameters specified. This
parameter must be equal to or greater than the sample rate. If this value is not
specified, no logging will occur.</td></tr>
<tr><td>terminate</td><td>Bool</td> <td>If the terminate key is true the GM
monitor will terminate the RVS process when a bounds violation is encountered on
any of the metrics specified.</td></tr>
</table>

@subsection usg52 5.2 Output

Module specific output keys are described in the table below:
<table>
<tr><th>Output Key</th> <th>Type</th><th> Description</th></tr>
<tr><td>metric_values</td><td>Time Series Collection of Result
Integers</td><td>A collection of integers containing the result values for each
of the metrics being monitored. </td></tr>
<tr><td>metric_violations</td><td>Collection of Result Integers </td><td>A
collection of integers containing the violation count for each of the metrics
being monitored. </td></tr>
<tr><td>metric_average</td><td>Collection of Result Integers </td><td></td></tr>
</table>

When monitoring is started for a target GPU, a result message is logged
with the following format:

    [RESULT][<timestamp>][<action name>] gm <gpu id> started

In addition, an informational message is provided for each for each metric
being monitored:

    [INFO ][<timestamp>][<action name>] gm <gpu id> monitoring < metric> bounds min:<min_metric> max: <max_metric>

During the monitoring informational output regarding the metrics of the GPU will
be sampled at every interval specified by the sample_rate key. If a bounding box
violation is discovered during a sampling interval, a warning message is
logged with the following format:

    [INFO ][<timestamp>][<action name>] gm <gpu id> < metric> bounds violation <metric value>

If the log_interval value is set an information message for each metric is
logged at every interval using the following format:

    [INFO ][<timestamp>][<action name>] gm <gpu id> < metric> <metric_value>

When monitoring is stopped for a target GPU, a result message is logged
with the following format:

    [RESULT][<timestamp>][<action name>] gm <gpu id> gm stopped

The following messages, reporting the number of metric violations that were
sampled over the duration of the monitoring and the average metric value is
reported:

    [RESULT][<timestamp>][<action name>] gm <gpu id> <metric> violations <metric_violations>
    [RESULT][<timestamp>][<action name>] gm <gpu id> <metric> average <metric_average>

@subsection usg53 5.3 Examples

**Example 1:**

Consider action:

    actions:
    - name: action_1
      module: gm
      device: all
      monitor: true
      metrics:
        temp: true 20 0
        fan: true 10 0
      duration: 5000
    - name: another_action
    ...

This action will monitor temperature and fan speed for 5 seconds and then continue
with the next action. Output for such configuration may be:

    [RESULT] [694381.521373] [action_1] gm 33367 started
    [INFO  ] [694381.531803] action_1 gm 33367  monitoring temp bounds min:0 max:20
    [INFO  ] [694381.531817] action_1 gm 33367  monitoring temp bounds min:0 max:20
    [INFO  ] [694381.531828] action_1 gm 33367  monitoring fan bounds min:0 max:10
    [RESULT] [694381.521373] [action_1] gm 3254 started
    [INFO  ] [694381.532257] action_1 gm 3254  monitoring temp bounds min:0 max:20
    [INFO  ] [694381.532276] action_1 gm 3254  monitoring temp bounds min:0 max:20
    [INFO  ] [694381.532293] action_1 gm 3254  monitoring fan bounds min:0 max:10
    [RESULT] [694381.521373] [action_1] gm 50599 started
    [INFO  ] [694381.534471] action_1 gm 50599  monitoring temp bounds min:0 max:20
    [INFO  ] [694381.534487] action_1 gm 50599  monitoring temp bounds min:0 max:20
    [INFO  ] [694381.534502] action_1 gm 50599  monitoring fan bounds min:0 max:10
    [INFO  ] [694381.534623] action_1 gm 33367 temp  bounds violation 22C
    [INFO  ] [694381.534822] action_1 gm 3254 temp  bounds violation 22C
    [INFO  ] [694381.534946] action_1 gm 50599 temp  bounds violation 22C
    [INFO  ] [694382.535329] action_1 gm 33367 temp  bounds violation 22C
    ...
    [INFO  ] [694385.537777] action_1 gm 50599 temp  bounds violation 21C
    [RESULT] [694386.538037] [action_1] gm 3254 stopped
    [RESULT] [694386.538037] [action_1] gm 50599 stopped
    [RESULT] [694386.538037] [action_1] gm 33367 stopped
    [RESULT] [694386.521449] [action_1] gm 3254 temp violations 1
    [RESULT] [694386.521449] [action_1] gm 3254 temp average 19C
    [RESULT] [694386.521449] [action_1] gm 3254 fan violations 0
    [RESULT] [694386.521449] [action_1] gm 3254 fan average 0%
    [RESULT] [694386.521449] [action_1] gm 50599 temp violations 5
    [RESULT] [694386.521449] [action_1] gm 50599 temp average 21C
    [RESULT] [694386.521449] [action_1] gm 50599 fan violations 0
    [RESULT] [694386.521449] [action_1] gm 50599 fan average 0%
    [RESULT] [694386.521449] [action_1] gm 33367 temp violations 5
    [RESULT] [694386.521449] [action_1] gm 33367 temp average 22C
    [RESULT] [694386.521449] [action_1] gm 33367 fan violations 0
    [RESULT] [694386.521449] [action_1] gm 33367 fan average 0%

**Example 2:**

Consider action:

    actions:
    - name: action_1
      module: gm
      device: all
      monitor: true
      metrics:
        temp: true 20 0
        fan: true 10 0
        power: true 100 0
      sample_interval: 1000
      log_interval: 1200
      terminate: false
      duration: 5000

This configuration is similar to that in *Example 1* but has explicitely
given values for *sample_interval* and *log_interval*. Output is similar to
the previous one but averaging and the printout are performe at a different
rate.

**Example 3:**

Conside action with syntax error ('temp' key is missing lower value):

    actions:
    - name: action_1
      module: gm
      device: 33367 50599
      monitor: true
      metrics:
        temp: true 20
        fan: true 10 0
        power: true 100 0
      sample_interval: 1000
      log_interval: 1200

Output for such configuration is:

    RVS-GM: action: action_1 Wrong number of metric parameters

**Example 4:**

Consider action with logical error:

    actions:
    - name: action_1
      module: gm
      device: all
      monitor: true
      metrics:
        temp: false 20 0
        clock: true 1500 852
        power: true 100 0
      sample_interval: 5000
      log_interval: 4000
      duration: 8000

Output for such configuration is:

    RVS-GM: action: action_1 Log interval has the lower value than the sample interval


@section usg6 6 PESM Module
The PCIe State Monitor (PESM) tool is used to actively monitor the PCIe
interconnect between the host platform and the GPU. The module registers
“listener” on a target GPUs PCIe interconnect, and log a message whenever it
detects a state change. The PESM is able to detect the following state changes:

1. PCIe link speed changes
2. GPU device power state changes

This module is intended to run concurrently with other actions, and provides a
‘start’ and ‘stop’ configuration key to start the monitoring and then stop it
after testing has completed. For information on GPU power state monitoring
please consult the 7.6. PCI Power Management Capability Structure, Gen 3 spec,
page 601, device states D0-D3. For information on link status changes please
consult the 7.8.8. Link Status Register (Offset 12h), Gen 3 spec, page 635.

Monitoring is performed by polling respective PCIe registers roughly every 1ms
(one millisecond).

@subsection usg61 6.1 Module Specific Keys
<table>
<tr><th>Config Key</th> <th>Type</th><th> Description</th></tr>
<tr><td>monitor</td><td>Bool</td><td>This this key is set to true, the PESM
module will start monitoring on specified devices. If this key is set to false,
all other keys are ignored and monitoring will be stopped for all devices.</td>
</tr> </table>

@subsection usg62 6.2 Output

Module specific output keys are described in the table below:
<table>
<tr><th>Output Key</th> <th>Type</th><th> Description</th></tr>
<tr><td>state</td><td>String</td><td>A string detailing the current power state
of the GPU or the speed of the PCIe link.</td></tr>
</table>

When monitoring is started for a target GPU, a result message is logged
with the following format:

    [RESULT][<timestamp>][<action name>] pesm <gpu id> started

When monitoring is stopped for a target GPU, a result message is logged
with the following format:

    [RESULT][<timestamp>][<action name>] pesm all stopped

When monitoring is enabled, any detected state changes in link speed or GPU
power state will generate the following informational messages:

    [INFO ][<timestamp>][<action name>] pesm <gpu id> power state change <state>
    [INFO ][<timestamp>][<action name>] pesm <gpu id> link speed change <state>

@subsection usg63 6.3 Examples

**Example 1**

Here is a typical check utilizing PESM functionlit:

    actions:
    - name: action_1
      device: all
      module: pesm
      monitor: true
    - name: action_2
      device: 33367
      module: gst
      parallel: false
      count: 2
      wait: 100
      duration: 18000
      ramp_interval: 7000
      log_interval: 1000
      max_violations: 1
      copy_matrix: false
      target_stress: 5000
      tolerance: 0.07
      matrix_size: 5760
    - name: action_3
      device: all
      module: pesm
      monitor: false

-  **action_1** will initiate monitoring on all devices by setting key **monitor** to **true**\n
-  **action_2** will start GPU stgress test
-  **action_3** will stop monitoring

If executed like this:

    sudo rvs -c conf/pesm8.conf -d 3

output similar to this one can be produced:

    [RESULT] [497544.637462] [action_1] pesm all started
    [INFO  ] [497544.648299] [action_1] pesm 33367 link speed change 8 GT/s
    [INFO  ] [497544.648299] [action_1] pesm 33367 power state change D0
    [INFO  ] [497544.648733] [action_1] pesm 3254 link speed change 8 GT/s
    [INFO  ] [497544.648733] [action_1] pesm 3254 power state change D0
    [INFO  ] [497544.650413] [action_1] pesm 50599 link speed change 8 GT/s
    [INFO  ] [497544.650413] [action_1] pesm 50599 power state change D0
    [INFO  ] [497545.170392] [action_2] gst 33367 start 5000.000000 copy matrix:false
    [INFO  ] [497547.36602 ] [action_2] gst 33367 Gflops 6478.066983
    [INFO  ] [497548.69221 ] [action_2] gst 33367 target achieved 5000.000000
    [INFO  ] [497549.101219] [action_2] gst 33367 Gflops 5189.993529
    [INFO  ] [497550.132376] [action_2] gst 33367 Gflops 5189.993529
    ...
    [INFO  ] [497563.569370] [action_2] gst 33367 Gflops 5174.935520
    [RESULT] [497564.86904 ] [action_2] gst 33367 Gflop: 6478.066983 flops_per_op: 382.205952x1e9 bytes_copied_per_op: 398131200 try_ops_per_sec: 13.081952 pass: TRUE
    [INFO  ] [497564.220311] [action_2] gst 33367 start 5000.000000 copy matrix:false
    [INFO  ] [497566.70585 ] [action_2] gst 33367 Gflops 6521.049418
    [INFO  ] [497567.99929 ] [action_2] gst 33367 target achieved 5000.000000
    [INFO  ] [497568.143096] [action_2] gst 33367 Gflops 5130.281235
    ...
    [INFO  ] [497582.683893] [action_2] gst 33367 Gflops 5135.204729
    [RESULT] [497583.130945] [action_2] gst 33367 Gflop: 6521.049418 flops_per_op: 382.205952x1e9 bytes_copied_per_op: 398131200 try_ops_per_sec: 13.081952 pass: TRUE
    [RESULT] [497583.155470] [action_3] pesm all stopped

**Example 2:**

Consider this file:

    actions:
    - name: act1
      device: all
      deviceid: xxx
      module: pesm
      monitor: true


This file has and ivalid entry in **deviceid** key.
If execute, an error will be reported:

    RVS-PESM: action: act1  invalide 'deviceid' key value: xxx


@section usg7 7 RCQT Module

This ‘module’ is actually a set of feature checks that target and qualify the
configuration of the platform. Many of the checks can be done manually using the
operating systems command line tools and general knowledge about ROCm’s
requirements. The purpose of the RCQT modules is to provide an extensible, OS
independent and scriptable interface capable for performing the configuration
checks required for ROCm support. The checks in this module do not target a
specific device (instead the underlying platform is targeted), and any device or
device id keys specified will be ignored. Iteration keys, i.e. count, wait and
duration, are also ignored.
\n\n
One RCQT action can perform only one check at the time. Checks are decoded in
this order:\n

- If 'package' key is detected, packaging check will be performed
- If 'user' key is detected, user check will be performed
- If 'os_versions' and 'kernel_versions' keys are detected, OS check will be performed
- If 'soname', 'arch' and 'ldpath' keys are detected, linker/loader check will be
performed
- If 'file' key is detected, file check will be performed

All other keys not pertinent to the detected action are ignored.

@subsection usg71 7.1 Packaging Check

This feature is used to check installed packages on the system. It provides
checks for installed packages and the currently available package versions, if
applicable.

@subsubsection usg711 7.1.1 Packaging Check Specific Keys

Input keys are described in the table below:

<table>
<tr><th>Config Key</th> <th>Type</th><th> Description</th></tr>
<tr><td>package</td><td>String</td>
<td>Specifies the package to check. This key is required.</td></tr>
<tr><td>version</td><td>String</td>
<td>This is an optional key specifying a package version. If it is provided, the
tool will check if the version is installed, failing otherwise. If it is not
provided any version matching the package name will result in success.
</td></tr>
</table>

@subsubsection usg712 7.1.2 Output

Output keys are described in the table below:

<table>
<tr><th>Output Key</th> <th>Type</th><th> Description</th></tr>
<tr><td>installed</td><td>Bool</td>
<td>If the test has passed, the output will be true. Otherwise it will be false.
</td></tr>
</table>

The check will emit a result message with the following format:

    [RESULT][<timestamp>][<action name>] rcqt packagecheck <package> <installed>

The package name will include the version of the package if the version key is
specified. The installed output value will either be true or false depending on
if the package is installed or not.

@subsubsection usg713 7.1.3 Examples

**Example 1:**

In this example, given package does not exist.

    actions:
    - name: action_1
      module: rcqt
      package: zip12345

The output for such configuration is:

    [RESULT] [500022.877512] [action_1] rcqt packagecheck zip12345 FALSE

**Example 2:**

In this example, version of the given package is incorrect.

    actions:
    - name: action_1
      module: rcqt
      package: zip
      version: 3.0-11****

The output for such configuration is:

    [RESULT] [500123.480561] [action_1] rcqt packagecheck zip FALSE

**Example 3:**

In this example, given package exists.

    actions:
    - name: action_1
      module: rcqt
      package: zip

The output for such configuration is:

    [RESULT] [500329.824495] [action_1] rcqt packagecheck zip TRUE

**Example 4:**

In this example, given package exists and its version is correct.

    actions:
    - name: action_1
      module: rcqt
      package: zip
      version: 3.0-11

The output for such configuration is:

    [RESULT] [500595.859025] [action_1] rcqt packagecheck zip TRUE


@subsection usg72 7.2 User Check

This feature checks for the existence of a user and the user’s group membership.

@subsubsection usg721 7.2.1 User Check Specific Keys

Input keys are described in the table below:

<table>
<tr><th>Config Key</th> <th>Type</th><th> Description</th></tr>
<tr><td>user</td><td>String</td>
<td>Specifies the user name to check. This key is required.</td></tr>
<tr><td>groups</td><td>Collection of Strings</td>
<td>This is an optional key specifying a collection of groups the user should
belong to. The user’s membership in each group will be checked.
</td></tr>
</table>

@subsubsection usg722 7.2.2 Output

Output keys are described in the table below:

<table>
<tr><th>Output Key</th> <th>Type</th><th> Description</th></tr>
<tr><td>exists</td><td>Bool</td>
<td>This value is true if the user exists.
</td></tr>
<tr><td>members</td><td>Collection of Bools</td>
<td>This value is true if the user is a member of the specified group.
</td></tr>
</table>

The status of the user’s existence is provided in a message with the following
format:

    [RESULT][<timestamp>][<action name>] rcqt usercheck <user> <exists>

For each group in the list, a result message with the following format will be
generated:

    [RESULT][<timestamp>][<action name>] rcqt usercheck <user> <group> <member>

If the user doesn’t exist no group checks will take place.

@subsubsection usg723 7.2.3 Examples

**Example 1:**

In this example, given user does not exist.

    actions:
    - name: action_1
      device: all
      module: rcqt
      user: jdoe
      group: sudo,video

The output for such configuration is:

    [RESULT] [496559.219160] [action_1] rcqt usercheck jdoe false

Group check is not performed.

**Example 2:**

In this example, group **rvs** does not exist.

    actions:
    - name: action_1
      device: all
      module: rcqt
      user: jovanbhdl
      group: rvs,video

The output for such configuration is:

    [RESULT] [496984.993394] [action_1] rcqt usercheck jovanbhdl true
    [ERROR ] [496984.993535] [action_1] rcqt usercheck group rvs doesn't exist
    [RESULT] [496984.993578] [action_1] rcqt usercheck jovanbhdl video true


**Example 3:**

In this example, given user exists and belongs to given groups.

    actions:
    - name: action_1
      device: all
      module: rcqt
      user: jovanbhdl
      group: sudo,video

The output for such configuration is:

    [RESULT] [497361.361045] [action_1] rcqt usercheck jovanbhdl true
    [RESULT] [497361.361168] [action_1] rcqt usercheck jovanbhdl sudo true
    [RESULT] [497361.361209] [action_1] rcqt usercheck jovanbhdl video true


@subsection usg73 7.3 File/device Check

This feature checks for the existence of a file, its owner, group, permissions
and type. The primary purpose of this module is to check that the device
interfaces for the driver and the kfd are available, but it can also be used to
check for the existence of important configuration files, libraries and
executables.

@subsubsection usg731 7.3.1 File/device Check Specific Keys

Input keys are described in the table below:

<table>
<tr><th>Config Key</th> <th>Type</th><th> Description</th></tr>
<tr><td>file</td><td>String</td>
<td>The value of this key should satisfy the filename limitations of the target
OS and specifies the file to check. This key is required.
</td></tr>
<tr><td>owner</td><td>String</td>
<td>The expected owner of the file. If this key is specified ownership is
tested.
</td></tr>
<tr><td>group</td><td>String</td>
<td>If this key is specified, group ownership is tested.
</td></tr>
<tr><td>permission</td><td>Integer</td>
<td>If this key is specified, the permissions on the file are tested. The
permissions are expected to match the permission value given.
</td></tr>
<tr><td>type</td><td>Integer</td>
<td>If this key is specified the file type is checked.
</td></tr>
<tr><td>exists</td><td>Bool</td>
<td>If this key is specified and set to false all optional parameters will be
ignored and a check will be made to make sure the file does not exist. The
default value for this key is true.
</td></tr>
</table>

@subsubsection usg732 7.3.2 Output

Output keys are described in the table below:

<table>
<tr><th>Output Key</th> <th>Type</th><th> Description</th></tr>
<tr><td>owner</td><td>Bool</td>
<td>True if the correct user owns the file.
</td></tr>
<tr><td>group</td><td>Bool</td>
<td>True if the correct group owns the file.
</td></tr>
<tr><td>permission</td><td>Bool</td>
<td>True if the file has the correct permissions.
</td></tr>
<tr><td>type</td><td>Bool</td>
<td>True if the file is of the right type.
</td></tr>
<tr><td>exists</td><td>Bool</td>
<td>True if the file exists and the ‘exists’ config key is true. True if the
file does not exist and the ‘exists’ key if false.
</td></tr>
</table>

If the ‘exists’ key is true a set of messages, one for each stat check, will be
generated with the following format:

    [RESULT][<timestamp>][<action name>] rcqt filecheck <config key> <matching output key>

If the ‘exists’ key is false the format of the message will be:

    [RESULT][<timestamp>][<action name>] rcqt filecheck <file> DNE <exists>

@subsubsection usg733 7.3.3 Examples

**Example 1:**

In this example, config key exists is set to **true** by default and file really
exists so parameters are tested. Permission number 644 equals to rw-r--r-- and
type number 40 indicates that it is a folder.

rcqt_fc4.conf :

    actions:
    - name: action_1
      device: all
      module: rcqt
      file: /work/mvisekrunahdl/ROCmValidationSuite/rcqt.so/src
      owner: mvisekrunahdl
      group: mvisekrunahdl
      permission: 664
      type: 40

Output from running this action:

    [RESULT] [240384.678074] [action_1] rcqt filecheck mvisekrunahdl owner:true
    [RESULT] [240384.678214] [action_1] rcqt filecheck mvisekrunahdl group:true
    [RESULT] [240384.678250] [action_1] rcqt filecheck 664 permission:true
    [RESULT] [240384.678275] [action_1] rcqt filecheck 100 type:true


**Example 2:**

In this example, config key exists is set to false, but file actually exists so
parameters are not tested.

rcqt_fc1.conf:

    actions:
    - name: action_1
      device: all
      module: rcqt
      file: /work/mvisekrunahdl/ROCmValidationSuite/src
      owner: root
      permission: 644
      type: 40
      exists: false

The output for such configuration is:

    [RESULT] [240188.150386] [action_1] rcqt filecheck /work/mvisekrunahdl/ROCmValidationSuite/src DNE false

**Example 3:**

In this example, config key **exists** is true by default and file really
exists. Config key **group, permission** and **type** are not specified so only
ownership is tested.

rcqt_fc2.conf:

    actions:
    - name: action_1
      device: all
      module: rcqt
      file: /work/mvisekrunahdl/build/test.txt
      owner: root

The output for such configuration is:

    [RESULT] [240253.957738] [action_1] rcqt filecheck root owner:true


**Example 4:**

In this example, config key **exists** is true by default, but given file does
not exist.

rcqt_fc3.conf:

    actions:
    - name: action_1
      device: all
      module: rcqt
      file: /work/mvisekrunahdl/ROCmValidationSuite/rcqt.so/src/tst
      owner: mvisekrunahdl
      group: mvisekrunahdl
      permission: 664
      type: 100

The output for such configuration is:

    [ERROR ] [240277.355553] [action_1] rcqt File is not found


@subsection usg74 7.4 Kernel compatibility Check

The rcqt-kernelcheck module determines the version of the operating system and
the kernel installed on the platform and compares the values against the list of
supported values.

@subsubsection usg741 7.4.1 Kernel compatibility Check Specific Keys

Input keys are described in the table below:

<table>
<tr><th>Config Key</th> <th>Type</th><th> Description</th></tr>
<tr><td>os_versions</td><td>Collection of Strings</td>
<td>A collection of strings corresponding to operating systems names, i.e.
{“Ubuntu 16.04.3 LTS”, “Centos 7.4”, etc.}
</td></tr>
<tr><td>kernel_versions</td><td>Collection of Strings</td>
<td>A collection of strings corresponding to kernel version names, i.e.
{“4.4.0-116-generic”, “4.13.0-36-generic”, etc.}
</td></tr>
</table>

@subsubsection usg742 7.4.2 Output

Output keys are described in the table below:

<table>
<tr><th>Output Key</th> <th>Type</th><th> Description</th></tr>
<tr><td>os</td><td>String</td>
<td>The actual OS installed on the system.
</td></tr>
<tr><td>kernel</td><td>String</td>
<td>The actual kernel version installed on the system.
</td></tr>
<tr><td>pass</td><td>Bool</td>
<td>True if the actual os version and kernel version match any value provided in
the collection.
</td></tr>
</table>

If the detected versions of the operating system and the kernel version match
any of the supported values the pass output key will be true. Otherwise it will
be false. The result message will contain the actual os version and the kernel
version regardless of where the check passed or failed.

    [RESULT][<timestamp>][<action name>] rcqt kernelcheck <os version> <kernel version> <pass>


@subsubsection usg743 7.4.3 Examples

**Example 1:**

In this example, given kernel version is incorrect.

    actions:
    - name: action_1
      device: all
      module: rcqt
      os_version: Ubuntu 16.04.5 LTS
      kernel_version: 4.4.0-116-generic-wrong

The output for such configuration is:

    [RESULT] [498398.774182] [action_1] rcqt kernelcheck Ubuntu 16.04.5 LTS 4.18.0-rc1-kfd-compute-roc-master-8874 fail

**Example 2**

In this example, given os version and kernel verison are the correct ones.

    actions:
    - name: action_1
      device: all
      module: rcqt
      os_version: Ubuntu 16.04.5 LTS
      kernel_version: 4.18.0-rc1-kfd-compute-roc-master-8874

The output for such configuration is:

    [RESULT] [515924.695932] [action_1] rcqt kernelcheck Ubuntu 16.04.5 LTS 4.18.0-rc1-kfd-compute-roc-master-8874 pass


@subsection usg75 7.5 Linker/Loader Check

This feature checks that a search by the linker/loader for a library finds the
correct version in the correct location. The check should include a SONAME
version of the library, the expected location and the architecture of the
library.


@subsubsection usg751 7.5.1 Linker/Loader Check Specific Keys

Input keys are described in the table below:

<table>
<tr><th>Config Key</th> <th>Type</th><th> Description</th></tr>
<tr><td>soname</td><td>String</td>
<td>This is the SONAME of the library for the check. An SONAME library contains
the major version of the library in question.
</td></tr>
<tr><td>arch</td><td>String</td>
<td>This value qualifies the architecture expected for the library.
</td></tr>
<tr><td>ldpath</td><td>String</td>
<td>This is the fully qualified path where the library is expected to be
located.
</td></tr>
</table>

@subsubsection usg752 7.5.2 Output

Output keys are described in the table below:

<table>
<tr><th>Output Key</th> <th>Type</th><th> Description</th></tr>
<tr><td>arch</td><td>String</td>
<td>The actual architecture found for the file, or NA if it wasn’t found.
</td></tr>
<tr><td>path</td><td>String</td>
<td>The actual path the linker is looking for the file at, or “not found” if the
file isn’t found.
</td></tr>
<tr><td>pass</td><td>Bool</td>
<td>True if the linker/loader is looking for the file in the correct place with
the correctly specified architecture.
</td></tr>
</table>

If the linker/loader search path looks for the soname version of the library,
qualified by arch, at the directory specified the test will pass. Otherwise it
will fail. The output message has the following format:

    [RESULT][<timestamp>][<action name>] rcqt ldconfigcheck <soname> <arch> <path> <pass>

@subsubsection usg753 7.5.3 Examples

**Example 1:**

Consider this action:

    actions:
    - name: action_1
      device: all
      module: rcqt
      soname: librcqt.so.0.0.3fail
      arch: i386:x86-64
      ldpath: /work/jovanbhdl/build/bin

The test will fail because the file given is not found on the specified path:

    [RESULT] [587042.789384] [action_1] rcqt ldconfigcheck librcqt.so.0.0.3fail i386:x86-64 /work/jovanbhdl/build/bin false

**Example 2:**

Consider this action:

    actions:
    - name: action_1
      device: all
      module: rcqt
      soname: librcqt.so.0.0.16
      arch: i386:x86-64
      ldpath: /work/jovanbhdl/build/bin

The test will pass and will output the message:

    [RESULT] [587047.395787] [action_1] rcqt ldconfigcheck librcqt.so.0.0.16 i386:x86-64 /work/jovanbhdl/build/bin true


@section usg8 8 PEQT Module

PCI Express Qualification Tool module targets and qualifies the configuration of
the platforms PCIe connections to the GPUs. The purpose of the PEQT module is to
provide an extensible, OS independent and scriptable interface capable of
performing the PCIe interconnect configuration checks required for ROCm support
of GPUs. This information can be obtained through the sysfs PCIe interface or by
using the PCIe development libraries to extract values from various PCIe
control, status and capabilities registers. These registers are specified in the
PCI Express Base Specification, Revision 3. Iteration keys, i.e. count, wait and
duration will be ignored for actions using the PEQT module.

@subsection usg81 8.1 Module Specific Keys
Module specific output keys are described in the table below:
<table>
<tr><th>Config Key</th> <th>Type</th><th> Description</th></tr>
<tr><td>capability</td><td>Collection of Structures with the
following format:\n{String,String}</td>
<td>The PCIe capability key contains a collection of structures that specify
which PCIe capability to check and the expected value of the capability. A check
structure must contain the PCIe capability value, but an expected value may be
omitted. The value of all valid capabilities that are a part of this collection
will be entered into the capability_value field. Possible capabilities, and
their value types are:\n\n
link_cap_max_speed\n
link_cap_max_width\n
link_stat_cur_speed\n
link_stat_neg_width\n
slot_pwr_limit_value\n
slot_physical_num\n
atomic_op_32_completer\n
atomic_op_64_completer\n
atomic_op_128_CAS_completer\n
atomic_op_routing\n
dev_serial_num\n
kernel_driver\n
pwr_base_pwr\n
pwr_rail_type\n
device_id\n
vendor_id\n\n

The expected value String is a regular expression that is used to check the
actual value of the capability.

</td></tr>
</table>

@subsection usg82 8.2 Output
Module specific output keys are described in the table below:
<table>
<tr><th>Output Key</th> <th>Type</th><th> Description</th></tr>
<tr><td>capability_value</td><td>Collection of Strings</td>
<td>For each of the capabilities specified in the capability key, the actual
value of the capability will be returned, represented as a String.</td></tr>
<tr><td>pass</td><td>String</td> <td>'true' if all of the properties match the
values given, 'false' otherwise.</td></tr>
</table>

The qualification check queries the specified PCIe capabilities and
properties and checks that their actual values satisfy the regular expression
provided in the ‘expected value’ field for that capability. The pass output key
will be true and the test will pass if all of the properties match the values
given. After the check is finished, the following informational messages will be
generated:

    [INFO  ][<timestamp>][<action name>] peqt <capability> <capability_value>
    [RESULT][<timestamp>][<action name>] peqt <pass>

For details regarding each of the capabilities and current values consult the
chapters in the PCI Express Base Specification, Revision 3.

@subsection usg83 8.3 Examples

**Example 1:**

A regular PEQT configuration file looks like this:

    actions:
    - name: pcie_act_1
      module: peqt
      capability:
        link_cap_max_speed:
        link_cap_max_width:
        link_stat_cur_speed:
        link_stat_neg_width:
        slot_pwr_limit_value:
        slot_physical_num:
        device_id:
        vendor_id:
        kernel_driver:
        dev_serial_num:
        D0_Maximum_Power_12V:
        D0_Maximum_Power_3_3V:
        D0_Sustained_Power_12V:
        D0_Sustained_Power_3_3V:
        atomic_op_routing:
        atomic_op_32_completer:
        atomic_op_64_completer:
        atomic_op_128_CAS_completer:
      device: all

Please note:
- when setting the 'device' configuration key to 'all', the RVS will detect all the AMD compatible GPUs and run the test on all of them

- there are no regular expression for this .conf file, therefore RVS will report TRUE if at least one AMD compatible GPU is registered within the system. Otherwise it will report FALSE.

Please note that the Power Budgeting capability is a dynamic one, having the following form:

    <PM_State>_<Type>_<Power rail>

where:

    PM_State = D0/D1/D2/D3
    Type=PMEAux/Auxiliary/Idle/Sustained/Maximum
    PowerRail = Power_12V/Power_3_3V/Power_1_5V_1_8V/Thermal

When the RVS tool runs against such a configuration file, it will query for the all the PCIe capabilities specified under the capability list (and log the corresponding values) for all the AMD compatible GPUs. For those PCIe capabilities that are not supported by the HW platform were the RVS is running, a "NOT SUPPORTED" message will be logged.

The output for such a configuration file may look like this:


    [INFO ] [177628.401176] pcie_act_1 peqt D0_Maximum_Power_12V NOT SUPPORTED
    [INFO ] [177628.401229] pcie_act_1 peqt D0_Maximum_Power_3_3V NOT SUPPORTED
    [INFO ] [177628.401248] pcie_act_1 peqt D0_Sustained_Power_12V NOT SUPPORTED
    [INFO ] [177628.401269] pcie_act_1 peqt D0_Sustained_Power_3_3V NOT SUPPORTED
    [INFO ] [177628.401282] pcie_act_1 peqt atomic_op_128_CAS_completer FALSE
    [INFO ] [177628.401291] pcie_act_1 peqt atomic_op_32_completer FALSE
    [INFO ] [177628.401303] pcie_act_1 peqt atomic_op_64_completer FALSE
    [INFO ] [177628.401311] pcie_act_1 peqt atomic_op_routing TRUE
    [INFO ] [177628.401317] pcie_act_1 peqt dev_serial_num NOT SUPPORTED
    [INFO ] [177628.401323] pcie_act_1 peqt device_id 26720
    [INFO ] [177628.401334] pcie_act_1 peqt kernel_driver amdgpu
    [INFO ] [177628.401342] pcie_act_1 peqt link_cap_max_speed 8 GT/s
    [INFO ] [177628.401352] pcie_act_1 peqt link_cap_max_width x16
    [INFO ] [177628.401359] pcie_act_1 peqt link_stat_cur_speed 8 GT/s
    [INFO ] [177628.401367] pcie_act_1 peqt link_stat_neg_width x16
    [INFO ] [177628.401375] pcie_act_1 peqt slot_physical_num #0
    [INFO ] [177628.401396] pcie_act_1 peqt slot_pwr_limit_value 0.000W
    [INFO ] [177628.401402] pcie_act_1 peqt vendor_id 4098
    [INFO ] [177628.401656] pcie_act_1 peqt D0_Maximum_Power_12V NOT SUPPORTED
    [INFO ] [177628.401675] pcie_act_1 peqt D0_Maximum_Power_3_3V NOT SUPPORTED
    [INFO ] [177628.401692] pcie_act_1 peqt D0_Sustained_Power_12V NOT SUPPORTED
    [INFO ] [177628.401709] pcie_act_1 peqt D0_Sustained_Power_3_3V NOT SUPPORTED
    [INFO ] [177628.401719] pcie_act_1 peqt atomic_op_128_CAS_completer FALSE
    [INFO ] [177628.401728] pcie_act_1 peqt atomic_op_32_completer FALSE
    [INFO ] [177628.401736] pcie_act_1 peqt atomic_op_64_completer FALSE
    [INFO ] [177628.401745] pcie_act_1 peqt atomic_op_routing TRUE
    [INFO ] [177628.401750] pcie_act_1 peqt dev_serial_num NOT SUPPORTED
    [INFO ] [177628.401757] pcie_act_1 peqt device_id 26720
    [INFO ] [177628.401771] pcie_act_1 peqt kernel_driver amdgpu
    [INFO ] [177628.401781] pcie_act_1 peqt link_cap_max_speed 8 GT/s
    [INFO ] [177628.401788] pcie_act_1 peqt link_cap_max_width x16
    [INFO ] [177628.401794] pcie_act_1 peqt link_stat_cur_speed 8 GT/s
    [INFO ] [177628.401800] pcie_act_1 peqt link_stat_neg_width x16
    [INFO ] [177628.401806] pcie_act_1 peqt slot_physical_num #0
    [INFO ] [177628.401814] pcie_act_1 peqt slot_pwr_limit_value 0.000W
    [INFO ] [177628.401819] pcie_act_1 peqt vendor_id 4098
    [RESULT] [177628.403781] pcie_act_1 peqt TRUE

**Example 2:**

Another example of a configuration file, which queries for a smaller subset of PCIe capabilities but adds regular expressions check, is given below

    actions:
    - name: pcie_act_1
      module: peqt
      capability:
        link_cap_max_speed: '^(2\.5 GT\/s|5 GT\/s|8 GT\/s)$'
        link_cap_max_width:
        link_stat_cur_speed: '^(2\.5 GT\/s|5 GT\/s|8 GT\/s)$'
        link_stat_neg_width:
        slot_pwr_limit_value: '[a-b][d-'
        slot_physical_num:
        device_id:
        vendor_id:
        kernel_driver:
      device: all

For this example, the expected PEQT check result is TRUE if:

- at least one AMD compatible GPU is registered within the system and:
- all \<link_cap_max_speed> values for all AMD compatible GPUs match the given regular expression and
- all \<link_stat_cur_speed> values for all AMD compatible GPUs match the given regular expression

Please note that the \<slot_pwr_limit_value> regular expression is not valid and will be skipped without affecting the PEQT module's check RESULT (however, an error will be logged out)

**Example 3:**

Another example with even more regular expressions is given below. The expected PEQT check result is TRUE if at least one AMD compatible GPU having the ID 3254 or 33367 is registered within the system and all the PCIe capabilities values match their corresponding regular expressions.

    actions:
    - name: pcie_act_1
      module: peqt
      deviceid: 26720
      capability:
        link_cap_max_speed: '^(2\.5 GT\/s|5 GT\/s|8 GT\/s)$'
        link_cap_max_width: ^(x8|x16)$
        link_stat_cur_speed: '^(8 GT\/s)$'
        link_stat_neg_width: ^(x8|x16)$
        kernel_driver: ^amdgpu$
        atomic_op_routing: ^((TRUE|FALSE){1})$
        atomic_op_32_completer: ^((TRUE|FALSE){1})$
        atomic_op_64_completer: ^((TRUE|FALSE){1})$
        atomic_op_128_CAS_completer: ^((TRUE|FALSE){1})$
      device: 3254 33367

@section usg9 9 SMQT Module
The GPU SBIOS mapping qualification tool is designed to verify that a platform’s
SBIOS has satisfied the BAR mapping requirements for VDI and Radeon Instinct
products for ROCm support. These are the current BAR requirements:\n\n

BAR 1: GPU Frame Buffer BAR – In this example it happens to be 256M, but
typically this will be size of the GPU memory (typically 4GB+). This BAR has to
be placed < 2^40 to allow peer- to-peer access from other GFX8 AMD GPUs. For
GFX9 (Vega GPU) the BAR has to be placed < 2^44 to allow peer-to-peer access
from other GFX9 AMD GPUs.\n\n

BAR 2: Doorbell BAR – The size of the BAR is typically will be < 10MB (currently
fixed at 2MB) for this generation GPUs. This BAR has to be placed < 2^40 to
allow peer-to-peer access from other current generation AMD GPUs.\n\n
BAR 3: IO BAR - This is for legacy VGA and boot device support, but since this
the GPUs in this project are not VGA devices (headless), this is not a concern
even if the SBIOS does not setup.\n\n

BAR 4: MMIO BAR – This is required for the AMD Driver SW to access the
configuration registers. Since the reminder of the BAR available is only 1 DWORD
(32bit), this is placed < 4GB. This is fixed at 256KB.\n\n

BAR 5: Expansion ROM – This is required for the AMD Driver SW to access the
GPU’s video-BIOS. This is currently fixed at 128KB.\n\n

Refer to the ROCm Use of Advanced PCIe Features and Overview of How BAR Memory
is Used In ROCm Enabled System web page for more information about how BAR
memory is initialized by VDI and Radeon products. Iteration keys, i.e. count,
wait and duration will be ignored.

@subsection usg91 9.1 Module Specific Keys

Module specific output keys are described in the table below:
<table>
<tr><th>Config Key</th> <th>Type</th><th> Description</th></tr>
<tr><td>bar1_req_size</td><td>Integer</td>
<td>This is an integer specifying the required size of the BAR1 frame buffer
region.</td></tr>
<tr><td>bar1_base_addr_min</td><td>Integer</td>
<td>This is an integer specifying the minimum value the BAR1 base address can
be.</td></tr>
<tr><td>bar1_base_addr_max</td><td>Integer</td>
<td>This is an integer specifying the maximum value the BAR1 base address can
be.</td></tr>
<tr><td>bar2_req_size</td><td>Integer</td>
<td>This is an integer specifying the required size of the BAR2 frame buffer
region.</td></tr>
<tr><td>bar2_base_addr_min</td><td>Integer</td>
<td>This is an integer specifying the minimum value the BAR2 base address can
be.</td></tr>
<tr><td>bar2_base_addr_max</td><td>Integer</td>
<td>This is an integer specifying the maximum value the BAR2 base address can
be.</td></tr>
<tr><td>bar4_req_size</td><td>Integer</td>
<td>This is an integer specifying the required size of the BAR4 frame buffer
region.</td></tr>
<tr><td>bar4_base_addr_min</td><td>Integer</td>
<td>This is an integer specifying the minimum value the BAR4 base address can
be.</td></tr>
<tr><td>bar4_base_addr_max</td><td>Integer</td>
<td>This is an integer specifying the maximum value the BAR4 base address can
be.</td></tr>
<tr><td>bar5_req_size</td><td>Integer</td>
<td>This is an integer specifying the required size of the BAR5 frame buffer
region.</td></tr>
</table>

@subsection usg92 9.2 Output
Module specific output keys are described in the table below:
<table>
<tr><th>Output Key</th> <th>Type</th><th> Description</th></tr>
<tr><td>bar1_size</td><td>Integer</td><td>The actual size of BAR1.</td></tr>
<tr><td>bar1_base_addr</td><td>Integer</td><td>The actual base address of BAR1
memory.</td></tr>
<tr><td>bar2_size</td><td>Integer</td><td>The actual size of BAR2.</td></tr>
<tr><td>bar2_base_addr</td><td>Integer</td><td>The actual base address of BAR2
memory.</td></tr>
<tr><td>bar4_size</td><td>Integer</td><td>The actual size of BAR4.</td></tr>
<tr><td>bar4_base_addr</td><td>Integer</td><td>The actual base address of BAR4
memory.</td></tr>
<tr><td>bar5_size</td><td>Integer</td><td>The actual size of BAR5.</td></tr>
<tr><td>pass</td><td>String</td> <td>'true' if all of the properties match the
values given, 'false' otherwise.</td></tr>
</table>

The qualification check will query the specified bar properties and check that
they satisfy the give parameters. The pass output key will be true and the test
will pass if all of the BAR properties satisfy the constraints. After the check
is finished, the following informational messages will be generated:

    [INFO  ][<timestamp>][<action name>] smqt bar1_size <bar1_size>
    [INFO  ][<timestamp>][<action name>] smqt bar1_base_addr <bar1_base_addr>
    [INFO  ][<timestamp>][<action name>] smqt bar2_size <bar2_size>
    [INFO  ][<timestamp>][<action name>] smqt bar2_base_addr <bar2_base_addr>
    [INFO  ][<timestamp>][<action name>] smqt bar4_size <bar4_size>
    [INFO  ][<timestamp>][<action name>] smqt bar4_base_addr <bar4_base_addr>
    [INFO  ][<timestamp>][<action name>] smqt bar5_size <bar5_size>
    [RESULT][<timestamp>][<action name>] smqt <pass>


@subsection usg93 9.3 Examples

**Example 1:**

Consider this file (sizes are in bytes):

    actions:
    - name: action_1
      device: all
      module: smqt
      bar1_req_size: 17179869184
      bar1_base_addr_min: 0
      bar1_base_addr_max: 17592168044416
      bar2_req_size: 2097152
      bar2_base_addr_min: 0
      bar2_base_addr_max: 1099511627776
      bar4_req_size: 262144
      bar4_base_addr_min: 0
      bar4_base_addr_max: 17592168044416
      bar5_req_size: 131072

Results for three GPUs are:

    [INFO  ] [257936.568768] [action_1]  smqt bar1_size      17179869184 (16.00 GB)
    [INFO  ] [257936.568768] [action_1]  smqt bar1_base_addr 13C0000000C
    [INFO  ] [257936.568768] [action_1]  smqt bar2_size      2097152 (2.00 MB)
    [INFO  ] [257936.568768] [action_1]  smqt bar2_base_addr 13B0000000C
    [INFO  ] [257936.568768] [action_1]  smqt bar4_size      524288 (512.00 KB)
    [INFO  ] [257936.568768] [action_1]  smqt bar4_base_addr E4B00000
    [INFO  ] [257936.568768] [action_1]  smqt bar5_size      0 (0.00 B)
    [RESULT] [257936.568920] [action_1]  smqt fail
    [INFO  ] [257936.569234] [action_1]  smqt bar1_size      17179869184 (16.00 GB)
    [INFO  ] [257936.569234] [action_1]  smqt bar1_base_addr 1A00000000C
    [INFO  ] [257936.569234] [action_1]  smqt bar2_size      2097152 (2.00 MB)
    [INFO  ] [257936.569234] [action_1]  smqt bar2_base_addr 19F0000000C
    [INFO  ] [257936.569234] [action_1]  smqt bar4_size      524288 (512.00 KB)
    [INFO  ] [257936.569234] [action_1]  smqt bar4_base_addr E9900000
    [INFO  ] [257936.569234] [action_1]  smqt bar5_size      0 (0.00 B)
    [RESULT] [257936.569281] [action_1]  smqt fail
    [INFO  ] [257936.570798] [action_1]  smqt bar1_size      17179869184 (16.00 GB)
    [INFO  ] [257936.570798] [action_1]  smqt bar1_base_addr 16C0000000C
    [INFO  ] [257936.570798] [action_1]  smqt bar2_size      2097152 (2.00 MB)
    [INFO  ] [257936.570798] [action_1]  smqt bar2_base_addr 1710000000C
    [INFO  ] [257936.570798] [action_1]  smqt bar4_size      524288 (512.00 KB)
    [INFO  ] [257936.570798] [action_1]  smqt bar4_base_addr E7300000
    [INFO  ] [257936.570798] [action_1]  smqt bar5_size      0 (0.00 B)
    [RESULT] [257936.570837] [action_1]  smqt fail

In this example, BAR sizes reported by GPUs match those listed in configuration key except for the BAR5, hence the test fails.

@section usg10 10 PQT Module
The P2P Qualification Tool is designed to provide the list of all GPUs that
support P2P and characterize the P2P links between peers. In addition to testing
for P2P compatibility, this test will perform a peer-to-peer throughput test
between all unique P2P pairs for performance evaluation. These are known as
device-to-device transfers, and can be either uni-directional or bi-directional.
The average bandwidth obtained is reported to help debug low bandwidth issues.

@subsection usg101 10.1 Module Specific Keys
<table>
<tr><th>Config Key</th> <th>Type</th><th> Description</th></tr>
<tr><td>peers</td><td>Collection of Strings</td>
<td>This is a required key, and specifies the set of GPU(s) considered being
peers of the GPU specified in the action. If ‘all’ is specified, all other
GPU(s) on the system will be considered peers. Otherwise only the GPU ids
specified in the list will be considered.</td></tr>
<tr><td>peer_deviceid</td><td>Integer</td>
<td>This is an optional parameter, but if specified it restricts the peers list
to a specific device type corresponding to the deviceid.</td></tr>
<tr><td>test_bandwidth</td><td>Bool</td>
<td>If this key is set to true the P2P bandwidth benchmark will run if a pair of
devices pass the P2P check.</td></tr>
<tr><td>bidirectional</td><td>Bool</td>
<td>This option is only used if test_bandwidth key is true. This specifies the
type of transfer to run:\n
true – Do a bidirectional transfer test\n
false – Do a unidirectional transfer test
from the agent to its peers.
</td></tr>
<tr><td>parallel</td><td>Bool</td>
<td>This option is only used if the test_bandwith
key is true.\n
true – Run transfer testing to all peers
in parallel.\n
false – Run transfer testing to a single
peer at a time.
</td></tr>
<tr><td>duration</td><td>Integer</td>
<td>This option is only used if test_bandwidth is true. This key specifies the
duration a transfer test should run, given in milliseconds. If this key is not
specified, the default value is 10000 (10 seconds).
</td></tr>
<tr><td>log_interval</td><td>Integer</td>
<td>This option is only used if test_bandwidth is true. This is a positive
integer, given in milliseconds, that specifies an interval over which the moving
average of the bandwidth will be calculated and logged. The default value is
1000 (1 second). It must be smaller than the duration key.</td></tr>
</table>

@subsection usg102 10.2 Output

Module specific output keys are described in the table below:
<table>
<tr><th>Output Key</th> <th>Type</th><th> Description</th></tr>
<tr><td>p2p_result</td><td>Collection of Result Bools</td>
<td>Indicates if the gpu and the specified peer have P2P capabilities. If this
quantity is true, the GPU pair tested has p2p capabilities. If false, they are
not peers.</td></tr>
<tr><td>interval_bandwidth</td><td>Collection of Time Series Floats</td>
<td>The average bandwidth of a p2p transfer, during the log_interval time
period. </td></tr>
<tr><td>bandwidth</td><td>Collection of Floats</td>
<td>The average bandwidth of a p2p transfer, averaged over the entire test
duration of the interval.</td></tr>
</table>

If the value of test_bandwidth key is false, the tool will only try to determine
if the GPU(s) in the peers key are P2P to the action’s GPU. In this case the
bidirectional and log_interval values will be ignored, if they are specified. If
a gpu is a P2P peer to the device the test will pass, otherwise it will fail. A
message indicating the result will be provided for each GPUs specified. It will
have the following format:

    [RESULT][<timestamp>][<action name>] p2p <gpu id> <peer gpu id> <p2p_result>

If the value of test_bandwidth is true bandwidth testing between the device and
each of its peers will take place in parallel or in sequence, depending on the
value of the parallel flag. During the duration of bandwidth benchmarking,
informational output providing the moving average of the transfer’s bandwidth
will be calculated and logged at every time increment specified by the
log_interval parameter. The messages will have the following output:

    [INFO  ][<timestamp>][<action name>] p2p-bandwidth <gpu id> <peer gpu id> bidirectional: <bidirectional> <interval_bandwidth >

At the end of the test the average bytes/second will be calculated over the
entire test duration, and will be logged as a result:

    [RESULT][<timestamp>][<action name>] p2p-bandwidth <gpu id> <peer gpu id> bidirectional: <bidirectional> <bandwidth > <duration>

@subsection usg103 10.3 Examples


**Example 1:**

Here all source GPUs (device: all) with all destination GPUs (peers: all) are
tested for p2p capability with no bandwidth testing (test_bandwidth: false).

    actions:
    - name: action_1
      device: all
      module: pqt
      peers: all
      test_bandwidth: false


Possible result is:

    [RESULT] [167896.97743 ] [action_1] p2p 3254 3254 false
    [RESULT] [167896.97764 ] [action_1] p2p 3254 50599 true
    [RESULT] [167896.97818 ] [action_1] p2p 3254 33367 true
    [RESULT] [167896.97848 ] [action_1] p2p 50599 3254 true
    [RESULT] [167896.97873 ] [action_1] p2p 50599 50599 false
    [RESULT] [167896.97881 ] [action_1] p2p 50599 33367 true
    [RESULT] [167896.97907 ] [action_1] p2p 33367 3254 true
    [RESULT] [167896.97935 ] [action_1] p2p 33367 50599 true
    [RESULT] [167896.97960 ] [action_1] p2p 33367 33367 false

From the first line of result, we can see that GPU (ID 3254) can't access itself.
From the second line of result, we can see that source GPU (ID 3254) can access destination GPU (ID 50599).

**Example 2:**

Here all source GPUs (device: all) with all destination GPUs (peers: all) are tested for p2p capability including bandwidth testing (test_bandwidth: true) with
bidirectional transfers (bidirectional: true) and running in paralell (parallel : true) - default values are used for duration and log_interval.

    actions:
    - name: action_1
      device: all
      module: pqt
      peers: all
      test_bandwidth: true
      bidirectional: true
      parallel : true

Possible result is:

    [RESULT] [170487.818898] [action_1] p2p 3254 3254 false
    [RESULT] [170487.818918] [action_1] p2p 3254 50599 true
    [RESULT] [170487.818969] [action_1] p2p 3254 33367 true
    [RESULT] [170487.818997] [action_1] p2p 50599 3254 true
    [RESULT] [170487.819022] [action_1] p2p 50599 50599 false
    [RESULT] [170487.819029] [action_1] p2p 50599 33367 true
    [RESULT] [170487.819054] [action_1] p2p 33367 3254 true
    [RESULT] [170487.819080] [action_1] p2p 33367 50599 true
    [RESULT] [170487.819105] [action_1] p2p 33367 33367 false
    [INFO  ] [170488.351341] [action_1] p2p-bandwidth  3254 50599  bidirectional: true  7.46 GBps
    [INFO  ] [170488.352438] [action_1] p2p-bandwidth  3254 33367  bidirectional: true  7.91 GBps
    [INFO  ] [170488.353512] [action_1] p2p-bandwidth  50599 3254  bidirectional: true  7.58 GBps
    [INFO  ] [170488.354581] [action_1] p2p-bandwidth  50599 33367  bidirectional: true  6.93 GBps
    [INFO  ] [170488.355652] [action_1] p2p-bandwidth  33367 3254  bidirectional: true  5.69 GBps
    [INFO  ] [170488.356725] [action_1] p2p-bandwidth  33367 50599  bidirectional: true  5.67 GBps
    [RESULT] [170488.880108] [action_1] p2p-bandwidth  3254 50599  bidirectional: true  7.95 GBps  duration: 0.031455 ms
    [RESULT] [170488.881314] [action_1] p2p-bandwidth  3254 33367  bidirectional: true  7.50 GBps  duration: 0.133382 ms
    [RESULT] [170488.882453] [action_1] p2p-bandwidth  50599 3254  bidirectional: true  7.58 GBps  duration: 0.016483 ms
    [RESULT] [170488.883580] [action_1] p2p-bandwidth  50599 33367  bidirectional: true  8.31 GBps  duration: 0.060156 ms
    [RESULT] [170488.884702] [action_1] p2p-bandwidth  33367 3254  bidirectional: true  6.90 GBps  duration: 0.036216 ms
    [RESULT] [170488.885863] [action_1] p2p-bandwidth  33367 50599  bidirectional: true  2.58 GBps  duration: 0.096949 ms

From the last line of result, we can see that source GPU (ID 33367) can access
destination GPU (ID 50599) and that bidirectional bandwidth is 2.58 GBps.

**Example 3:**

Here some source GPUs (device: 50599) are targeting some destination GPUs
(peers: 33367 3254) with specified log interval (log_interval: 500) and duration
(duration: 1000). Bandwidth is tested (test_bandwidth: true) but only
unidirectional (bidirectional: false) without parallel execution (parallel:
false).

    actions:
    - name: action_1
      device: 50599
      module: pqt
      log_interval: 500
      duration: 1000
      peers: 33367 3254
      test_bandwidth: true
      bidirectional: false
      parallel: false

Possible output is:

    [RESULT] [175852.862778] [action_1] p2p 50599 3254 true
    [RESULT] [175852.862837] [action_1] p2p 50599 33367 true
    [INFO  ] [175853.393991] [action_1] p2p-bandwidth  50599 3254  bidirectional: false  4.56 GBps
    [INFO  ] [175853.395080] [action_1] p2p-bandwidth  50599 33367  bidirectional: false  4.60 GBps
    [INFO  ] [175853.928202] [action_1] p2p-bandwidth  50599 3254  bidirectional: false  4.40 GBps
    [RESULT] [175854.235624] [action_1] p2p-bandwidth  50599 3254  bidirectional: false  4.50 GBps  duration: 0.444359 ms
    [RESULT] [175854.236772] [action_1] p2p-bandwidth  50599 33367  bidirectional: false  4.57 GBps  duration: 0.218679 ms

From the last line of result, we can see that source GPU (ID 50599) can access
destination GPU (ID 33367) and that bidirectional bandwidth is 4.57 GBps.

**Example 4:**

Here some source GPUs (device: 3254) are targeting some destination GPUs (peers:
33367) with specified log interval (log_interval: 500) and duration (duration:
1000). Bandwidth is tested (test_bandwidth: true) but only unidirectional
(bidirectional: false) without parallel execution (parallel: false). Also, only
GPUs with specified device id are considered (peer_deviceid: 26720).

    actions:
    - name: action_1
      device: 3254
      module: pqt
      log_interval: 500
      duration: 1000
      peers: 33367
      peer_deviceid: 26720
      test_bandwidth: true
      bidirectional: false
      parallel: false

Possible output is:

    [RESULT] [176419.658547] [action_1] p2p 3254 33367 true
    [INFO  ] [176420.192962] [action_1] p2p-bandwidth  3254 33367  bidirectional: false  4.61 GBps
    [INFO  ] [176420.726227] [action_1] p2p-bandwidth  3254 33367  bidirectional: false  4.67 GBps
    [RESULT] [176421.1565  ] [action_1] p2p-bandwidth  3254 33367  bidirectional: false  4.65 GBps  duration: 0.645375 ms


@section usg11 11 PEBB Module
The PCIe Bandwidth Benchmark attempts to saturate the PCIe bus with DMA
transfers between system memory and a target GPU card’s memory. These are known
as host-to-device or device- to-host transfers, and can be either unidirectional
or bidirectional transfers. The maximum bandwidth obtained is reported.

@subsection usg111 11.1 Module Specific Keys

<table>
<tr><th>Config Key</th> <th>Type</th><th> Description</th></tr>
<tr><td>host_to_device</td><td>Bool</td>
<td>This key indicates if host to device transfers
will be considered. The default value is true.</td></tr>
<tr><td>device_to_host</td><td>Bool</td>
<td>This key indicates if device to host transfers
will be considered. The default value is true.
</td></tr>
<tr><td>log_interval</td><td>Integer</td>
<td>This is a positive integer, given in milliseconds, that specifies an
interval over which the moving average of the bandwidth will be calculated and
logged.</td></tr>
</table>

@subsection usg112 11.2 Output

Module specific output keys are described in the table below:
<table>
<tr><th>Output Key</th> <th>Type</th><th> Description</th></tr>
<tr><td>interval_bandwidth</td><td>Collection of Time Series Floats</td>
<td>The average bandwidth of a transfer, during the log_interval time
period. </td></tr>
<tr><td>bandwidth</td><td>Collection of Floats</td>
<td>The average bandwidth of a transfer, averaged over the entire test
duration of the interval.</td></tr>
</table>

During the execution of the benchmark, informational output providing the moving
average of the bandwidth of the transfer will be calculated and logged. This
interval is provided by the log_interval parameter and will have the following
output format:

    [INFO ][<timestamp>][<action name>] pcie-bandwidth <gpu id> h2d: <host_to_device> d2h: <device_to_host> <interval_bandwidth >

At the end of the test the average bytes/second will be calculated over the
entire test duration, and will be logged as a result:

    [RESULT][<timestamp>][<action name>] pcie-bandwidth <gpu id> h2d: <host_to_device> d2h: <device_to_host> < bandwidth > <duration>


@subsection usg113 11.3 Examples

@section usg12 12 GST Module
The GPU Stress Test modules purpose is to bring the CUs of the specified GPU(s)
to a target performance level in gigaflops by doing large matrix multiplications
using SGEMM/DGEMM (Single/Double-precision General Matrix Multiplication)
available in a library like rocBlas. The GPU stress module may be configured so
it does not copy the source arrays to the GPU before every matrix
multiplication. This allows the GPU performance to not be capped by device to
host bandwidth transfers. The module calculates how many matrix operations per
second are necessary to achieve the configured performance target and fails if
it cannot achieve that target. \n\n

This module should be used in conjunction with the GPU Monitor, to watch for
thermal, power and related anomalies while the target GPU(s) are under realistic
load conditions. By setting the appropriate parameters a user can ensure that
all GPUs in a node or cluster reach desired performance levels. Further analysis
of the generated stats can also show variations in the required power, clocks or
temperatures to reach these targets, and thus highlight GPUs or nodes that are
operating less efficiently.

@subsection usg121 12.1 Module Specific Keys

Module specific keys are described in the table below:

<table>
<tr><th>Config Key</th> <th>Type</th><th> Description</th></tr>
<tr><td>target_stress</td><td>Float</td>
<td>The maximum relative performance the GPU will attempt to achieve in
gigaflops. This parameter is required.</td></tr>
<tr><td>copy_matrix</td><td>Bool</td>
<td>This parameter indicates if each operation should copy the matrix data to
the GPU before executing. The default value is true.</td></tr>
<tr><td>ramp_interval</td><td>Integer</td>
<td>This is an time interval, specified in milliseconds, given to the test to
reach the given target_stress gigaflops. The default value is 5000 (5 seconds).
This time is counted against the duration of the test. If the target gflops, or
stress, is not achieved in this time frame, the test will fail. If the target
stress (gflops) is achieved the test will attempt to run for the rest of the
duration specified by the action, sustaining the stress load during that
time.</td></tr>
<tr><td>tolerance</td><td>Float</td>
<td>A value indicating how much the target_stress can fluctuate after the ramp
period for the test to succeed. The default value is 0.1 or 10%.</td></tr>
<tr><td>max_violations</td><td>Integer</td>
<td>The number of tolerance violations that can occur after the ramp_interval
for the test to still pass. The default value is 0.</td></tr>
<tr><td>log_interval</td><td>Integer</td>
<td>This is a positive integer, given in milliseconds, that specifies an
interval over which the moving average of the bandwidth will be calculated and
logged.</td></tr>
<tr><td>matrix_size</td><td>Integer</td>
<td>Size of the matrices of the SGEMM operations. The default value is
5760.</td></tr>
</table>

@subsection usg122 12.2 Output

Module specific output keys are described in the table below:

<table>
<tr><th>Output Key</th> <th>Type</th><th> Description</th></tr>
<tr><td>target_stress</td><td>Time Series Floats</td>
<td>The average gflops over the last log interval.</td></tr>
<tr><td>max_gflops</td><td>Float</td>
<td>The maximum sustained performance obtained by the GPU during the
test.</td></tr>
<tr><td>stress_violations</td><td>Integer</td>
<td>The number of gflops readings that violated the tolerance of the test after
the ramp interval.</td></tr>
<tr><td>flops_per_op</td><td>Integer</td>
<td>Flops (floating point operations) per operation queued to the GPU queue.
One operation is one call to SGEMM/DGEMM.</td></tr>
<tr><td>bytes_copied_per_op</td><td>Integer</td>
<td>Calculated number of ops/second necessary to achieve target
gigaflops.</td></tr>
<tr><td>try_ops_per_sec</td><td>Float</td>
<td>Calculated number of ops/second necessary to achieve target
gigaflops.</td></tr>
<tr><td>pass</td><td>Bool</td>
<td>'true' if the GPU achieves its desired sustained performance
level.</td></tr>
</table>

An informational message indicating will be emitted when the test starts
execution:

    [INFO ][<timestamp>][<action name>] gst <gpu id> start <target_stress> copy matrix: <copy_matrix>


During the execution of the test, informational output providing the moving
average the GPU(s) gflops will be logged at each log_interval:

    [INFO ][<timestamp>][<action name>] gst Gflops: <interval_gflops>

When the target gflops is achieved, the following message will be logged:

    [INFO ][<timestamp>][<action name>] gst <gpu id> target achieved <target_stress>

If the target gflops, or stress, is not achieved in the “ramp_interval”
provided, the test will terminate and the following message will be logged:

    [INFO ][<timestamp>][<action name>] gst <gpu id> ramp time exceeded <ramp_time>

In this case the test will fail.\n

If the target stress (gflops) is achieved the test will attempt to run for the
rest of the duration specified by the action, sustaining the stress load during
that time. If the stress level violates the bounds set by the tolerance level
during that time a violation message will be logged:

    [INFO ][<timestamp>][<action name>] gst <gpu id> stress violation <interval_gflops>

When the test completes, the following result message will be printed:

    [RESULT][<timestamp>][<action name>] gst <gpu id> Gflop: < max_gflops> flops_per_op:<flops_per_op> bytes_copied_per_op: <bytes_copied_per_op> try_ops_per_sec: <try_ops_per_sec> pass: <pass>

The test will pass if the target_stress is reached before the end of the
ramp_interval and the stress_violations value is less than the given
max_violations value. Otherwise, the test will fail.

@subsection usg123 12.3 Examples

When running the __GST__ module, users should provide at least an action name,
the module name (gst), a list of GPU IDs, the test duration and a target stress
value (gigaflops). Thus, the most basic configuration file looks like this:

    actions:
    - name: action_gst_1
      module: gst
      device: all
      target_stress: 3500
      duration: 8000

For the above configuration file, all the missing configuration keys will have
their default
values (e.g.: __copy_matrix=true__, __matrix_size=5760__ etc.). For more
information about the default
values please consult the dedicated sections (__3.3 Common Configuration Keys__
and __5.1 Configuration keys__).

When the __RVS__ tool runs against such a configuration file, it will do the
following:
  - run the stress test on all available (and compatible) AMD GPUs, one after
the other
  - log a start message containing the GPU ID, the __target_stress__ and the
value of the __copy_matrix__:<br />

    [INFO  ] [164337.932824] action_gst_1 gst 50599 start 3500.000000 copy matrix:true

  - emit, each __log_interval__ (e.g.: 1000ms), a message containing the
gigaflops value that the current GPU achieved:<br />

    [INFO  ] [164355.111207] action_gst_1 gst 33367 Gflops 3535.670231

  - log a message as soon as the current GPU reaches the given __target_stress__:

    [INFO  ] [164350.804843] action_gst_1 gst 33367 target achieved 500.000000

  - log a __ramp time exceeded__ message if the GPU was not able to reach the
__target_stress__ in the __ramp_interval__ time frame (e.g.: 5000). In such a
case, the test will also terminate:<br/>

    [INFO  ] [164013.788870] action_gst_1 gst 3254 ramp time exceeded 5000

  - log the test result, when the stress test completes. The message contains
the test's overall result and some other statistics according to __5.2 Output
keys__:<br />

    [RESULT] [164355.647523] action_gst_1 gst 33367 Gflop: 4066.020766 flops_per_op: 382.205952x1e9 bytes_copied_per_op: 398131200 try_ops_per_sec: 9.157367 pass: TRUE

  - log a __stress violation__ message when the current gigaflops (for the last
__log_interval__, e.g.; 1000ms) violates the bounds set by the __tolerance__
configuration key (e.g.: 0.1). Please note that this message is not logged
during the __ramp_interval__ time frame:<br />

    [INFO  ] [164013.788870] action_gst_1 gst 3254 stress violation 2500

If a mandatory configuration key is missing, the __RVS__ tool will log an error
message and terminate the execution of the current module. For example, the
following configuration file will cause the __RVS__ to terminate with the
following error message:<br /> __RVS-GST: action: action_gst_1  key
'target_stress' was not found__

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
  - will run the stress test only for the GPUs having the ID 50599 or 33367. To
get all the available GPU IDs, run __RVS__ tool with __-g__ option
  - will run the test on the selected GPUs, one after the other
  - will run each test, 12 times
  - will only copy the matrices to the GPUs at the beginning of the test
  - will wait 100ms before each test execution
  - will try to reach 5000 gflops in maximum 3000ms
  - if __target_stress__ (5000) is achieved in the __ramp_interval__ (3000 ms)
it will attempt to run the test for the rest of the duration, sustaining the
stress load during that time
  - will allow a 7% __target_stress__ __tolerance__ (each __target_stress__
violation will generate a __stress violation__ message as shown in the first
example)
  - will allow only 2 __target_stress__ violations. Exceeding the
__max_violations__ will not terminate the test, but the __RVS__ will mark the
test result as "fail".

The output for such a configuration key may look like this:

__[INFO  ] [172061.758830] action_1 gst 50599 start 5000.000000 copy
matrix:false__<br />
__[INFO  ] [172063.547668] action_1 gst 50599 Gflops 6471.614725__<br />
__[INFO  ] [172064.577715] action_1 gst 50599 target achieved 5000.000000__<br
/>
__[INFO  ] [172065.609224] action_1 gst 50599 Gflops 5189.993529__<br />
__[INFO  ] [172066.634360] action_1 gst 50599 Gflops 5220.373979__<br />
__[INFO  ] [172067.659262] action_1 gst 50599 Gflops 5225.472000__<br />
__[INFO  ] [172068.694305] action_1 gst 50599 Gflops 5169.935583__<br />
__[RESULT] [172069.573967] action_1 gst 50599 Gflop: 6471.614725 flops_per_op:
382.205952x1e9 bytes_copied_per_op: 398131200 try_ops_per_sec: 13.081952 pass:
TRUE__<br />
__[INFO  ] [172069.574369] action_1 gst 33367 start 5000.000000 copy
matrix:false__<br />
__[INFO  ] [172071.409483] action_1 gst 33367 Gflops 6558.348080__<br />
__[INFO  ] [172072.438104] action_1 gst 33367 target achieved 5000.000000__<br
/>
__[INFO  ] [172073.465033] action_1 gst 33367 Gflops 5215.285895__<br />
__[INFO  ] [172074.501571] action_1 gst 33367 Gflops 5164.945297__<br />
__[INFO  ] [172075.529468] action_1 gst 33367 Gflops 5210.207720__<br />
__[INFO  ] [172076.558102] action_1 gst 33367 Gflops 5205.139424__<br />
__[RESULT] [172077.448182] action_1 gst 33367 Gflop: 6558.348080 flops_per_op:
382.205952x1e9 bytes_copied_per_op: 398131200 try_ops_per_sec: 13.081952 pass:
TRUE__<br />

When setting the __parallel__ to false, the __RVS__ will run the stress tests on
all selected GPUs in parallel and the output may look like this:

__[INFO  ] [173381.407428] action_1 gst 50599 start 5000.000000 copy
matrix:false__<br />
__[INFO  ] [173381.407744] action_1 gst 33367 start 5000.000000 copy
matrix:false__<br />
__[INFO  ] [173383.245771] action_1 gst 33367 Gflops 6558.348080__<br />
__[INFO  ] [173383.256935] action_1 gst 50599 Gflops 6484.532120__<br />
__[INFO  ] [173384.274202] action_1 gst 33367 target achieved 5000.000000__<br
/>
__[INFO  ] [173384.286014] action_1 gst 50599 target achieved 5000.000000__<br
/>
__[INFO  ] [173385.301038] action_1 gst 33367 Gflops 5215.285895__<br />
__[INFO  ] [173385.315794] action_1 gst 50599 Gflops 5200.080980__<br />
__[INFO  ] [173386.337638] action_1 gst 33367 Gflops 5164.945297__<br />
__[INFO  ] [173386.353274] action_1 gst 50599 Gflops 5159.964636__<br />
__[INFO  ] [173387.365494] action_1 gst 33367 Gflops 5210.207720__<br />
__[INFO  ] [173387.383437] action_1 gst 50599 Gflops 5195.032357__<br />
__[INFO  ] [173388.401250] action_1 gst 33367 Gflops 5169.935583__<br />
__[INFO  ] [173388.421599] action_1 gst 50599 Gflops 5154.993572__<br />
__[RESULT] [173389.282710] action_1 gst 33367 Gflop: 6558.348080 flops_per_op:
382.205952x1e9 bytes_copied_per_op: 398131200 try_ops_per_sec: 13.081952 pass:
TRUE__<br />
__[RESULT] [173389.305479] action_1 gst 50599 Gflop: 6484.532120 flops_per_op:
382.205952x1e9 bytes_copied_per_op: 398131200 try_ops_per_sec: 13.081952 pass:
TRUE__<br />

It is important that all the configuration keys will be adjusted/fine-tuned
according to the actual GPUs and HW platform capabilities. For example, a matrix
size of 5760 should fit the VEGA 10 GPUs while 8640 should work with the VEGA 20
GPUs.

@section usg13 13 IET Module

The Input EDPp Test can be used to characterize the peak power capabilities of a
GPU to different levels of use. This tool can leverage the functionality of the
GST to drive the compute load on the GPU, but the test will use different
configuration and output keys and should focus on driving power usage rather
than calculating compute load. The purpose of the IET module is to bring the
GPU(s) to a preconfigured power level in watts by gradually increasing the
compute load on the GPUs until the desired power level is achieved. This
verifies that the GPUs can sustain a power level for a reasonable amount of time
without problems like thermal violations arising.\n

This module should be used in conjunction with the GPU Monitor, to watch for
thermal, power and related anomalies while the target GPU(s) are under realistic
load conditions. By setting the appropriate parameters a user can ensure that
all GPUs in a node or cluster reach desired performance levels. Further analysis
of the generated stats can also show variations in the required power, clocks or
temperatures to reach these targets, and thus highlight GPUs or nodes that are
operating less efficiently.

@subsection usg131 13.1 Module Specific Keys

Module specific keys are described in the table below:

<table>
<tr><th>Config Key</th> <th>Type</th><th> Description</th></tr>
<tr><td>target_power</td><td>Float</td>
<td>This is a floating point value specifying the target sustained power level
for the test.</td></tr>
<tr><td>ramp_interval</td><td>Integer</td>
<td>This is an time interval, specified in milliseconds, given to the test to
determine the compute load that will sustain the target power. The default value
is 5000 (5 seconds). This time is counted against the duration of the test.
</td></tr>
<tr><td>tolerance</td><td>Float</td>
<td>A value indicating how much the target_power can fluctuate after the ramp
period for the test to succeed. The default value is 0.1 or 10%.
</td></tr>
<tr><td>max_violations</td><td>Integer</td>
<td>The number of tolerance violations that can occur after the ramp_interval
for the test to still pass. The default value is 0.</td></tr>
<tr><td>sample_interval</td><td>Integer</td>
<td>The sampling rate for target_power values given in milliseconds. The default
value is 100 (.1 seconds).
</td></tr>
<tr><td>log_interval</td><td>Integer</td>
<td>This is a positive integer, given in milliseconds, that specifies an
interval over which the moving average of the bandwidth will be calculated and
logged.</td></tr>
</table>


@subsection usg132 13.2 Output

Module specific output keys are described in the table below:

<table>
<tr><th>Output Key</th> <th>Type</th><th> Description</th></tr>
<tr><td>current_power</td><td>Time Series Floats</td>
<td>The current measured power of the GPU.</td></tr>
<tr><td>power_violations</td><td>Integer</td>
<td>The number of power reading that violated the tolerance of the test after
the ramp interval.
</td></tr>
<tr><td>pass</td><td>Bool</td>
<td>'true' if the GPU achieves its desired sustained power level in the ramp
interval.</td></tr>
</table>

@subsection usg133 13.3 Examples

**Example 1:**

A regular IET configuration file looks like this:

    actions:
    - name: action_1
      device: all
      module: iet
      parallel: false
      count: 2
      wait: 100
      duration: 10000
      ramp_interval: 5000
      sample_interval: 500
      log_interval: 500
      max_violations: 1
      target_power: 135
      tolerance: 0.1
      matrix_size: 5760

*Please note:*
- when setting the 'device' configuration key to 'all', the RVS will detect all the AMD compatible GPUs and run the test on all of them
- the test will run 2 times on each GPU (count = 2)
- only one power violation is allowed. If the total number of violations is bigger than 1 the IET test result will be marked as 'failed'

When the RVS tool runs against such a configuration file, it will do the following:
- run the test on all AMD compatible GPUs

- log a start message containing the GPU ID and the target_power, e.g.:

    [INFO ] [167316.308057] action_1 iet 50599 start 135.000000

- emit, each log_interval (e.g.: 500ms), a message containing the power for the current GPU

    [INFO ] [167319.266707] action_1 iet 50599 current power 136.878342

- log a message as soon as the current GPU reaches the given target_power

    [INFO ] [167318.793062] action_1 iet 50599 target achieved 135.000000

- log a 'ramp time exceeded' message if the GPU was not able to reach the target_power in the ramp_interval time frame (e.g.: 5000ms). In such a case, the test will also terminate

    [INFO ] [167648.832413] action_1 iet 50599 ramp time exceeded 5000

- log a 'power violation message' when the current power (for the last sample_interval, e.g.; 500ms) violates the bounds set by the tolerance configuration key (e.g.: 0.1). Please note that this message is never logged during the ramp_interval time frame

    [INFO ] [161251.971277] action_1 iet 3254 power violation 73.783211

- log the test result, when the stress test completes.

    [RESULT] [167305.260051] action_1 iet 33367 pass: TRUE

The output for such a configuration file may look like this:

    [INFO ] [167261.27161 ] action_1 iet 33367 start 135.000000
    [INFO ] [167263.516803] action_1 iet 33367 current power 136.934479
    [INFO ] [167263.521355] action_1 iet 33367 target achieved 135.000000
    [INFO ] [167264.16925 ] action_1 iet 33367 current power 138.421844
    [INFO ] [167264.517018] action_1 iet 33367 current power 138.394608
    ...
    [INFO ] [167271.518402] action_1 iet 33367 current power 139.231918
    [RESULT] [167272.67686 ] action_1 iet 33367 pass: TRUE
    [INFO ] [167272.68029 ] action_1 iet 3254 start 135.000000
    [INFO ] [167274.552026] action_1 iet 3254 current power 139.363525
    [INFO ] [167274.552059] action_1 iet 3254 target achieved 135.000000
    [INFO ] [167275.52168 ] action_1 iet 3254 current power 138.661453
    [INFO ] [167275.552241] action_1 iet 3254 current power 138.857635
    ...
    [INFO ] [167282.553983] action_1 iet 3254 current power 140.069687
    [RESULT] [167283.95763 ] action_1 iet 3254 pass: TRUE
    [INFO ] [167283.96158 ] action_1 iet 50599 start 135.000000
    [INFO ] [167285.532999] action_1 iet 50599 current power 137.205032
    [INFO ] [167285.543084] action_1 iet 50599 target achieved 135.000000
    [INFO ] [167286.33050 ] action_1 iet 50599 current power 136.137115
    ...
    [INFO ] [167293.534672] action_1 iet 50599 current power 139.753464
    [RESULT] [167294.131420] action_1 iet 50599 pass: TRUE


**Example 2:**

Another configuration file, which may raise some 'power violation' messages (due to the small tolerance value) looks like this

    - name: action_1
      device: all
      module: iet
      parallel: false
      count: 1
      wait: 100
      duration: 8000
      ramp_interval: 5000
      sample_interval: 700
      log_interval: 700
      max_violations: 1
      target_power: 80
      tolerance: 0.06
      matrix_size: 5760

The output for such a configuration file may look like this:

    [INFO ] [161236.677785] action_1 iet 33367 start 80.000000
    [INFO ] [161239.350055] action_1 iet 33367 current power 84.186142
    [INFO ] [161239.354542] action_1 iet 33367 target achieved 80.000000
    ...
    [INFO ] [161241.450517] action_1 iet 33367 current power 77.001945
    [INFO ] [161241.459600] action_1 iet 33367 power violation 75.163689
    [INFO ] [161242.150642] action_1 iet 33367 current power 82.063576
    [RESULT] [161245.698113] action_1 iet 33367 pass: TRUE
    [INFO ] [161245.698525] action_1 iet 3254 start 80.000000
    [INFO ] [161248.394003] action_1 iet 3254 current power 78.842796
    [INFO ] [161248.418631] action_1 iet 3254 target achieved 80.000000
    [INFO ] [161249.94149 ] action_1 iet 3254 current power 79.938454
    ...
    [INFO ] [161249.794201] action_1 iet 3254 current power 76.511711
    [INFO ] [161249.818803] action_1 iet 3254 power violation 74.279594
    [INFO ] [161250.494263] action_1 iet 3254 current power 74.615120
    ...
    [INFO ] [161254.117386] action_1 iet 3254 power violation 73.682312
    [RESULT] [161254.738939] action_1 iet 3254 pass: FALSE
    [INFO ] [161254.739387] action_1 iet 50599 start 80.000000
    [INFO ] [161257.374079] action_1 iet 50599 current power 81.560165
    [INFO ] [161257.392085] action_1 iet 50599 target achieved 80.000000
    [INFO ] [161258.774304] action_1 iet 50599 current power 75.057304
    ...
    [INFO ] [161262.974833] action_1 iet 50599 current power 80.200668
    [RESULT] [161263.771631] action_1 iet 50599 pass: TRUE


*Important notes:*


- all the missing configuration keys (if any) will have their default values. For more information about the default values please consult the dedicated sections (3.3 Common Configuration Keys and 13.1 Module specific keys).


- if a mandatory configuration key is missing, the RVS tool will log an error message and terminate the execution of the current module. For example, if the target_power is missing, the RVS to terminate with the following error message: "RVS-IET: action: action_1 key 'target_power' was not found"


- it is important that all the configuration keys will be adjusted/fine-tuned according to the actual GPUs and HW platform capabilities.


*) for example, a matrix size of 5760 should fit the VEGA 10 GPUs while 8640 should work with the VEGA 20 GPUs


*) for small target_power values (e.g.: 30-40W), the sample_interval should be increased, otherwise the IET may fail either to achieve the given target_power or to sustain it (e.g.: ramp_interval = 1500 for target_power = 40)


*) in case there are problems reaching/sustaining the given target_power

**) please increase the ramp_interval and/or the tolerance value(s) and try again (in case of a 'ramp time exceeded' message)

**) please increase the tolerance value (in case too many 'power violation message' are logged out)



