# User guide

The ROCm Validation Suite (RVS) is a system validation and diagnostics tool
for monitoring, stress testing, detecting and troubleshooting issues that
affect the functionality and performance of AMD GPU(s) operating in a
high-performance/AI/ML computing environment. RVS is enabled using the ROCm
software stack on a compatible software and hardware platform.

RVS is a collection of tests, benchmarks, and qualification tools each
targeting a specific sub-system of the ROCm platform. The tools are
implemented in software and share a common command-line interface. Each set of
tests is implemented in a “module” which is a library encapsulating the
functionality specific to the tool. The CLI can specify the directory containing
modules to use when searching for libraries to load. Each module may have a set
of options that it defines and a configuration file that supports its execution.

## Installing RVS

RVS can be obtained by building it from source code base or by installing from
pre-built package.

### Building from source code

RVS has been developed as open source solution. Its source code and belonging
documentation can be found at AMD's GitHub page.
In order to build RVS from source code, refer
[ROCm Validation Suite GitHub
site](https://github.com/ROCm/ROCmValidationSuite)
and follow instructions in README file.

### Installing from package
Based on the OS, use the appropriate package manager to install the **rocm-validation-suite** package.
For more details, refer to the [ROCm Validation Suite GitHub site](https://github.com/ROCm/ROCmValidationSuite).

RVS package components are installed in `/opt/rocm`. Package contains:
- executable binary (located in `_install-base_/bin/rvs`)
- public shared libraries (located in `_install-base_/lib`)
- module specific shared libraries (located in `_install-base_/lib/rvs`)
- default configuration files (located in `_install-base_/share/rocm-validation-suite/conf`)
- GPU specific configuration files (located in `_install-base_/share/rocm-validation-suite/conf/<GPU folder>`)
- testscripts (located in `_install-base_/share/rocm-validation-suite/testscripts`)
- user guide (located in `_install-base_/share/rocm-validation-suite/userguide`)
- man page (located in `_install-base_/share/man`)

### Running RVS

#### Run version built from source code

```bash
cd <source folder>/build/bin
```

    Command examples

```bash
./rvs --help                       # List all options
./rvs -g                           # List supported GPUs available in the machine
./rvs -c conf/gst_single.conf      # Run GST module default test configuration
./rvs -m gst                       # Run GST module using platform-detected config
./rvs -r 3                         # Run predefined level 3 tests (range: 1–5, 5 = highest stress)
```

#### Run version pre-compiled and packaged with ROCm release

```bash
cd /opt/rocm/bin
```

    Command examples
```bash
./rvs --help                                                                      # List all options
./rvs -g                                                                          # List supported GPUs available in the machine
./rvs -c ../share/rocm-validation-suite/conf/gst_single.conf                     # Run GST default test configuration
./rvs -m gst                                                                      # Run GST module using platform-detected config
./rvs -r 3                                                                        # Run predefined level 3 tests (range: 1–5, 5 = highest stress)
```

To run a GPU-specific test configuration, use configuration files from the GPU subfolders under `/opt/rocm/share/rocm-validation-suite/conf/`:

```bash
./rvs -c ../share/rocm-validation-suite/conf/MI300X/gst_single.conf  # Run MI300X-specific GST test configuration
./rvs -c ../share/rocm-validation-suite/conf/nv32/gst_single.conf    # Run Navi 32-specific GST test configuration
```

```{note}
If present, always use GPU specific configurations instead of default test configurations.
```

## Basic concepts

### RVS architecture

RVS is implemented as a set of modules each implementing particular test
functionality. Modules are invoked from one central place (aka Launcher) which
is responsible for reading input (command line and test configuration file),
loading and running appropriate modules and providing test output. RVS
architecture is built around concept of Linux shared objects, thus
allowing for easy addition of new modules in the future.


### Available modules

#### GPU Properties – GPUP module
The GPU Properties module queries the configuration of a target device and returns the device’s static characteristics. These static values can be used to debug issues such as device support, performance and firmware problems.

#### GPU Monitor – GM module
The GPU monitor tool is capable of running on one, some or all of the GPU(s) installed and will report various information at regular intervals. The module can be configured to halt another RVS module's execution if one of the quantities exceeds a specified boundary value.

#### PCI Express State Monitor – PESM module
The PCIe State Monitor tool is used to actively monitor the PCIe interconnect between the host platform and the GPU. The module will register a “listener” on a target GPU’s PCIe interconnect, and log a message whenever it detects a state change. The PESM will be able to detect the following state changes:

1.	PCIe link speed changes
2.	GPU power state changes

#### ROCm Configuration Qualification Tool - RCQT module
The ROCm Configuration Qualification Tool ensures the platform is capable of running ROCm applications and is configured correctly. It checks the installed versions of the ROCm components and the platform configuration of the system. This includes verifying that the dependencies corresponding to the ROCm meta-packages are installed correctly.

#### PCI Express Qualification Tool – PEQT module
The PCIe Qualification Tool is used to qualify the PCIe bus on which the GPU is connected. The qualification test will be capable of determining the following characteristics of the PCIe bus interconnect to a GPU:

1.	Support for Gen 3 atomic completers
2.	DMA transfer statistics
3.	PCIe link speed
4.	PCIe link width

#### SBIOS Mapping Qualification Tool – SMQT module
The GPU SBIOS mapping qualification tool is designed to verify that a platform’s SBIOS has satisfied the BAR mapping requirements for VDI and Radeon Instinct products for ROCm support.

Refer to the “ROCm Use of Advanced PCIe Features and Overview of How BAR Memory is Used In ROCm Enabled System” web page for more information about how BAR memory is initialized by VDI and Radeon products.

#### P2P Benchmark and Qualification Tool – PBQT module
The P2P Benchmark and Qualification Tool  is designed to provide the list of all GPUs that support P2P and characterize the P2P links between peers. In addition to testing for P2P compatibility, this test will perform a peer-to-peer throughput test between all P2P pairs for performance evaluation. The P2P Benchmark and Qualification Tool will allow users to pick a collection of two or more GPUs on which to run. The user will also be able to select whether or not they want to run the throughput test on each of the pairs.

Please see the web page “ROCm, a New Era in Open GPU Computing” to find out more about the P2P solutions available in a ROCm environment.

#### PCI Express Bandwidth Benchmark – PEBB module
The PCIe Bandwidth Benchmark attempts to saturate the PCIe bus with DMA transfers between system memory and a target GPU card’s memory. The maximum bandwidth obtained is reported to help debug low bandwidth issues. The benchmark should be capable of  targeting one, some or all of the GPUs installed in a platform, reporting individual benchmark statistics for each.

#### GPU Stress Test - GST module
The GPU Stress Test runs various GEMM computations as workloads to stress the GPU FLOPS performance and check whether it meets the configured target GFLOPS. GEMM workloads shall be configured as either operation type or data type. GEMM based on operation types include SGEMM, DGEMM and HGEMM (Single/Double/Half-precision General Matrix Multiplication) - configured using operation parameter. GEMM based on data types include `fp8`, `i8`, `fp16`, `bf16`, `fp32` and `tf32` (`xf32`) - configured using data type parameter. The duration of the test is configurable, both in terms of time (how long to run) and iterations (how many times to run).

#### Input EDPp Test - IET module
The Input EDPp Test runs GEMM workloads to stress the GPU power (that is, TGP). This test is used to verify if the GPU is capable of handling max. power stress for a sustained period of time. Also checks whether GPU power reaches a set target power.

#### GPU Power Pulse Test - PULSE module
The Pulse test drives repeating **high-power** (GEMM compute) and **low-power** (idle, minimum clocks) phases at a configurable rate so that GPU power swings over time. That pattern stresses the power supply and voltage regulators with transients rather than a single sustained power level. With **parallel: true** on multiple GPUs, the module uses a CPU-side barrier and a GPU-side fine-grained barrier so that devices tend to enter the heavy phase together, increasing aggregate current steps. Power and temperature are read through **AMD SMI**. GEMM execution uses the same **rvs_blas** stack as GST/IET (**rocBLAS** or **hipBLASLt**).

```{Warning}
This is a beta feature and is not intended for production use. Pass/fail criteria are still being refined.
```

#### Memory Test - MEM module
The Memory module tests the GPU memory for hardware errors and soft errors using HIP. It consists of various tests that use algorithms like Walking 1 bit, Moving inversion and Modulo 20. The module executes the following memory tests [Algorithm, data pattern]

1. Walking 1 bit
2. Own address test
3. Moving inversions, ones & zeros
4. Moving inversions, 8 bit pattern
5. Moving inversions, random pattern
6. Block move, 64 moves
7. Moving inversions, 32 bit pattern
8. Random number sequence
9. Modulo 20, random pattern
10. Memory stress test

#### BABEL Benchmark Test - BABEL module
The Babel module executes BabelStream (synthetic GPU benchmark based on the original STREAM benchmark for CPUs) benchmark that measures memory transfer rates (bandwidth) to and from global device memory. Various benchmark tests are implemented using GPU kernels in HIP (Heterogeneous Interface for Portability) programming language.

### Command line options

Command line options are summarized in the table below:

```{note}
Command line options take precedence over the same parameters set in the configuration file. For example, if `parallel: false` is specified in the configuration file but `-p true` is passed on the command line, parallel execution will be enabled.
```

<div class="pst-scrollable-table-container">
<table class="table table--middle-left">
<tr><th class="head">Short option</th><th class="head">Long option</th><th class="head"> Description</th></tr>
<tr><td>-a</td><td>--appendLog</td><td>When generating a debug logfile, do not overwrite the content
of the current log. Use in conjuction with <b>-d</b> and <b>-l</b> options.
</td></tr>

<tr><td>-c</td><td>--config</td><td>Specify the test configuration file to use.
This is a mandatory field for test execution.
</td></tr>

<tr><td>-r</td><td>--run</td><td>Specify the predefined test level to run (1–5).
Each level selects a GPU-specific configuration file without requiring a <b>-c</b> argument.
See <a href="#test-levels">Test Levels</a> for details.
</td></tr>

<tr><td>-d</td><td>--debugLevel</td><td>Specify the debug level for the output log.
The range is 0-5 with 5 being the highest verbose level.
</td></tr>

<tr><td>-g</td><td>--listGpus</td><td>List all the GPUs available in the machine,
that RVS supports and has visibility.
</td></tr>

<tr><td>-i</td><td>--indexes</td><td>Comma-separated list of GPU IDs or SMI-based indexes to run tests on.
Overrides the <b>device</b> and <b>device_index</b> values specified in all actions of the configuration file, including the <b>all</b> value.
</td></tr>

<tr><td>-s</td><td>--selectActions</td><td>Comma separated list of action names or 0-based action
index numbers to run from the configuration file. Only the matching actions will be executed.
All other actions are skipped.
</td></tr>

<tr><td>-j</td><td>--json</td><td>Generate output file in JSON format.
if a path follows this argument, that will be used as json log file;
else a file created in <b>/var/tmp/</b> with timestamp in name.
</td></tr>

<tr><td>-l</td><td>--debugLogFile</td><td>Generate log file with output and debug information.
</td></tr>

<tr><td>-m</td><td>--module</td><td>Specify a module name to run the corresponding
platform-specific (MI-series GPUs) module configuration file. Valid modules: <b>babel</b>, <b>gpup</b>,
<b>gst</b>, <b>iet</b>, <b>mem</b>, <b>pebb</b>, <b>peqt</b>, <b>pbqt</b>, <b>rcqt</b>.
</td></tr>

<tr><td>-t</td><td>--duration</td><td>Specify the test duration (in seconds) for each action.
Overrides the <b>duration</b> value in all actions of the configuration file.
</td></tr>

<tr><td></td><td>--listTests</td><td>List the test modules present in RVS.
</td></tr>

<tr><td>-v</td><td>--verbose</td><td>Enable detailed logging. Equivalent to specifying <b>-d 5</b> option.
</td></tr>

<tr><td>-p</td><td>--parallel</td><td>Enables or Disables parallel execution across multiple GPUs.
Use this option in conjunction with <b>-c</b> option.
Accepted Values:
<b>true</b> – Enables parallel execution.
<b>false</b> – Disables parallel execution.
If no value is provided for the option, it defaults to <b>true</b>.
</td></tr>

<tr><td>-n</td><td>--numTimes</td><td>Number of times the test repeatedly executes.
Use this option in conjunction with <b>-c</b> option.
</td></tr>

<tr><td>-q</td><td>--quiet</td><td>No console output given. See logs and return
code for errors.</td></tr>

<tr><td></td><td>--version</td><td>Displays the version information and exits.
</td></tr>

<tr><td>-h</td><td>--help</td><td>Display usage information and exit.
</td></tr>

</table>
</div>

### Test Levels

When using the `-r` option, RVS automatically selects a predefined configuration file based on the detected GPU platform (for example, `conf/MI300X/levels/rvs_level_N.conf`). No `-c` argument is needed. The five levels represent progressively increasing test coverage and duration:

<div class="pst-scrollable-table-container">
<table class="table table--middle-left">
<tr><th class="head">Level</th><th class="head">Test</th><th class="head">Description</th><th class="head">Modules</th><th class="head">Typical Duration</th></tr>
<tr><td>1</td><td>Software validation</td><td>Verifies that all required ROCm packages and metapackages are correctly installed.</td><td>rcqt</td><td>Short (seconds)</td></tr>
<tr><td>2</td><td>Hardware capability checks</td><td>Checks PCIe link speed and width, performs a quick HBM read/write pass, and verifies basic PCIe and XGMI interface connectivity.</td><td>peqt, babel, pebb, pbqt</td><td>Short (seconds)</td></tr>
<tr><td>3</td><td>Basic performance</td><td>Measures HBM bandwidth across common operations, PCIe host-to-device and device-to-host throughput, bidirectional XGMI bandwidth, and limited gemm compute performance runs.</td><td>babel, pebb, pbqt, gst</td><td>Minutes</td></tr>
<tr><td>4</td><td>Extended performance</td><td>Same coverage as level 3 with longer test durations, full HBM operation set, bidirectional PCIe transfers, a basic memory test pass, and full gemm compute performance runs.</td><td>babel, pebb, pbqt, mem, gst, iet</td><td>Tens of minutes</td></tr>
<tr><td>5</td><td>Full stress test</td><td>Runs all tests from level 4 at maximum duration, adds high-iteration memory stress testing, and includes a sustained power stress test targeting peak GPU TDP. Intended for thorough pre-production validation.</td><td>babel, pebb, pbqt, mem, gst, iet</td><td>Hours</td></tr>
</table>
</div>

The exact tests and thresholds within each level vary by GPU platform.

#### Examples

Print version information and exit:
```bash
./rvs --version
```

List all available test modules:
```bash
./rvs --listTests
```
List all GPUs visible to RVS:
```bash
./rvs -g
```

Run a specific configuration file:
```bash
./rvs -c conf/gst_single.conf
```

Run a platform-specific configuration file (for example, MI350X GST):
```bash
./rvs -c conf/MI350X/gst_single.conf
```

Repeat a test 5 times in sequence (soak/stability testing):
```bash
./rvs -c conf/gst_single.conf -n 5
```

Run only the GST module using the built-in platform configuration:
```bash
./rvs -m gst
```


Run the predefined level 3 test suite (basic performance):
```bash
./rvs -r 3
```

Run the predefined level 5 (full stress) test suite on specific GPUs only:
```bash
./rvs -r 5 -i 0,1
```

Run a configuration file on specific GPUs (by index) with parallel execution enabled:
```bash
./rvs -c conf/gst_single.conf -i 0,1 -p true
```

Run only selected actions from a configuration file (by name or 0-based index):
```bash
./rvs -c conf/iet_single.conf -s action_1,action_2
./rvs -c conf/iet_single.conf -s 0,2
```

Generate JSON output to a specific file:
```bash
./rvs -c conf/gst_single.conf -j /tmp/rvs_results.json
```

Run with verbose logging and save the log to a file:
```bash
./rvs -c conf/gst_single.conf -v -l /tmp/rvs_debug.log
```

Suppress console output and write results to a JSON file (useful in CI pipelines):
```bash
./rvs -c conf/gst_single.conf --quiet -j /tmp/rvs_results.json
```

The configuration files in the top-level `conf/` folder are generic samples and may not reflect the correct thresholds or parameters for your hardware. For accurate results, use the platform-specific configuration files under `conf/<GPU>/` (for example, `conf/MI350X/`), or use the `-r` or `-m` option, which automatically selects the appropriate platform configuration.

### GPU-Specific Configurations

RVS includes optimized test configurations for a range of GPU families, organized under `conf/<GPU>/`. 
Level-based configurations (usable with `-r`) are available for: MI300X, MI300X-HF, MI308X, MI308X-HF, MI325X, MI350X, MI355X, MI350P-450W, MI350P-600W.

For MI350P, the 450W and 600W SKUs share a single PCI device ID (`0x75a8`); RVS disambiguates between them at runtime by reading the GPU's hardware power cap via amdsmi (~450 W → `MI350P-450W`, ~600 W → `MI350P-600W`). If the power cap cannot be read, RVS falls back to `MI350P-450W` and prints a warning; use `-c conf/MI350P-600W/levels/rvs_level_N.conf` explicitly if your card is the 600W SKU.

**AMD Instinct Series:**
- MI210, MI250X, MI300A, MI300X, MI308X, MI325X, MI350X, MI355X
- High-frequency variants: MI300X-HF, MI308X-HF

**AMD Radeon Series:**
- nv21, nv31, nv32, gfx1200, gfx1201
- RX9060, RX9070, RX9070GRE, R9600D

### Configuration files

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


### Common configuration keys

Common configuration keys applicable to most module are summarized in the
table below:

<div class="pst-scrollable-table-container">
<table class="table table--middle-left">
<tr><th class="head">Config Key</th> <th class="head">Type</th><th class="head"> Description</th></tr>
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
wait between executions, in milliseconds. Some
modules will ignore this parameter. If the
count key is not specified, this key is ignored.</td></tr>

<tr><td>duration</td><td>Integer</td><td>This parameter overrides the count key, if
specified. This indicates how long the test
should run, given in milliseconds. Some
modules will ignore this parameter.</td></tr>


<tr><td>module</td><td>String</td><td>This parameter specifies the module that
will be used in the execution of the action. Each module has a set of sub-tests
or sub-actions that can be configured based on its specific
parameters.</td></tr>

<tr><td>log_interval</td><td>Integer</td><td>This specifies how often, in
milliseconds, the module emits a progress or status log message during a
running test. If a value isn't specified the default is 1000 ms. Some modules
will ignore this parameter.</td></tr>
</table>
</div>

## GPUP module
The GPU properties module provides an interface to easily dump the static
characteristics of a GPU. This information is stored in the sysfs file system
for the kfd, with the following path:

    /sys/class/kfd/kfd/topology/nodes/<node id>

Each of the GPU nodes in the directory is identified with a number,
indicating the device index of the GPU. This module will ignore count, duration
or wait key values.

### Module specific keys

<div class="pst-scrollable-table-container">
<table class="table table--middle-left">
  <thead>
    <tr>
      <th class="head">Config Key</th>
      <th class="head">Type</th>
      <th class="head">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>properties</td>
      <td>Collection of Strings</td>
      <td>
        The properties key specifies what configuration property or properties the
        query is interested in. Possible values are:
        <ul>
          <li>all - collect all settings</li>
          <li>gpu_id</li>
          <li>cpu_cores_count</li>
          <li>simd_count</li>
          <li>mem_banks_count</li>
          <li>caches_count</li>
          <li>io_links_count</li>
          <li>cpu_core_id_base</li>
          <li>simd_id_base</li>
          <li>max_waves_per_simd</li>
          <li>lds_size_in_kb</li>
          <li>gds_size_in_kb</li>
          <li>wave_front_size</li>
          <li>array_count</li>
          <li>simd_arrays_per_engine</li>
          <li>cu_per_simd_array</li>
          <li>simd_per_cu</li>
          <li>max_slots_scratch_cu</li>
          <li>vendor_id</li>
          <li>device_id</li>
          <li>location_id</li>
          <li>drm_render_minor</li>
          <li>max_engine_clk_fcompute</li>
          <li>local_mem_size</li>
          <li>fw_version</li>
          <li>capability</li>
          <li>max_engine_clk_ccompute</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>io_links-properties</td>
      <td>Collection of Strings</td>
      <td>
        The properties key specifies what configuration property or properties the
        query is interested in. Possible values are:
        <ul>
          <li>all - collect all settings</li>
          <li>count - the number of io_links</li>
          <li>type</li>
          <li>version_major</li>
          <li>version_minor</li>
          <li>node_from</li>
          <li>node_to</li>
          <li>weight</li>
          <li>min_latency</li>
          <li>max_latency</li>
          <li>min_bandwidth</li>
          <li>max_bandwidth</li>
          <li>recommended_transfer_size</li>
          <li>flags</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>
</div>

### Output

Module specific output keys are described in the table below:

<div class="pst-scrollable-table-container">
<table class="table table--middle-left">
<tr><th class="head">Output Key</th> <th class="head">Type</th><th class="head"> Description</th></tr>
<tr><td>properties-values</td><td>Collection of Integers</td>
<td>The collection will contain a positive integer value for each of the valid
properties specified in the properties config key.</td></tr>
<tr><td>io_links-propertiesvalues</td><td>Collection of Integers</td>
<td>The collection will contain a positive integer value for each of the valid
properties specified in the io_links-properties config key.</td></tr>
</table>
</div>

Each of the settings specified has a positive integer value. For each
setting requested in the properties key a message with the following format will
be returned:

    [RESULT][<timestamp>][<action name>] gpup <gpu id> <property> <property value>

For each setting in the io_links-properties key a message with the following
format will be returned:

    [RESULT][<timestamp>][<action name>] gpup <gpu id> <io_link id> <property> <property value>

### Examples

**Example 1:**

Consider action:

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

This action explicitly lists some of the properties.
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


## GM module
The GPU monitor module can be used monitor and characterize the response of a
GPU to different levels of use. This module is intended to run concurrently with
other actions, and provides a start and stop configuration key to start the
monitoring and then stop it after testing has completed. The module can also be
configured with bounding box values for interested GPU parameters. If any of the
GPU’s parameters exceed the bounding values on a specific GPU an INFO warning
message will be printed to stdout while the bounding value is still exceeded.

### Module specific keys

<div class="pst-scrollable-table-container">
<table class="table table--middle-left">
  <thead>
    <tr>
      <th class="head">Config Key</th>
      <th class="head">Type</th>
      <th class="head">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>monitor</td>
      <td>Bool</td>
      <td>
        If this key is set to true, the GM module will start monitoring on
        specified devices. If this key is set to false, all other keys are ignored
        and monitoring of the specified device will be stopped.
      </td>
    </tr>
    <tr>
      <td>metrics</td>
      <td>
        Collection of Structures, specifying the metric, if there are bounds and the
        bound values. The structures have the following format:
        <pre>{String, Bool, Integer, Integer}</pre>
      </td>
      <td>
        The set of metrics to monitor during the monitoring period. Example values are:
        <ul>
          <li>{'temp', 'true', max_temp, min_temp}</li>
          <li>{'clock', 'false', max_clock, min_clock}</li>
          <li>{'mem_clock', 'true', max_mem_clock, min_mem_clock}</li>
          <li>{'fan', 'true', max_fan, min_fan}</li>
          <li>{'power', 'true', max_power, min_power}</li>
        </ul>
        The set of upper bounds for each metric are specified as an integer. The units and
        values for each metric are:
        <ul>
          <li>temp - degrees Celsius</li>
          <li>clock - MHz</li>
          <li>mem_clock - MHz</li>
          <li>fan - Integer between 0 and 255</li>
          <li>power - Power in Watts</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>sample_interval</td>
      <td>Integer</td>
      <td>
        If this key is specified metrics will be sampled at the given rate. The units for
        the sample_interval are milliseconds. The default value is 1000.
      </td>
    </tr>
    <tr>
      <td>log_interval</td>
      <td>Integer</td>
      <td>
        If this key is specified informational messages will be emitted at the given
        interval, providing the current values of all parameters specified. This parameter
        must be equal to or greater than the sample rate. If this value is not specified,
        no logging will occur.
      </td>
    </tr>
    <tr>
      <td>terminate</td>
      <td>Bool</td>
      <td>
        If the terminate key is true the GM monitor will terminate
        the RVS process when a bounds violation is encountered on any of the metrics specified.
      </td>
    </tr>
    <tr>
      <td>force</td>
      <td>Bool</td>
      <td>
        If true and terminate key is also true the
        RVS process will terminate immediately. <strong>Note:</strong> this may cause
        resource leaks within GPUs.
      </td>
    </tr>
  </tbody>
</table>
</div>

### Output

Module specific output keys are described in the table below:

<div class="pst-scrollable-table-container">
<table class="table table--middle-left">
<tr><th class="head">Output Key</th> <th class="head">Type</th><th class="head"> Description</th></tr>
<tr><td>metric_values</td><td>Time Series Collection of Result
Integers</td><td>A collection of integers containing the result values for each
of the metrics being monitored. </td></tr>
<tr><td>metric_violations</td><td>Collection of Result Integers </td><td>A
collection of integers containing the violation count for each of the metrics
being monitored. </td></tr>
<tr><td>metric_average</td><td>Collection of Result Integers </td><td></td></tr>
</table>
</div>

When monitoring is started for a target GPU, a result message is logged
with the following format:

    [RESULT][<timestamp>][<action name>] gm <gpu id> started

In addition, an informational message is provided for each metric
being monitored:

    [INFO ][<timestamp>][<action name>] gm <gpu id> monitoring <metric> bounds min:<min_metric> max: <max_metric>

During the monitoring informational output regarding the metrics of the GPU will
be sampled at every interval specified by the sample_rate key. If a bounding box
violation is discovered during a sampling interval, a warning message is
logged with the following format:

    [INFO ][<timestamp>][<action name>] gm <gpu id> <metric> bounds violation <metric value>

If the log_interval value is set an information message for each metric is
logged at every interval using the following format:

    [INFO ][<timestamp>][<action name>] gm <gpu id> <metric> <metric_value>

When monitoring is stopped for a target GPU, a result message is logged
with the following format:

    [RESULT][<timestamp>][<action name>] gm <gpu id> gm stopped

The following messages, reporting the number of metric violations that were
sampled over the duration of the monitoring and the average metric value is
reported:

    [RESULT][<timestamp>][<action name>] gm <gpu id> <metric> violations <metric_violations>
    [RESULT][<timestamp>][<action name>] gm <gpu id> <metric> average <metric_average>

### Examples

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

This configuration is similar to that in *Example 1* but has explicitly
given values for *sample_interval* and *log_interval*. Output is similar to
the previous one but averaging and the printout are performed at a different
rate.

**Example 3:**

Consider action with syntax error ('temp' key is missing lower value):

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


## PESM module
The PCIe State Monitor (PESM) tool is used to actively monitor the PCIe
interconnect between the host platform and the GPU. The module registers
“listener” on a target GPUs PCIe interconnect, and log a message whenever it
detects a state change. The PESM is able to detect the following state changes:

1. PCIe link speed changes
2. GPU device power state changes

This module is intended to run concurrently with other actions, and provides a
‘start’ and ‘stop’ configuration key to start the monitoring and then stop it
after testing has completed. For information on GPU power state monitoring, 
see PCI Power Management Capability Structure, Gen 3 spec, device states D0-D3. 
For information on link status changes see Status Register (Offset 12h), Gen 3 spec.

Monitoring is performed by polling respective PCIe registers roughly every 1ms
(one millisecond).

### Module Specific Keys

<div class="pst-scrollable-table-container">
<table class="table table--middle-left">
<tr><th class="head">Config Key</th> <th class="head">Type</th><th class="head"> Description</th></tr>
<tr><td>monitor</td><td>Bool</td><td>This key is set to true, the PESM
module will start monitoring on specified devices. If this key is set to false,
all other keys are ignored and monitoring will be stopped for all devices.</td>
</tr>
<tr><td>debugwait</td><td>Integer</td><td>Debug wait period in milliseconds
inserted before monitoring begins. Intended for development and debugging use.
The default value is 0 (no wait).</td></tr>
</table>
</div>

### Output

Module specific output keys are described in the table below:

<div class="pst-scrollable-table-container">
<table class="table table--middle-left">
<tr><th class="head">Output Key</th> <th class="head">Type</th><th class="head"> Description</th></tr>
<tr><td>state</td><td>String</td><td>A string detailing the current power state
of the GPU or the speed of the PCIe link.</td></tr>
</table>
</div>

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

### Examples

**Example 1**

Here is a typical check utilizing PESM functionality:

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

-  `action_1` will initiate monitoring on all devices by setting key `monitor` to `true`
-  `action_2` will start GPU stress test
-  `action_3` will stop monitoring

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


This file has an invalid entry in `deviceid` key.
If execute, an error will be reported:

    RVS-PESM: action: act1  invalide 'deviceid' key value: xxx


## RCQT module


RCQT ensures the platform is capable of running ROCm applications and is 
configured correctly. It checks the installed versions of the ROCm components
and the platform configuration of the system.
This includes checking the dependencies corresponding 
to the ROCm meta-packages are installed correctly.
The purpose of the RCQT is to provide an extensible, OS
independent and scriptable interface capable for performing the configuration
checks required for ROCm support. The checks in this module do not target a
specific device.
\n\n
Two types of actions are performed by RCQT.
1) Metapackage Check
metapackage-validation: This will check the installation of the mentioned 
metapackages and their dependencies and their respective versions as required
by metapackage. List of metapackages are provided with key `package`

2) Packages installation check
packagelist-install-validation: This action checks if the package is installed.
  Packages are provided against key `rpmpackagelist` and `debpackagelist`

This feature is used to check installed packages on the system. It provides
checks for installed packages and the currently available package versions, if
applicable.

#### Metapackage Check Specific Keys

Input keys are described in the table below:

<div class="pst-scrollable-table-container">
<table class="table table--middle-left">
<tr><th class="head">Config Key</th> <th class="head">Type</th><th class="head"> Description</th></tr>
<tr><td>package</td><td>Collection of Strings</td>
<td>Specifies the list of metapackages to check. This key is required.</td></tr>
</table>
</div>

#### Output

Output keys are described in the table below for each metapackage 
along with versions of each sub package:

<div class="pst-scrollable-table-container">
<table class="table table--middle-left">
<tr><th class="head">Output Key</th> <th class="head">Type</th><th class="head"> Description</th></tr>
<tr><td>Total packages validated</td><td>Integer</td>
<td>total dependency packages under the said metapackage 
</td></tr>
<tr><td>Installed packages</td><td>Integer</td>
<td>installed dependency packages under the said metapackage
</td></tr>
<tr><td>Missing packages </td><td>Integer</td>
<td>missing packages under the said metapackage
</td></tr>
<tr><td>Version mismatch packages</td><td>Integer</td>
<td>installed dependency packages but with wrong versions
</td></tr>
</table>
</div>

The check will emit a result message with the following format:

    Meta package <metapackage-name> :
    Package <dep-package1> installed version is <version>
    Package <dep-package2> installed version is <version>
    Package <dep-package3> installed version is <version>
    Meta package validation complete :
        Total packages validated     : <3>
        Installed packages           : <3>
        Missing packages             : <0>
        Version mismatch packages    : <0>

#### Examples

**Example 1:**

In this example, given package has all dependencies installed.

    actions:
    - name: metapackage-validation
      module: rcqt
      package: rocm-ml-sdk

The output for such configuration is:

    [RESULT] [3648664.1164  ] Action name :metapackage-validation
    [RESULT] [3648664.1363  ] Module name :rcqt

    Meta package rocm-ml-sdk :
    Package miopen-hip-dev installed version is 3.3.0.60300
    Package rocm-core installed version is 6.3.0.60300
    Package rocm-hip-sdk installed version is 6.3.0.60300
    Package rocm-ml-libraries installed version is 6.3.0.60300
    Meta package validation complete :
        Total packages validated     : 4
        Installed packages           : 4
        Missing packages             : 0
        Version mismatch packages    : 0


For other cases, we will see mismatched/missing packages printed
with respective count

### Packages installation check

This action checks if the package is installed.
  Packages are provided against key `rpmpackagelist` and `debpackagelist`

#### Packages installation Specific Keys

Input keys are described in the table below:

<div class="pst-scrollable-table-container">
<table class="table table--middle-left">
<tr><th class="head">Config Key</th> <th class="head">Type</th><th class="head"> Description</th></tr>
<tr><td>rpmpackagelist</td><td>Collection of Strings</td>
<td>Specifies the packages checked if installed on system for rhel family.</td></tr>
<tr><td>debpackagelist</td><td>Collection of Strings</td>
<td>Specifies the packages checked if installed on system for ubuntu family.
</td></tr>
</table>
</div>

#### Output

Output keys are described in the table below:

<div class="pst-scrollable-table-container">
<table class="table table--middle-left">
<tr><th class="head">Output Key</th> <th class="head">Type</th><th class="head"> Description</th></tr>
<tr><td>Package</td><td>String</td>
<td>Name of checked package
</td></tr>
<tr><td>version</td><td>Floating Number</td>
<td>Installed version of the package
</td></tr>
<tr><td>Missing packages</td><td>Integer</td>
<td>Number of packages not installed.
</td></tr>
<tr><td>Installed packages</td><td>Integer</td>
<td>Number of packages installed.
</td></tr>
</table>
</div>

#### Examples

**Example 1:**

In this example, all given packages are installed.

    actions:
    - name: packagelist-install-validation
      device: all
      module: rcqt
      rpmpackagelist: rocm-hip-libraries rocm-core

The output for such configuration is:

    [RESULT] [496559.219160] Action name :packagelist-install-validation
    [RESULT] [496559.219161]  Module name :rcqt

    Package rocm-hip-libraries installed version is 6.3.0.60300
    Package rocm-core installed version is 6.3.0.60300
    Packages install validation complete :
        Missing packages      : 0
        Installed packages    : 2


## PEQT module

PCI Express Qualification Tool module targets and qualifies the configuration of
the platforms PCIe connections to the GPUs. The purpose of the PEQT module is to
provide an extensible, OS independent and scriptable interface capable of
performing the PCIe interconnect configuration checks required for ROCm support
of GPUs. This information can be obtained through the sysfs PCIe interface or by
using the PCIe development libraries to extract values from various PCIe
control, status and capabilities registers. These registers are specified in the
PCI Express Base Specification, Revision 3. Iteration keys, i.e. count, wait and
duration will be ignored for actions using the PEQT module.

```{note}
The PEQT module requires elevated privileges. Run RVS with `sudo` when using this module.
```

### Module specific keys
Module specific output keys are described in the table below:

<div class="pst-scrollable-table-container">
<table class="table table--middle-left">
  <thead>
    <tr>
      <th class="head">Config Key</th>
      <th class="head">Type</th>
      <th class="head">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>capability</td>
      <td>
        Collection of Structures with the following format:
        <pre>{String, String}</pre>
      </td>
      <td>
        The PCIe capability key contains a collection of structures that specify
        which PCIe capability to check and the expected value of the capability. A check
        structure must contain the PCIe capability value, but an expected value may be
        omitted. The value of all valid capabilities that are a part of this collection
        will be entered into the capability_value field. Possible capabilities,
        and their value types are:
        <ul>
          <li>link_cap_max_speed</li>
          <li>link_cap_max_width</li>
          <li>link_stat_cur_speed</li>
          <li>link_stat_neg_width</li>
          <li>slot_pwr_limit_value</li>
          <li>slot_physical_num</li>
          <li>bus_id</li>
          <li>atomic_op_32_completer</li>
          <li>atomic_op_64_completer</li>
          <li>atomic_op_128_CAS_completer</li>
          <li>atomic_op_routing</li>
          <li>dev_serial_num</li>
          <li>kernel_driver</li>
          <li>pwr_base_pwr</li>
          <li>pwr_rail_type</li>
          <li>device_id</li>
          <li>vendor_id</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>
</div>

The expected value String is a regular expression that is used to check the
actual value of the capability.

### Output
Module specific output keys are described in the table below:

<div class="pst-scrollable-table-container">
<table class="table table--middle-left">
<tr><th class="head">Output Key</th> <th class="head">Type</th><th class="head"> Description</th></tr>
<tr><td>capability_value</td><td>Collection of Strings</td>
<td>For each of the capabilities specified in the capability key, the actual
value of the capability will be returned, represented as a String.</td></tr>
<tr><td>pass</td><td>String</td> <td>'true' if all of the properties match the
values given, 'false' otherwise.</td></tr>
</table>
</div>

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

### Examples

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

```{note}
- When setting the `device` configuration key to `all`, the RVS will detect all the AMD compatible GPUs and run the test on all of them
- There are no regular expression for this `.conf` file, therefore RVS will report `TRUE` if at least one AMD compatible GPU is registered within the system. Otherwise it will report `FALSE`.
```

The Power Budgeting capability is a dynamic one, having the following form:

    <PM_State>_<Type>_<Power rail>

Where:

    PM_State = D0/D1/D2/D3
    Type=PMEAux/Auxiliary/Idle/Sustained/Maximum
    PowerRail = Power_12V/Power_3_3V/Power_1_5V_1_8V/Thermal

When the RVS tool runs against such a configuration file, it will query for the
all the PCIe capabilities specified under the capability list (and log the
corresponding values) for all the AMD compatible GPUs. For those PCIe
capabilities that are not supported by the HW platform were the RVS is running,
a "NOT SUPPORTED" message will be logged.

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

For this example, the expected PEQT check result is `TRUE` if:

- At least one AMD compatible GPU is registered within the system and:
- All `<link_cap_max_speed>` values for all AMD compatible GPUs match the given regular expression and
- All `<link_stat_cur_speed>` values for all AMD compatible GPUs match the given regular expression

Please note that the `<slot_pwr_limit_value>` regular expression is not valid and
will be skipped without affecting the PEQT module's check result, however, an
error will be logged out.

**Example 3:**

Another example with even more regular expressions is given below. The expected
PEQT check result is TRUE if at least one AMD compatible GPU having the ID 3254
or 33367 is registered within the system and all the PCIe capabilities values
match their corresponding regular expressions.

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

## SMQT module
The GPU SBIOS mapping qualification tool is designed to verify that a platform’s
SBIOS has satisfied the BAR mapping requirements for VDI and Radeon Instinct
products for ROCm support. These are the current BAR requirements:

**BAR 1: GPU Frame Buffer BAR**

In this example it happens to be 256M, but
typically this will be size of the GPU memory (typically 4GB+). This BAR has to
be placed < 2^40 to allow peer- to-peer access from other GFX8 AMD GPUs. For
GFX9 (Vega GPU) the BAR has to be placed < 2^44 to allow peer-to-peer access
from other GFX9 AMD GPUs.

**BAR 2: Doorbell BAR**

The size of the BAR is typically will be < 10MB (currently
fixed at 2MB) for this generation GPUs. This BAR has to be placed < 2^40 to
allow peer-to-peer access from other current generation AMD GPUs.

**BAR 3: IO BAR**

This is for legacy VGA and boot device support, but since this
the GPUs in this project are not VGA devices (headless), this is not a concern
even if the SBIOS does not setup.

**BAR 4: MMIO BAR**

This is required for the AMD Driver SW to access the
configuration registers. Since the reminder of the BAR available is only 1 DWORD
(32bit), this is placed < 4GB. This is fixed at 256KB.

**BAR 5: Expansion ROM**

This is required for the AMD Driver SW to access the
GPU’s video-BIOS. This is currently fixed at 128KB.

Refer to the ROCm Use of Advanced PCIe Features and Overview of How BAR Memory
is Used In ROCm Enabled System page for more information about how BAR
memory is initialized by VDI and Radeon products. Iteration keys, for example, count,
wait and duration will be ignored.

### Module specific keys

Module specific output keys are described in the table below:

<div class="pst-scrollable-table-container">
<table class="table table--middle-left">
<tr><th class="head">Config Key</th> <th class="head">Type</th><th class="head"> Description</th></tr>
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
</div>

### Output

Module specific output keys are described in the table below:

<div class="pst-scrollable-table-container">
<table class="table table--middle-left">
<tr><th class="head">Output Key</th> <th class="head">Type</th><th class="head"> Description</th></tr>
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
</div>

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


### Examples

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

In this example, BAR sizes reported by GPUs match those listed in configuration
key except for the BAR5, hence the test fails.

## PBQT module

The P2P Qualification Tool is designed to provide the list of all GPUs that
support P2P and characterize the P2P links between peers. In addition to testing
for P2P compatibility, this test will perform a peer-to-peer throughput test
between all unique P2P pairs for performance evaluation. These are known as
device-to-device transfers, and can be either uni-directional or bi-directional.
The average bandwidth obtained is reported to help debug low bandwidth issues.

### Module specific keys

<div class="pst-scrollable-table-container">
<table class="table table--middle-left">
<tr><th class="head">Config Key</th> <th class="head">Type</th><th class="head"> Description</th></tr>
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
- true – Do a bidirectional transfer test\n
- false – Do a unidirectional transfer test
from one node to another.

</td></tr>
<tr><td>parallel</td><td>Bool</td>
<td>This option is only used if the test_bandwidth
key is true.\n
- true – Run all test transfers in parallel.\n
- false – Run test transfers one by one.

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
1000 (1 second). It must be smaller than the duration key.\n
if this key is 0 (zero), results are displayed as soon as the test transfer
is completed.</td></tr>
<tr><td>block_size</td><td>Collection of Integers</td>
<td>Optional. Defines list of block sizes to be used in transfer tests.\n
If "all" or missing list of block sizes used in rocm_bandwidth_test is used:
- 1 * 1024
- 2 * 1024
- 4 * 1024
- 8 * 1024
- 16 * 1024
- 32 * 1024
- 64 * 1024
- 128 * 1024
- 256 * 1024
- 512 * 1024
- 1 * 1024 * 1024
- 2 * 1024 * 1024
- 4 * 1024 * 1024
- 8 * 1024 * 1024
- 16 * 1024 * 1024
- 32 * 1024 * 1024
- 64 * 1024 * 1024
- 128 * 1024 * 1024
- 256 * 1024 * 1024
- 512 * 1024 * 1024
</td></tr>
<tr><td>b2b_block_size</td><td>Integer</td>
<td>This option is only used if both 'test_bandwidth' and 'parallel' keys are
true. This is a positive integer indicating size in Bytes of a data block to be
transferred continuously ("back-to-back") for the duration of one test pass. If
the key is not present, ordinary transfers with size indicated in 'block_size'
key will be performed.</td></tr>
<tr><td>link_type</td><td>Integer</td>
<td>This is a positive integer indicating type of link to be included in
bandwidth test. Numbering follows that listed in **hsa\_amd\_link\_info\_type\_t** in
**hsa\_ext\_amd.h** file.</td></tr>

<tr><td>b2b</td><td>Bool</td>
<td>If true, transfers are run back-to-back continuously for the full test
duration. Only applicable when using the native transfer method. If not
specified the default value is false.</td></tr>

<tr><td>hot_calls</td><td>Integer</td>
<td>Number of timed transfer iterations used to measure bandwidth. If not
specified the default value is 1.</td></tr>

<tr><td>warm_calls</td><td>Integer</td>
<td>Number of warm-up transfer iterations run before timing begins. Warm-up
iterations are not included in the bandwidth measurement. If not specified
the default value is 1.</td></tr>

<tr><td>transfer_method</td><td>String</td>
<td>Transfer backend to use. Accepted values:
<b>native</b> – Use the ROCm native SDMA transfer path (default).
<b>transferbench</b> – Use the TransferBench library as the transfer backend.
If not specified the default is <b>native</b>.</td></tr>

<tr><td>transferbench_test</td><td>String</td>
<td>TransferBench test type. Only applicable when
<b>transfer_method: transferbench</b>. Accepted values:
<b>p2p</b> – Point-to-point transfer test (default).
<b>alltoall</b> – All-to-all transfer test across all participating GPUs.
If not specified the default is <b>p2p</b>.</td></tr>

<tr><td>executor</td><td>String</td>
<td>Execution engine used by TransferBench. Only applicable when
<b>transfer_method: transferbench</b>. Accepted values:
<b>gfx</b> – Use GPU shader kernels (default).
<b>dma</b> – Use the DMA engine.
If not specified the default is <b>gfx</b>.</td></tr>

<tr><td>subexecutor</td><td>Integer</td>
<td>Number of sub-executors (wavefronts or threads) per transfer. Only
applicable when <b>transfer_method: transferbench</b>. If not specified the
default value is 1.</td></tr>

<tr><td>gfx_unroll</td><td>Integer</td>
<td>Unroll factor for the GFX shader kernel. Only applicable when
<b>transfer_method: transferbench</b> and <b>executor: gfx</b>. If not
specified the default value is 4.</td></tr>

<tr><td>use_remote_read</td><td>Integer</td>
<td>If set to 1, transfers use remote read instead of remote write. If not
specified the default value is 0 (remote write).</td></tr>

<tr><td>a2a_mode</td><td>Integer</td>
<td>All-to-all transfer pattern mode. Only applicable when
<b>transferbench_test: alltoall</b>. If not specified the default value is
0.</td></tr>

<tr><td>a2a_direct</td><td>Integer</td>
<td>If set to 1, enables direct peer-to-peer transfers in the all-to-all
test. Only applicable when <b>transferbench_test: alltoall</b>. If not
specified the default value is 1.</td></tr>

<tr><td>a2a_local</td><td>Integer</td>
<td>If set to 1, includes local (same-GPU) transfers in the all-to-all test.
Only applicable when <b>transferbench_test: alltoall</b>. If not specified
the default value is 0.</td></tr>

<tr><td>a2a_num_gpus</td><td>Integer</td>
<td>Number of GPUs to include in the all-to-all test. A value of 0 means all
detected GPUs are included. Only applicable when
<b>transferbench_test: alltoall</b>. If not specified the default value is
0.</td></tr>
</table>
</div>

Suitable values for **log\_interval** and **duration** depend
on your system.

- `log_interval`, in sequential mode, should be long enough to allow all
transfer tests to finish at least once or "(pending)" and "(*)" will be displayed
(see below). Number of transfers depends on number of peer NUMA nodes in your
system. In parallel mode, it should be roughly 1.5 times the duration of single
longest individual test.
- `duration`, regardless of mode should be at least, 4 * `log_interval`.

You may obtain indication of how long single transfer between two NUMA nodes
take by running test with "-d 4" switch and observing DEBUG messages for
transfer start/finish. An output may look like this:

    [DEBUG ] [183940.634118] [action_1] pbqt transfer 6 5 start
    [DEBUG ] [183941.311671] [action_1] pbqt transfer 6 5 finish
    [DEBUG ] [183941.312746] [action_1] pbqt transfer 4 5 start
    [DEBUG ] [183941.990174] [action_1] pbqt transfer 4 5 finish
    [DEBUG ] [183941.991244] [action_1] pbqt transfer 4 6 start
    [DEBUG ] [183942.668687] [action_1] pbqt transfer 4 6 finish
    [DEBUG ] [183942.669756] [action_1] pbqt transfer 5 4 start
    [DEBUG ] [183943.340957] [action_1] pbqt transfer 5 4 finish
    [DEBUG ] [183943.342037] [action_1] pbqt transfer 5 6 start
    [DEBUG ] [183944.17957 ] [action_1] pbqt transfer 5 6 finish
    [DEBUG ] [183944.19032 ] [action_1] pbqt transfer 6 4 start
    [DEBUG ] [183944.700868] [action_1] pbqt transfer 6 4 finish

From this printout, it can be concluded that single transfer takes on average
800ms. Values for `log_interval` and `duration` should be set accordingly.

### Output

Module specific output keys are described in the table below:

<div class="pst-scrollable-table-container">
<table class="table table--middle-left">
<tr><th class="head">Output Key</th> <th class="head">Type</th><th class="head"> Description</th></tr>
<tr><td>p2p_result</td><td>Bool</td>
<td>Indicates if the gpu and the specified peer have P2P capabilities. If this
quantity is true, the GPU pair tested has p2p capabilities. If false, they are
not peers.</td></tr>
<tr><td>distance</td><td>Integer</td>
<td>NUMA distance for these two peers</td></tr>
<tr><td>hop_type</td><td>String</td>
<td>Link type for each link hop (e.g., PCIe, HyperTransport, QPI, ...)</td></tr>
<tr><td>hop_distance</td><td>Integer</td>
<td>NUMA distance for this particular hop</td></tr>
<tr><td>transfer_id</td><td>String</td>
<td>String with format "<transfer_index>/<transfer_number>" where
- transfer_index - is number, starting from 1, for each device-peer combination
- transfer_number - is total number of device-peer combinations

</td></tr>

<tr><td>interval_bandwidth</td><td>Float</td>
<td>The average bandwidth of a p2p transfer, during the log_interval time
period.\n This field may also take values:
- (pending) - this means that no measurement has taken place
yet.
- xxxGBps (*) - this means no measurement within current log_interval but
average from previous measurements is displayed.

</td></tr>
<tr><td>bandwidth</td><td>Float</td>
<td>The average bandwidth of a p2p transfer, averaged over the entire test
duration of the interval. This field may also take value:
- (not measured) - this means no test transfer completed for those
peers. You may need to increase test duration.

</td></tr>
<tr><td>duration</td><td>Float</td>
<td>Cumulative duration of all transfers between the two particular nodes</td></tr>
</table>
</div>

If the value of test_bandwidth key is false, the tool will only try to determine
if the GPU(s) in the peers key are P2P to the action’s GPU. In this case the
bidirectional and log_interval values will be ignored, if they are specified. If
a gpu is a P2P peer to the device the test will pass, otherwise it will fail. A
message indicating the result will be provided for each GPUs specified. It will
have the following format:

    [RESULT][<timestamp>][<action name>] p2p <gpu id> <peer gpu id> peers:<p2p_result> distance:<distance> <hop_type>:<hop_dist>[ <hop_type>:<hop_dist>]

If the value of test_bandwidth is true bandwidth testing between the device and
each of its peers will take place in parallel or in sequence, depending on the
value of the parallel flag. During the duration of bandwidth benchmarking,
informational output providing the moving average of the transfer’s bandwidth
will be calculated and logged at every time increment specified by the
`log_interval` parameter. The messages will have the following output:

    [INFO  ][<timestamp>][<action name>] p2p-bandwidth [<transfer_id>] <gpu id> <peer gpu id> bidirectional: <bidirectional> <interval_bandwidth>

At the end of the test the average bytes/second will be calculated over the
entire test duration, and will be logged as a result:

    [RESULT][<timestamp>][<action name>] p2p-bandwidth [<transfer_id>] <gpu id> <peer gpu id> bidirectional: <bidirectional> <bandwidth> <duration>


### Examples


**Example 1:**

Here all source GPUs (device: all) with all destination GPUs (peers: all) are
tested for p2p capability with no bandwidth testing (test_bandwidth: false).

    actions:
    - name: action_1
      device: all
      module: pbqt
      peers: all
      test_bandwidth: false


Possible result is:

    [RESULT] [1656631.262875] [action_1] p2p 3254 3254 peers:false distance:-1
    [RESULT] [1656631.262968] [action_1] p2p 3254 50599 peers:true distance:56 HyperTransport:56
    [RESULT] [1656631.263039] [action_1] p2p 3254 33367 peers:true distance:56 HyperTransport:56
    [RESULT] [1656631.263103] [action_1] p2p 50599 3254 peers:true distance:56 HyperTransport:56
    [RESULT] [1656631.263151] [action_1] p2p 50599 50599 peers:false distance:-1
    [RESULT] [1656631.263203] [action_1] p2p 50599 33367 peers:true distance:56 HyperTransport:56
    [RESULT] [1656631.263265] [action_1] p2p 33367 3254 peers:true distance:56 HyperTransport:56
    [RESULT] [1656631.263321] [action_1] p2p 33367 50599 peers:true distance:56 HyperTransport:56
    [RESULT] [1656631.263360] [action_1] p2p 33367 33367 peers:false distance:-1

From the first line of result, we can see that GPU (ID 3254) can't access itself.
From the second line of result, we can see that source GPU (ID 3254) can access destination GPU (ID 50599).

**Example 2:**

Here all source GPUs (device: all) with all destination GPUs (peers: all) are
tested for p2p capability including bandwidth testing (test_bandwidth: true)
with bidirectional transfers (bidirectional: true) and with immediate output
for each completed transfer (log_interval: 0)

    actions:
    - name: action_1
      device: all
      module: pbqt
      log_interval: 0
      duration: 0
      peers: all
      test_bandwidth: true
      bidirectional: true

When run with "-d 3" switch, possible result is:

    [RESULT] [1657122.364752] [action_1] p2p 3254 3254 peers:false distance:-1
    [RESULT] [1657122.364845] [action_1] p2p 3254 50599 peers:true distance:56 HyperTransport:56
    [RESULT] [1657122.364917] [action_1] p2p 3254 33367 peers:true distance:56 HyperTransport:56
    [RESULT] [1657122.364985] [action_1] p2p 50599 3254 peers:true distance:56 HyperTransport:56
    [RESULT] [1657122.365037] [action_1] p2p 50599 50599 peers:false distance:-1
    [RESULT] [1657122.365094] [action_1] p2p 50599 33367 peers:true distance:56 HyperTransport:56
    [RESULT] [1657122.365157] [action_1] p2p 33367 3254 peers:true distance:56 HyperTransport:56
    [RESULT] [1657122.365221] [action_1] p2p 33367 50599 peers:true distance:56 HyperTransport:56
    [RESULT] [1657122.365270] [action_1] p2p 33367 33367 peers:false distance:-1
    [INFO  ] [1657123.644203] [action_1] p2p-bandwidth  [1/6] 3254 50599  bidirectional: true  7.013 GBps
    [INFO  ] [1657123.644376] [action_1] p2p-bandwidth  [2/6] 3254 33367  bidirectional: true  6.615 GBps
    [INFO  ] [1657123.644453] [action_1] p2p-bandwidth  [3/6] 50599 3254  bidirectional: true  2.367 GBps
    [INFO  ] [1657123.644522] [action_1] p2p-bandwidth  [4/6] 50599 33367  bidirectional: true  7.504 GBps
    [INFO  ] [1657123.644590] [action_1] p2p-bandwidth  [5/6] 33367 3254  bidirectional: true  8.207 GBps
    [INFO  ] [1657123.644673] [action_1] p2p-bandwidth  [6/6] 33367 50599  bidirectional: true  7.680 GBps
    [INFO  ] [1657124.926221] [action_1] p2p-bandwidth  [1/6] 3254 50599  bidirectional: true  6.646 GBps
    [INFO  ] [1657124.926368] [action_1] p2p-bandwidth  [2/6] 3254 33367  bidirectional: true  8.418 GBps
    [INFO  ] [1657124.926438] [action_1] p2p-bandwidth  [3/6] 50599 3254  bidirectional: true  7.402 GBps
    [INFO  ] [1657124.926506] [action_1] p2p-bandwidth  [4/6] 50599 33367  bidirectional: true  6.161 GBps
    [INFO  ] [1657124.926573] [action_1] p2p-bandwidth  [5/6] 33367 3254  bidirectional: true  9.024 GBps
    [INFO  ] [1657124.926640] [action_1] p2p-bandwidth  [6/6] 33367 50599  bidirectional: true  8.740 GBps
    [INFO  ] [1657126.208742] [action_1] p2p-bandwidth  [1/6] 3254 50599  bidirectional: true  5.680 GBps
    [INFO  ] [1657126.208905] [action_1] p2p-bandwidth  [2/6] 3254 33367  bidirectional: true  8.011 GBps
    [INFO  ] [1657126.208990] [action_1] p2p-bandwidth  [3/6] 50599 3254  bidirectional: true  3.918 GBps
    [INFO  ] [1657126.209066] [action_1] p2p-bandwidth  [4/6] 50599 33367  bidirectional: true  6.058 GBps
    [INFO  ] [1657126.209140] [action_1] p2p-bandwidth  [5/6] 33367 3254  bidirectional: true  6.650 GBps
    [INFO  ] [1657126.209213] [action_1] p2p-bandwidth  [6/6] 33367 50599  bidirectional: true  0.000 GBps
    [RESULT] [1657126.742128] [action_1] p2p-bandwidth  [1/6] 3254 50599  bidirectional: true  5.767 GBps  duration: 0.368453 sec
    [RESULT] [1657126.743287] [action_1] p2p-bandwidth  [2/6] 3254 33367  bidirectional: true  6.013 GBps  duration: 0.498944 sec
    [RESULT] [1657126.744411] [action_1] p2p-bandwidth  [3/6] 50599 3254  bidirectional: true  5.278 GBps  duration: 0.380393 sec
    [RESULT] [1657126.745534] [action_1] p2p-bandwidth  [4/6] 50599 33367  bidirectional: true  4.160 GBps  duration: 0.484577 sec
    [RESULT] [1657126.746684] [action_1] p2p-bandwidth  [5/6] 33367 3254  bidirectional: true  5.219 GBps  duration: 0.407190 sec
    [RESULT] [1657126.747827] [action_1] p2p-bandwidth  [6/6] 33367 50599  bidirectional: true  4.001 GBps  duration: 0.562350 sec

We can see that on this particular machine there are three GPUs and six
possible device-to-peer transfers.

**Example 3:**

Here some source GPUs (device: 50599) are targeting some destination GPUs
(peers: 33367 3254) with specified log interval (log_interval: 1000) and duration
(duration: 5000). Bandwidth is tested (test_bandwidth: true) but only
unidirectional (bidirectional: false) without parallel execution (parallel:
false).

    actions:
    - name: action_1
      device: 50599
      module: pbqt
      log_interval: 1000
      duration: 5000
      count: 0
      peers: 33367 3254
      test_bandwidth: true
      bidirectional: false
      parallel: false

Possible output is:

    [RESULT] [1657218.801555] [action_1] p2p 50599 3254 peers:true distance:56 HyperTransport:56
    [RESULT] [1657218.801655] [action_1] p2p 50599 33367 peers:true distance:56 HyperTransport:56
    [INFO  ] [1657219.871532] [action_1] p2p-bandwidth  [1/2] 50599 3254  bidirectional: false  4.517 GBps
    [INFO  ] [1657219.871717] [action_1] p2p-bandwidth  [2/2] 50599 33367  bidirectional: false  4.475 GBps
    [INFO  ] [1657220.940263] [action_1] p2p-bandwidth  [1/2] 50599 3254  bidirectional: false  4.476 GBps
    [INFO  ] [1657220.940461] [action_1] p2p-bandwidth  [2/2] 50599 33367  bidirectional: false  4.601 GBps
    [INFO  ] [1657222.7589  ] [action_1] p2p-bandwidth  [1/2] 50599 3254  bidirectional: false  4.488 GBps
    [INFO  ] [1657222.7760  ] [action_1] p2p-bandwidth  [2/2] 50599 33367  bidirectional: false  4.470 GBps
    [INFO  ] [1657223.74647 ] [action_1] p2p-bandwidth  [1/2] 50599 3254  bidirectional: false  4.666 GBps
    [INFO  ] [1657223.74810 ] [action_1] p2p-bandwidth  [2/2] 50599 33367  bidirectional: false  4.576 GBps
    [RESULT] [1657224.181106] [action_1] p2p-bandwidth  [1/2] 50599 3254  bidirectional: false  4.539 GBps  duration: 1.321909 sec
    [RESULT] [1657224.182255] [action_1] p2p-bandwidth  [2/2] 50599 33367  bidirectional: false  4.551 GBps  duration: 1.318517 sec

From the last line of result, we can see that source GPU (ID 50599) can access
destination GPU (ID 33367) and that the bandwidth is 4.495 GBps.

**Example 4:**

Here, all GPUs are targeted with bidirectional transfers and parallel execution
of tests:

    actions:
    - name: action_1
      device: all
      module: pbqt
      log_interval: 1200
      duration: 4000
      peers: all
      test_bandwidth: true
      bidirectional: true
      parallel: true

Possible output is:

    [RESULT] [1657295.937184] [action_1] p2p 3254 3254 peers:false distance:-1
    [RESULT] [1657295.937267] [action_1] p2p 3254 50599 peers:true distance:56 HyperTransport:56
    [RESULT] [1657295.937324] [action_1] p2p 3254 33367 peers:true distance:56 HyperTransport:56
    [RESULT] [1657295.937379] [action_1] p2p 50599 3254 peers:true distance:56 HyperTransport:56
    [RESULT] [1657295.937429] [action_1] p2p 50599 50599 peers:false distance:-1
    [RESULT] [1657295.937482] [action_1] p2p 50599 33367 peers:true distance:56 HyperTransport:56
    [RESULT] [1657295.937543] [action_1] p2p 33367 3254 peers:true distance:56 HyperTransport:56
    [RESULT] [1657295.937607] [action_1] p2p 33367 50599 peers:true distance:56 HyperTransport:56
    [RESULT] [1657295.937655] [action_1] p2p 33367 33367 peers:false distance:-1
    [INFO  ] [1657297.216212] [action_1] p2p-bandwidth  [1/6] 3254 50599  bidirectional: true  4.972 GBps
    [INFO  ] [1657297.216351] [action_1] p2p-bandwidth  [2/6] 3254 33367  bidirectional: true  8.183 GBps
    [INFO  ] [1657297.216423] [action_1] p2p-bandwidth  [3/6] 50599 3254  bidirectional: true  8.911 GBps
    [INFO  ] [1657297.216490] [action_1] p2p-bandwidth  [4/6] 50599 33367  bidirectional: true  7.690 GBps
    [INFO  ] [1657297.216558] [action_1] p2p-bandwidth  [5/6] 33367 3254  bidirectional: true  7.768 GBps
    [INFO  ] [1657297.216642] [action_1] p2p-bandwidth  [6/6] 33367 50599  bidirectional: true  4.589 GBps
    [INFO  ] [1657298.487427] [action_1] p2p-bandwidth  [1/6] 3254 50599  bidirectional: true  8.778 GBps
    [INFO  ] [1657298.487593] [action_1] p2p-bandwidth  [2/6] 3254 33367  bidirectional: true  7.921 GBps
    [INFO  ] [1657298.487730] [action_1] p2p-bandwidth  [3/6] 50599 3254  bidirectional: true  8.164 GBps
    [INFO  ] [1657298.487807] [action_1] p2p-bandwidth  [4/6] 50599 33367  bidirectional: true  8.921 GBps
    [INFO  ] [1657298.487878] [action_1] p2p-bandwidth  [5/6] 33367 3254  bidirectional: true  8.487 GBps
    [INFO  ] [1657298.487956] [action_1] p2p-bandwidth  [6/6] 33367 50599  bidirectional: true  7.648 GBps
    [INFO  ] [1657299.760175] [action_1] p2p-bandwidth  [1/6] 3254 50599  bidirectional: true  7.210 GBps
    [INFO  ] [1657299.760249] [action_1] p2p-bandwidth  [2/6] 3254 33367  bidirectional: true  4.274 GBps
    [INFO  ] [1657299.760284] [action_1] p2p-bandwidth  [3/6] 50599 3254  bidirectional: true  0.000 GBps
    [INFO  ] [1657299.760318] [action_1] p2p-bandwidth  [4/6] 50599 33367  bidirectional: true  5.942 GBps
    [INFO  ] [1657299.760349] [action_1] p2p-bandwidth  [5/6] 33367 3254  bidirectional: true  0.001 GBps
    [INFO  ] [1657299.760381] [action_1] p2p-bandwidth  [6/6] 33367 50599  bidirectional: true  5.490 GBps
    [RESULT] [1657300.293126] [action_1] p2p-bandwidth  [1/6] 3254 50599  bidirectional: true  6.964 GBps  duration: 0.287248 sec
    [RESULT] [1657300.294334] [action_1] p2p-bandwidth  [2/6] 3254 33367  bidirectional: true  3.960 GBps  duration: 0.536554 sec
    [RESULT] [1657300.295528] [action_1] p2p-bandwidth  [3/6] 50599 3254  bidirectional: true  5.442 GBps  duration: 0.368977 sec
    [RESULT] [1657300.296691] [action_1] p2p-bandwidth  [4/6] 50599 33367  bidirectional: true  4.187 GBps  duration: 0.477756 sec
    [RESULT] [1657300.297840] [action_1] p2p-bandwidth  [5/6] 33367 3254  bidirectional: true  4.942 GBps  duration: 0.607009 sec
    [RESULT] [1657300.299016] [action_1] p2p-bandwidth  [6/6] 33367 50599  bidirectional: true  3.828 GBps  duration: 0.523495 sec

It can be seen that transfers [2/6] and [5/6] did not take place in the second
log interval so average from the previous cycle is displayed instead and
marked with "(*)"

## PEBB module
The PCIe Bandwidth Benchmark attempts to saturate the PCIe bus with DMA
transfers between system memory and a target GPU card’s memory. These are known
as host-to-device or device- to-host transfers, and can be either unidirectional
or bidirectional transfers. The maximum bandwidth obtained is reported.

### Module specific keys

<div class="pst-scrollable-table-container">
<table class="table table--middle-left">
<tr><th class="head">Config Key</th> <th class="head">Type</th><th class="head"> Description</th></tr>
<tr><td>host_to_device</td><td>Bool</td>
<td>This key indicates if host to device transfers
will be considered. The default value is true.</td></tr>
<tr><td>device_to_host</td><td>Bool</td>
<td>This key indicates if device to host transfers
will be considered. The default value is true.
</td></tr>
<tr><td>parallel</td><td>Bool</td>
<td>This option is only used if the test_bandwidth
key is true.\n
- true – Run all test transfers in parallel.\n
- false – Run test transfers one by one.

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
1000 (1 second). It must be smaller than the duration key.\n
if this key is 0 (zero), results are displayed as soon as the test transfer
is completed.</td></tr>
<tr><td>block_size</td><td>Collection of Integers</td>
<td>Optional. Defines list of block sizes to be used in transfer tests.\n
If "all" or missing list of block sizes used in rocm_bandwidth_test is used:
- 1 * 1024
- 2 * 1024
- 4 * 1024
- 8 * 1024
- 16 * 1024
- 32 * 1024
- 64 * 1024
- 128 * 1024
- 256 * 1024
- 512 * 1024
- 1 * 1024 * 1024
- 2 * 1024 * 1024
- 4 * 1024 * 1024
- 8 * 1024 * 1024
- 16 * 1024 * 1024
- 32 * 1024 * 1024
- 64 * 1024 * 1024
- 128 * 1024 * 1024
- 256 * 1024 * 1024
- 512 * 1024 * 1024
</td></tr>
<tr><td>b2b_block_size</td><td>Integer</td>
<td>This option is only used if both 'test_bandwidth' and 'parallel' keys are
true. This is a positive integer indicating size in Bytes of a data block to be
transferred continuously ("back-to-back") for the duration of one test pass. If
the key is not present, ordinary transfers with size indicated in 'block_size'
key will be performed.</td></tr>
<tr><td>link_type</td><td>Integer</td>
<td>This is a positive integer indicating type of link to be included in
bandwidth test. Numbering follows that listed in **hsa\_amd\_link\_info\_type\_t** in
**hsa\_ext\_amd.h** file.</td></tr>

<tr><td>b2b</td><td>Bool</td>
<td>If true, transfers are run back-to-back continuously for the full test
duration. Only applicable when using the native transfer method. If not
specified the default value is false.</td></tr>

<tr><td>hot_calls</td><td>Integer</td>
<td>Number of timed transfer iterations used to measure bandwidth. If not
specified the default value is 1.</td></tr>

<tr><td>warm_calls</td><td>Integer</td>
<td>Number of warm-up transfer iterations run before timing begins. Warm-up
iterations are not included in the bandwidth measurement. If not specified
the default value is 1.</td></tr>

<tr><td>transfer_method</td><td>String</td>
<td>Transfer backend to use. Accepted values:
<b>native</b> – Use the ROCm native SDMA transfer path (default).
<b>transferbench</b> – Use the TransferBench library as the transfer backend.
If not specified the default is <b>native</b>.</td></tr>

<tr><td>executor</td><td>String</td>
<td>Execution engine used by TransferBench. Only applicable when
<b>transfer_method: transferbench</b>. Accepted values:
<b>gfx</b> – Use GPU shader kernels (default).
<b>dma</b> – Use the DMA engine.
If not specified the default is <b>gfx</b>.</td></tr>

<tr><td>subexecutor</td><td>Integer</td>
<td>Number of sub-executors (wavefronts or threads) per transfer. Only
applicable when <b>transfer_method: transferbench</b>. If not specified the
default value is 1.</td></tr>

<tr><td>gfx_unroll</td><td>Integer</td>
<td>Unroll factor for the GFX shader kernel. Only applicable when
<b>transfer_method: transferbench</b> and <b>executor: gfx</b>. If not
specified the default value is 4.</td></tr>

<tr><td>source_memory</td><td>String</td>
<td>Memory type for the transfer source. Only applicable when
<b>transfer_method: transferbench</b>. Accepted values:
<b>cpu</b> – System (host) memory.
<b>gpu</b> – GPU device memory.
<b>null</b> – No explicit memory allocation; let TransferBench decide (default).
If not specified the default is <b>null</b>.</td></tr>

<tr><td>destination_memory</td><td>String</td>
<td>Memory type for the transfer destination. Only applicable when
<b>transfer_method: transferbench</b>. Accepted values:
<b>cpu</b> – System (host) memory.
<b>gpu</b> – GPU device memory.
<b>null</b> – No explicit memory allocation; let TransferBench decide (default).
If not specified the default is <b>null</b>.</td></tr>
</table>
</div>

Suitable values for `log_interval` and `duration` depend
on your system.

- `log_interval`, in sequential mode, should be long enough to allow all
transfer tests to finish at least once or "(pending)" and "(*)" will be displayed
(see below). Number of transfers depends on number of peer NUMA nodes in your
system. In parallel mode, it should be roughly 1.5 times the duration of single
longest individual test.
- `duration`, regardless of mode should be at least, 4 * `log_interval`.

You may obtain indication of how long single transfer between two NUMA nodes
take by running test with "-d 4" switch and observing DEBUG messages for
transfer start/finish. An output may look like this:

    [DEBUG ] [187024.729433] [action_1] pebb transfer 0 6 start
    [DEBUG ] [187029.327818] [action_1] pebb transfer 0 6 finish
    [DEBUG ] [187024.299150] [action_1] pebb transfer 1 6 start
    [DEBUG ] [187029.473378] [action_1] pebb transfer 1 6 finish
    [DEBUG ] [187023.227009] [action_1] pebb transfer 1 5 start
    [DEBUG ] [187029.530203] [action_1] pebb transfer 1 5 finish
    [DEBUG ] [187025.737675] [action_1] pebb transfer 3 5 start
    [DEBUG ] [187030.134100] [action_1] pebb transfer 3 5 finish
    [DEBUG ] [187027.19961 ] [action_1] pebb transfer 2 6 start
    [DEBUG ] [187030.421181] [action_1] pebb transfer 2 6 finish
    [DEBUG ] [187027.41475 ] [action_1] pebb transfer 2 5 start
    [DEBUG ] [187031.293998] [action_1] pebb transfer 2 5 finish
    [DEBUG ] [187027.71717 ] [action_1] pebb transfer 0 5 start
    [DEBUG ] [187031.605326] [action_1] pebb transfer 0 5 finish

From this printout, it can be concluded that single transfer takes on average
5500ms. Values for `log_interval` and `duration` should be set accordingly.


### Output

Module specific output keys are described in the table below:

<div class="pst-scrollable-table-container">
<table class="table table--middle-left">
<tr><th class="head">Output Key</th> <th class="head">Type</th><th class="head"> Description</th></tr>
<tr><td>CPU node</td><td>Integer</td>
<td>Particular CPU node involved in transfer</td></tr>
<tr><td>distance</td><td>Integer</td>
<td>NUMA distance for these two peers</td></tr>
<tr><td>hop_type</td><td>String</td>
<td>Link type for each link hop (e.g., PCIe, HyperTransport, QPI, ...)</td></tr>
<tr><td>hop_distance</td><td>Integer</td>
<td>NUMA distance for this particular hop</td></tr>
<tr><td>transfer_id</td><td>String</td>
<td>String with format "<transfer_index>/<transfer_number>" where
- transfer_index - is number, starting from 1, for each device-peer combination
- transfer_number - is total number of device-peer combinations

</td></tr>

<tr><td>interval_bandwidth</td><td>Float</td>
<td>The average bandwidth of a CPU-to-GPU or GPU-to-CPU transfer, during the
log_interval time period.\n This field may also take values:
- (pending) - this means that no measurement has taken place
yet.
- xxxGBps (*) - this means no measurement within current log_interval but
average from previous measurements is displayed.

</td></tr>
<tr><td>bandwidth</td><td>Float</td>
<td>The average bandwidth of a CPU-to-GPU or GPU-to-CPU transfer, averaged
over the entire test duration. This field may also take value:
- (not measured) - this means no test transfer completed for those
peers. You may need to increase test duration.

</td></tr>
<tr><td>duration</td><td>Float</td>
<td>Cumulative duration of all transfers between the two particular nodes</td></tr>
</table>
</div>

At the beginning, the test will display link info for every CPU/GPU pair:

    [RESULT][<timestamp>][<action name>] pcie-bandwidth [<transfer_id>] <cpu node> <gpu node> <gpu id> distance:<distance> <hop_type>:<hop_dist>[ <hop_type>:<hop_dist>]

During the execution of the benchmark, informational output providing the moving
average of the bandwidth of the transfer will be calculated and logged. This
interval is provided by the log_interval parameter and will have the following
output format:

    [INFO ][<timestamp>][<action name>] pcie-bandwidth [<transfer_id>] <cpu node> <gpu id> h2d: <host_to_device> d2h: <device_to_host> <interval_bandwidth>

At the end of test, the average bytes/second will be calculated over the
entire test duration, and will be logged as a result:

    [RESULT][<timestamp>][<action name>] pcie-bandwidth [<transfer_id>] <cpu node> <gpu id> h2d: <host_to_device> d2h: <device_to_host> <bandwidth> <duration>



### Examples

**Example 1:**

Consider action:

    actions:
    - name: action_1
      device: all
      module: pebb
      log_interval: 0
      duration: 0
      device_to_host: false
      host_to_device: true
      parallel: false

This will initiate host to device transfer to all GPUs with immediate output
(`parallel: false`, `log_interval: 0`)

Output from this action might look like:

    [RESULT] [1658774.978614] [action_1] pcie-bandwidth 0 4 3254  distance:36 HyperTransport:36
    [RESULT] [1658774.978664] [action_1] pcie-bandwidth 1 4 3254  distance:20 PCIe:20
    [RESULT] [1658774.978695] [action_1] pcie-bandwidth 2 4 3254  distance:36 HyperTransport:36
    [RESULT] [1658774.978728] [action_1] pcie-bandwidth 3 4 3254  distance:36 HyperTransport:36
    [RESULT] [1658774.978763] [action_1] pcie-bandwidth 0 5 50599  distance:36 HyperTransport:36
    [RESULT] [1658774.978795] [action_1] pcie-bandwidth 1 5 50599  distance:36 HyperTransport:36
    [RESULT] [1658774.978825] [action_1] pcie-bandwidth 2 5 50599  distance:20 PCIe:20
    [RESULT] [1658774.978856] [action_1] pcie-bandwidth 3 5 50599  distance:36 HyperTransport:36
    [RESULT] [1658774.978889] [action_1] pcie-bandwidth 0 6 33367  distance:36 HyperTransport:36
    [RESULT] [1658774.978922] [action_1] pcie-bandwidth 1 6 33367  distance:36 HyperTransport:36
    [RESULT] [1658774.978952] [action_1] pcie-bandwidth 2 6 33367  distance:36 HyperTransport:36
    [RESULT] [1658774.978982] [action_1] pcie-bandwidth 3 6 33367  distance:20 PCIe:20
    [INFO  ] [1658774.983743] [action_1] pcie-bandwidth  [1/12] 0 3254  h2d: true  d2h: false  12.233 GBps
    [INFO  ] [1658774.988272] [action_1] pcie-bandwidth  [2/12] 1 3254  h2d: true  d2h: false  12.227 GBps
    [INFO  ] [1658774.993197] [action_1] pcie-bandwidth  [3/12] 2 3254  h2d: true  d2h: false  11.770 GBps
    [INFO  ] [1658774.998105] [action_1] pcie-bandwidth  [4/12] 3 3254  h2d: true  d2h: false  11.313 GBps
    [INFO  ] [1658775.4457  ] [action_1] pcie-bandwidth  [5/12] 0 50599  h2d: true  d2h: false  12.218 GBps
    [INFO  ] [1658775.9589  ] [action_1] pcie-bandwidth  [6/12] 1 50599  h2d: true  d2h: false  10.292 GBps
    [INFO  ] [1658775.14627 ] [action_1] pcie-bandwidth  [7/12] 2 50599  h2d: true  d2h: false  10.456 GBps
    [INFO  ] [1658775.19664 ] [action_1] pcie-bandwidth  [8/12] 3 50599  h2d: true  d2h: false  10.614 GBps
    [INFO  ] [1658775.26210 ] [action_1] pcie-bandwidth  [9/12] 0 33367  h2d: true  d2h: false  12.222 GBps
    [INFO  ] [1658775.31188 ] [action_1] pcie-bandwidth  [10/12] 1 33367  h2d: true  d2h: false  12.215 GBps
    [INFO  ] [1658775.36137 ] [action_1] pcie-bandwidth  [11/12] 2 33367  h2d: true  d2h: false  12.219 GBps
    [INFO  ] [1658775.41117 ] [action_1] pcie-bandwidth  [12/12] 3 33367  h2d: true  d2h: false  12.219 GBps
    [RESULT] [1658775.42219 ] [action_1] pcie-bandwidth  [1/12] 0 3254  h2d: true  d2h: false  12.233 GBps  duration: 0.000780 sec
    [RESULT] [1658775.42235 ] [action_1] pcie-bandwidth  [2/12] 1 3254  h2d: true  d2h: false  12.227 GBps  duration: 0.000780 sec
    [RESULT] [1658775.42246 ] [action_1] pcie-bandwidth  [3/12] 2 3254  h2d: true  d2h: false  11.770 GBps  duration: 0.000810 sec
    [RESULT] [1658775.42256 ] [action_1] pcie-bandwidth  [4/12] 3 3254  h2d: true  d2h: false  11.313 GBps  duration: 0.000843 sec
    [RESULT] [1658775.42271 ] [action_1] pcie-bandwidth  [5/12] 0 50599  h2d: true  d2h: false  12.218 GBps  duration: 0.000781 sec
    [RESULT] [1658775.42286 ] [action_1] pcie-bandwidth  [6/12] 1 50599  h2d: true  d2h: false  10.292 GBps  duration: 0.000927 sec
    [RESULT] [1658775.42297 ] [action_1] pcie-bandwidth  [7/12] 2 50599  h2d: true  d2h: false  10.456 GBps  duration: 0.000912 sec
    [RESULT] [1658775.42309 ] [action_1] pcie-bandwidth  [8/12] 3 50599  h2d: true  d2h: false  10.614 GBps  duration: 0.000898 sec
    [RESULT] [1658775.42321 ] [action_1] pcie-bandwidth  [9/12] 0 33367  h2d: true  d2h: false  12.222 GBps  duration: 0.000780 sec
    [RESULT] [1658775.42332 ] [action_1] pcie-bandwidth  [10/12] 1 33367  h2d: true  d2h: false  12.215 GBps  duration: 0.000781 sec
    [RESULT] [1658775.42344 ] [action_1] pcie-bandwidth  [11/12] 2 33367  h2d: true  d2h: false  12.219 GBps  duration: 0.000780 sec
    [RESULT] [1658775.42355 ] [action_1] pcie-bandwidth  [12/12] 3 33367  h2d: true  d2h: false  12.219 GBps  duration: 0.000780 sec

**Example 2:**

Consider action:

    actions:
    - name: action_1
      device: all
      module: pebb
      log_interval: 500
      duration: 5000
      device_to_host: true
      host_to_device: true
      parallel: true

Here, although parallel execution of transfers is requested, log_interval is to
short for some transfers to complete. For them, cumulative average is displayed
and marked with (*):

    [RESULT] [1659672.517170] [action_1] pcie-bandwidth 0 4 3254  distance:36 HyperTransport:36
    [RESULT] [1659672.517222] [action_1] pcie-bandwidth 1 4 3254  distance:20 PCIe:20
    [RESULT] [1659672.517257] [action_1] pcie-bandwidth 2 4 3254  distance:36 HyperTransport:36
    [RESULT] [1659672.517290] [action_1] pcie-bandwidth 3 4 3254  distance:36 HyperTransport:36
    [RESULT] [1659672.517324] [action_1] pcie-bandwidth 0 5 50599  distance:36 HyperTransport:36
    [RESULT] [1659672.517357] [action_1] pcie-bandwidth 1 5 50599  distance:36 HyperTransport:36
    [RESULT] [1659672.517388] [action_1] pcie-bandwidth 2 5 50599  distance:20 PCIe:20
    [RESULT] [1659672.517419] [action_1] pcie-bandwidth 3 5 50599  distance:36 HyperTransport:36
    [RESULT] [1659672.517452] [action_1] pcie-bandwidth 0 6 33367  distance:36 HyperTransport:36
    [RESULT] [1659672.517483] [action_1] pcie-bandwidth 1 6 33367  distance:36 HyperTransport:36
    [RESULT] [1659672.517515] [action_1] pcie-bandwidth 2 6 33367  distance:36 HyperTransport:36
    [RESULT] [1659672.517546] [action_1] pcie-bandwidth 3 6 33367  distance:20 PCIe:20
    [INFO  ] [1659673.49782 ] [action_1] pcie-bandwidth  [1/12] 0 3254  h2d: true  d2h: true  1.489 GBps
    [INFO  ] [1659673.49814 ] [action_1] pcie-bandwidth  [2/12] 1 3254  h2d: true  d2h: true  2.701 GBps
    ...
    [INFO  ] [1659673.582639] [action_1] pcie-bandwidth  [1/12] 0 3254  h2d: true  d2h: true  1.489 GBps (*)
    [INFO  ] [1659673.582686] [action_1] pcie-bandwidth  [2/12] 1 3254  h2d: true  d2h: true  16.367 GBps
    [INFO  ] [1659673.582700] [action_1] pcie-bandwidth  [3/12] 2 3254  h2d: true  d2h: true  17.300 GBps
    ...
    [INFO  ] [1659677.851697] [action_1] pcie-bandwidth  [1/12] 0 3254  h2d: true  d2h: true  16.793 GBps
    [INFO  ] [1659677.851727] [action_1] pcie-bandwidth  [2/12] 1 3254  h2d: true  d2h: true  16.872 GBps (*)
    [INFO  ] [1659677.851741] [action_1] pcie-bandwidth  [3/12] 2 3254  h2d: true  d2h: true  14.796 GBps (*)
    [INFO  ] [1659677.851754] [action_1] pcie-bandwidth  [4/12] 3 3254  h2d: true  d2h: true  20.358 GBps
    [INFO  ] [1659677.851770] [action_1] pcie-bandwidth  [5/12] 0 50599  h2d: true  d2h: true  15.632 GBps (*)
    [INFO  ] [1659677.851828] [action_1] pcie-bandwidth  [6/12] 1 50599  h2d: true  d2h: true  14.541 GBps (*)
    ...
    [RESULT] [1659678.148280] [action_1] pcie-bandwidth  [1/12] 0 3254  h2d: true  d2h: true  16.309 GBps  duration: 0.061316 sec
    [RESULT] [1659678.148318] [action_1] pcie-bandwidth  [2/12] 1 3254  h2d: true  d2h: true  16.871 GBps  duration: 0.118547 sec
    [RESULT] [1659678.148332] [action_1] pcie-bandwidth  [3/12] 2 3254  h2d: true  d2h: true  13.360 GBps  duration: 0.149705 sec
    [RESULT] [1659678.148349] [action_1] pcie-bandwidth  [4/12] 3 3254  h2d: true  d2h: true  15.371 GBps  duration: 0.130115 sec
    [RESULT] [1659678.148363] [action_1] pcie-bandwidth  [5/12] 0 50599  h2d: true  d2h: true  15.631 GBps  duration: 0.127954 sec
    [RESULT] [1659678.148377] [action_1] pcie-bandwidth  [6/12] 1 50599  h2d: true  d2h: true  14.185 GBps  duration: 0.140989 sec
    [RESULT] [1659678.148390] [action_1] pcie-bandwidth  [7/12] 2 50599  h2d: true  d2h: true  15.242 GBps  duration: 0.131245 sec
    [RESULT] [1659678.148404] [action_1] pcie-bandwidth  [8/12] 3 50599  h2d: true  d2h: true  16.071 GBps  duration: 0.124452 sec
    [RESULT] [1659678.148418] [action_1] pcie-bandwidth  [9/12] 0 33367  h2d: true  d2h: true  16.505 GBps  duration: 0.121178 sec
    [RESULT] [1659678.148432] [action_1] pcie-bandwidth  [10/12] 1 33367  h2d: true  d2h: true  16.720 GBps  duration: 0.059807 sec
    [RESULT] [1659678.148445] [action_1] pcie-bandwidth  [11/12] 2 33367  h2d: true  d2h: true  15.604 GBps  duration: 0.128168 sec
    [RESULT] [1659678.148458] [action_1] pcie-bandwidth  [12/12] 3 33367  h2d: true  d2h: true  16.193 GBps  duration: 0.123525 sec

Please note that in link information results, some records could be marked with
(R). This means, that communication is possible if initiated by the destination
NUMA node HSA agent.

## GST module

The GPU Stress Test drives and measures the specified GPU(s) performance (GFLOPS) -
by means of large matrix multiplications using GEMM operation types based computations like
SGEMM/DGEMM/HGEMM (Single/Double-precision/Half-precision General Matrix Multiplication)
or GEMM data types based computations like `fp8`, `i8`, `fp16`, `bf16`, `fp32` or `tf32` (`xf32`) via BLAS
libraries like rocBLAS or hipBLASLt. The GPU stress module may be configured so it does not
copy the host source matrix array to the GPU before every matrix multiplication. This allows
the GPU performance to not be capped by device to host bandwidth transfers. The module calculates
the GFLOPS performance for configured GEMM computation and checks if it meets configured
performance target. The test passes if it achieves the target performance GFLOPS number
during the duration of the test else reported as fail.

This module should be used in conjunction with the GPU Monitor, to watch for
thermal, power and related anomalies while the target GPU(s) are under realistic
load conditions. By setting the appropriate parameters a user can ensure that
all GPUs in a node or cluster reach desired performance levels. Further analysis
of the generated stats can also show variations in the required power, clocks or
temperatures to reach these targets, and thus highlight GPUs or nodes that are
operating less efficiently.

### Module specific keys

Module specific keys are described in the table below:

<div class="pst-scrollable-table-container">
<table class="table table--middle-left">
<tr><th class="head">Config Key</th> <th class="head">Type</th><th class="head"> Description</th></tr>
<tr><td>target_stress</td><td>Float</td>
<td>The maximum relative performance the GPU will attempt to achieve in
gigaflops. This parameter is required.</td></tr>
<tr><td>copy_matrix</td><td>Bool</td>
<td>This parameter indicates if each operation should copy the matrix data to
the GPU before executing. The default value is true.</td></tr>
<tr><td>ramp_interval</td><td>Integer</td>
<td>This is a time interval, specified in milliseconds, given to the test to
reach the given target_stress gigaflops. The default value is 5000 (5 seconds).
This time is counted against the duration of the test. If the target gflops, or
stress, is not achieved in this time frame, the test will fail. If the target
stress (gflops) is achieved the test will attempt to run for the rest of the
duration specified by the action, sustaining the stress load during that
time.</td></tr>
<tr><td>tolerance</td><td>Float</td>
<td>A value indicating how much the target_stress can fluctuate after the ramp
period for the test to succeed. The default value is 0.05 (5%).</td></tr>
<tr><td>max_violations</td><td>Integer</td>
<td>The number of tolerance violations that can occur after the ramp_interval
for the test to still pass. The default value is 0.</td></tr>
<tr><td>log_interval</td><td>Integer</td>
<td>This is a positive integer, given in milliseconds, that specifies an
interval over which the moving average of the bandwidth will be calculated and
logged.</td></tr>
<tr><td>matrix_size</td><td>Integer</td>
<td>Sets all three matrix dimensions (M, N, and K) to the same value. Equivalent
to setting <b>matrix_size_a</b>, <b>matrix_size_b</b>, and <b>matrix_size_c</b>
to the same value. The default value is 5760.</td></tr>
<tr><td>matrix_size_a</td><td>Integer</td>
<td>Number of rows of matrix A (the M dimension in GEMM). Overrides
<b>matrix_size</b> for this dimension. The default value is 5760.</td></tr>
<tr><td>matrix_size_b</td><td>Integer</td>
<td>Number of columns of matrix B (the N dimension in GEMM). Overrides
<b>matrix_size</b> for this dimension. The default value is 5760.</td></tr>
<tr><td>matrix_size_c</td><td>Integer</td>
<td>Inner (shared) dimension K of the GEMM operation (columns of A / rows of B).
Overrides <b>matrix_size</b> for this dimension. The default value is
5760.</td></tr>

<tr><td>ops_type</td><td>String</td>
<td>GEMM operation type. Accepted values: <b>sgemm</b>, <b>dgemm</b>,
<b>hgemm</b>. If neither <b>ops_type</b> nor <b>data_type</b> is set, the
module defaults to <b>sgemm</b>. Mutually exclusive with <b>data_type</b>.</td></tr>

<tr><td>data_type</td><td>String</td>
<td>Data type for the GEMM computation. Accepted values include:
<b>fp4_r</b>, <b>fp6_r</b>, <b>fp8_r</b>, <b>bf16_r</b>, <b>fp16_r</b>,
<b>fp32_r</b>, <b>tf32_r</b> (<b>xf32_r</b>), <b>i8_r</b>. If not specified
the module falls back to the type implied by <b>ops_type</b>.</td></tr>

<tr><td>out_data_type</td><td>String</td>
<td>Output (result) data type, e.g. <b>fp16_r</b>, <b>fp32_r</b>. Only
applicable when using hipBLASLt. If not specified the default matches the
compute type.</td></tr>

<tr><td>compute_type</td><td>String</td>
<td>Accumulation/compute type used internally by the BLAS library. If not
specified the default is <b>fp32_r</b>.</td></tr>

<tr><td>blas_source</td><td>String</td>
<td>BLAS library backend to use. Accepted values:
<b>rocblas</b> (default), <b>hipblaslt</b>.</td></tr>

<tr><td>hot_calls</td><td>Integer</td>
<td>Number of GEMM kernel invocations per measurement window used to amortise
launch overhead. The default value is 1.</td></tr>

<tr><td>matrix_init</td><td>String</td>
<td>Matrix initialization method. Accepted values:
<b>default</b> – Initialize with default pattern (default).
<b>trig</b> – Initialize with trigonometric (sine/cosine) values.
<b>random</b> – Initialize with random values.</td></tr>

<tr><td>transa</td><td>Integer</td>
<td>Transpose operation applied to matrix A before the GEMM call.
0 = no transpose (default), 1 = transpose.</td></tr>

<tr><td>transb</td><td>Integer</td>
<td>Transpose operation applied to matrix B before the GEMM call.
0 = no transpose, 1 = transpose (default).</td></tr>

<tr><td>alpha</td><td>Float</td>
<td>Scalar multiplier applied to the product of matrices A and B in the GEMM
operation (C = alpha * A * B + beta * C). The default value is 1.</td></tr>

<tr><td>beta</td><td>Float</td>
<td>Scalar multiplier applied to matrix C in the GEMM operation. The default
value is 1.</td></tr>

<tr><td>lda</td><td>Integer</td>
<td>Leading dimension offset added to the computed leading dimension of matrix
A. The default value is 0.</td></tr>

<tr><td>ldb</td><td>Integer</td>
<td>Leading dimension offset added to the computed leading dimension of matrix
B. The default value is 0.</td></tr>

<tr><td>ldc</td><td>Integer</td>
<td>Leading dimension offset added to the computed leading dimension of matrix
C. The default value is 0.</td></tr>

<tr><td>ldd</td><td>Integer</td>
<td>Leading dimension offset added to the computed leading dimension of matrix
D (output). The default value is 0.</td></tr>

<tr><td>scale_a</td><td>String</td>
<td>Scaling mode applied to matrix A. Used with low-precision types such as
fp8. Accepted values include <b>block</b>. If not specified no scaling is
applied.</td></tr>

<tr><td>scale_b</td><td>String</td>
<td>Scaling mode applied to matrix B. Used with low-precision types such as
fp8. Accepted values include <b>block</b>. If not specified no scaling is
applied.</td></tr>

<tr><td>rotating</td><td>Integer</td>
<td>Size of the rotating buffer (in elements) used to prevent data from
residing in cache between iterations, enabling cache-cold benchmarking. A
value of 0 disables rotating buffers. The default value is 0.</td></tr>

<tr><td>gemm_mode</td><td>String</td>
<td>GEMM execution mode. Accepted values:
<b>""</b> or unset – Standard (single) GEMM (default).
<b>batched</b> – Batched GEMM; use with <b>batch_size</b>.
<b>strided_batched</b> – Strided batched GEMM; use with <b>batch_size</b> and
<b>stride_*</b> keys.</td></tr>

<tr><td>batch_size</td><td>Integer</td>
<td>Number of GEMM operations in a batched or strided-batched call. Only
applicable when <b>gemm_mode</b> is <b>batched</b> or
<b>strided_batched</b>. The default value is 0.</td></tr>

<tr><td>stride_a</td><td>Integer</td>
<td>Stride (in elements) between consecutive matrices A in a strided-batched
GEMM. Only applicable when <b>gemm_mode: strided_batched</b>. The default
value is 0.</td></tr>

<tr><td>stride_b</td><td>Integer</td>
<td>Stride (in elements) between consecutive matrices B in a strided-batched
GEMM. The default value is 0.</td></tr>

<tr><td>stride_c</td><td>Integer</td>
<td>Stride (in elements) between consecutive matrices C in a strided-batched
GEMM. The default value is 0.</td></tr>

<tr><td>stride_d</td><td>Integer</td>
<td>Stride (in elements) between consecutive output matrices D in a
strided-batched GEMM. The default value is 0.</td></tr>

<tr><td>self_check</td><td>Bool</td>
<td>If true, validates the GEMM result for correctness after each operation.
Adds overhead; intended for debugging. The default value is false.</td></tr>

<tr><td>accuracy_check</td><td>Bool</td>
<td>If true, runs a numerical accuracy check after each GEMM operation. The
default value is false.</td></tr>

<tr><td>error_inject</td><td>Bool</td>
<td>If true, enables error injection mode to deliberately introduce errors
into the computation for testing error detection. The default value is
false.</td></tr>

<tr><td>error_freq</td><td>Integer</td>
<td>Frequency of error injection; specifies how often (every N operations) an
error is injected. Only applicable when <b>error_inject: true</b>. The
default value is 0.</td></tr>

<tr><td>error_count</td><td>Integer</td>
<td>Number of errors to inject per injection event. Only applicable when
<b>error_inject: true</b>. The default value is 0.</td></tr>
</table>
</div>

### Output

Module specific output keys are described in the table below:

<div class="pst-scrollable-table-container">
<table class="table table--middle-left">
<tr><th class="head">Output Key</th> <th class="head">Type</th><th class="head"> Description</th></tr>
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
<td>Number of bytes copied to the GPU per GEMM operation when
<b>copy_matrix</b> is true.</td></tr>
<tr><td>try_ops_per_sec</td><td>Float</td>
<td>Calculated number of ops/second necessary to achieve target
gigaflops.</td></tr>
<tr><td>pass</td><td>Bool</td>
<td>'true' if the GPU achieves its desired sustained performance
level.</td></tr>
</table>
</div>

An informational message will be emitted when the test starts
execution:

    [INFO ][<timestamp>][<action name>] gst <gpu id> start <target_stress> copy matrix: <copy_matrix>


During the execution of the test, informational output providing the moving
average the GPU(s) gflops will be logged at each `log_interval`:

    [INFO ][<timestamp>][<action name>] gst Gflops: <interval_gflops>

When the target gflops is achieved, the following message will be logged:

    [INFO ][<timestamp>][<action name>] gst <gpu id> target achieved <target_stress>

If the target gflops, or stress, is not achieved in the `ramp_interval`
provided, the test will terminate and the following message will be logged:

    [INFO ][<timestamp>][<action name>] gst <gpu id> ramp time exceeded <ramp_time>

In this case the test will fail.\n

If the target stress (gflops) is achieved the test will attempt to run for the
rest of the duration specified by the action, sustaining the stress load during
that time. If the stress level violates the bounds set by the tolerance level
during that time a violation message will be logged:

    [INFO ][<timestamp>][<action name>] gst <gpu id> stress violation <interval_gflops>

When the test completes, the following result message will be printed:

    [RESULT][<timestamp>][<action name>] gst <gpu id> Gflop: <max_gflops> flops_per_op:<flops_per_op> bytes_copied_per_op: <bytes_copied_per_op> try_ops_per_sec: <try_ops_per_sec> pass: <pass>

The test will pass if the target_stress is reached before the end of the
`ramp_interval` and the `stress_violations` value is less than the given
`max_violations` value. Otherwise, the test will fail.

### Examples

When running the GST module, users should provide at least an action name,
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
values (for example: `copy_matrix=true`, `matrix_size=5760`). For more
information about the default
values, see Common configuration keys
and Configuration keys.

When the RVS tool runs against such a configuration file, it will do the
following:
  - run the stress test on all available (and compatible) AMD GPUs, one after
the other
  - log a start message containing the GPU ID, the `target_stress` and the
value of the `copy_matrix`:

    ```
    [INFO  ] [164337.932824] action_gst_1 gst 50599 start 3500.000000 copy matrix:true
    ```

  - emit, each `log_interval` (for example: 1000ms), a message containing the
gigaflops value that the current GPU achieved:

    ```
    [INFO  ] [164355.111207] action_gst_1 gst 33367 Gflops 3535.670231
    ```

  - log a message as soon as the current GPU reaches the given `target_stress`:

    ```
    [INFO  ] [164350.804843] action_gst_1 gst 33367 target achieved 500.000000
    ```

  - log a ramp time exceeded message if the GPU was not able to reach the
`target_stress` in the `ramp_interval` time frame (for example: 5000). In such a
case, the test will also terminate:

    ```
    [INFO  ] [164013.788870] action_gst_1 gst 3254 ramp time exceeded 5000
    ```

  - log the test result, when the stress test completes. The message contains
the test's overall result and some other statistics according in Output
keys:

    ```
    [RESULT] [164355.647523] action_gst_1 gst 33367 Gflop: 4066.020766 flops_per_op: 382.205952x1e9 bytes_copied_per_op: 398131200 try_ops_per_sec: 9.157367 pass: TRUE
    ```

  - log a stress violation message when the current gigaflops (for the last
`log_interval`, for example: 1000ms) violates the bounds set by the `tolerance`
configuration key (for example: 0.1). Note that this message is not logged
during the `ramp_interval` time frame:

    ```
    [INFO  ] [164013.788870] action_gst_1 gst 3254 stress violation 2500
    ```

If a mandatory configuration key is missing, the RVS tool will log an error
message and terminate the execution of the current module. For example, the
following configuration file will cause the RVS to terminate with the
following error message: `RVS-GST: action: action_gst_1  key 'target_stress' was not found`

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

For this configuration file, the RVS tool will:
  - run the stress test only for the GPUs having the ID 50599 or 33367. To
get all the available GPU IDs, run __RVS__ tool with __-g__ option
  - run the test on the selected GPUs, one after the other
  - run each test, 12 times
  - only copy the matrices to the GPUs at the beginning of the test
  - wait 100ms before each test execution
  - try to reach 5000 gflops in maximum 3000ms
  - if `target_stress` (5000) is achieved in the `ramp_interval` (3000 ms)
it will attempt to run the test for the rest of the duration, sustaining the
stress load during that time
  - allow a 7% `target_stress` tolerance (each `target_stress`
violation will generate a stress violation message as shown in the first
example)
  - allow only 2 `target_stress` violations. Exceeding the
`max_violations` will not terminate the test, but RVS will mark the
test result as "fail".

The output for such a configuration key may look like this:

    [INFO  ] [172061.758830] action_1 gst 50599 start 5000.000000 copy
    matrix:false
    [INFO  ] [172063.547668] action_1 gst 50599 Gflops 6471.614725
    [INFO  ] [172064.577715] action_1 gst 50599 target achieved 5000.000000
    [INFO  ] [172065.609224] action_1 gst 50599 Gflops 5189.993529
    [INFO  ] [172066.634360] action_1 gst 50599 Gflops 5220.373979
    [INFO  ] [172067.659262] action_1 gst 50599 Gflops 5225.472000
    [INFO  ] [172068.694305] action_1 gst 50599 Gflops 5169.935583
    [RESULT] [172069.573967] action_1 gst 50599 Gflop: 6471.614725 flops_per_op:
    382.205952x1e9 bytes_copied_per_op: 398131200 try_ops_per_sec: 13.081952 pass:
    TRUE
    [INFO  ] [172069.574369] action_1 gst 33367 start 5000.000000 copy
    matrix:false
    [INFO  ] [172071.409483] action_1 gst 33367 Gflops 6558.348080
    [INFO  ] [172072.438104] action_1 gst 33367 target achieved 5000.000000
    [INFO  ] [172073.465033] action_1 gst 33367 Gflops 5215.285895
    [INFO  ] [172074.501571] action_1 gst 33367 Gflops 5164.945297
    [INFO  ] [172075.529468] action_1 gst 33367 Gflops 5210.207720
    [INFO  ] [172076.558102] action_1 gst 33367 Gflops 5205.139424
    [RESULT] [172077.448182] action_1 gst 33367 Gflop: 6558.348080 flops_per_op:
    382.205952x1e9 bytes_copied_per_op: 398131200 try_ops_per_sec: 13.081952 pass:
    TRUE

When setting the parallel to true, RVS will run the stress tests on
all selected GPUs in parallel and the output may look like this:

    [INFO  ] [173381.407428] action_1 gst 50599 start 5000.000000 copy
    matrix:false
    [INFO  ] [173381.407744] action_1 gst 33367 start 5000.000000 copy
    matrix:false
    [INFO  ] [173383.245771] action_1 gst 33367 Gflops 6558.348080
    [INFO  ] [173383.256935] action_1 gst 50599 Gflops 6484.532120
    [INFO  ] [173384.274202] action_1 gst 33367 target achieved 5000.000000
    [INFO  ] [173384.286014] action_1 gst 50599 target achieved 5000.000000
    [INFO  ] [173385.301038] action_1 gst 33367 Gflops 5215.285895
    [INFO  ] [173385.315794] action_1 gst 50599 Gflops 5200.080980
    [INFO  ] [173386.337638] action_1 gst 33367 Gflops 5164.945297
    [INFO  ] [173386.353274] action_1 gst 50599 Gflops 5159.964636
    [INFO  ] [173387.365494] action_1 gst 33367 Gflops 5210.207720
    [INFO  ] [173387.383437] action_1 gst 50599 Gflops 5195.032357
    [INFO  ] [173388.401250] action_1 gst 33367 Gflops 5169.935583
    [INFO  ] [173388.421599] action_1 gst 50599 Gflops 5154.993572
    [RESULT] [173389.282710] action_1 gst 33367 Gflop: 6558.348080 flops_per_op:
    382.205952x1e9 bytes_copied_per_op: 398131200 try_ops_per_sec: 13.081952 pass:
    TRUE
    [RESULT] [173389.305479] action_1 gst 50599 Gflop: 6484.532120 flops_per_op:
    382.205952x1e9 bytes_copied_per_op: 398131200 try_ops_per_sec: 13.081952 pass:
    TRUE

It is important that all the configuration keys will be adjusted/fine-tuned
according to the actual GPUs and HW platform capabilities. For example, a matrix
size of 5760 should fit the VEGA 10 GPUs while 8640 should work with the VEGA 20
GPUs.

## IET module

The Input EDPp Test can be used to characterize the peak power capabilities of a
GPU (that is, TGP) for a sustained duration of time. This tool leverage GEMM workload
to drive the compute load on the GPU and check whether the power consumed meets configured
target power in watts. The GEMM compute workloads are also pre-configured. This verifies
that the GPUs can sustain a power level for a reasonable amount of time without problems
like thermal violations arising. The test passes if GPU power meets or crosses the
target power during the duration of the test else reported as fail.

This module should be used in conjunction with the GPU Monitor, to watch for
thermal, power and related anomalies while the target GPU(s) are under realistic
load conditions. By setting the appropriate parameters a user can ensure that
all GPUs in a node or cluster reach desired performance levels. Further analysis
of the generated stats can also show variations in the required power, clocks or
temperatures to reach these targets, and thus highlight GPUs or nodes that are
operating less efficiently.

### Module specific keys

Module specific keys are described in the table below:

<div class="pst-scrollable-table-container">
<table class="table table--middle-left">
<tr><th class="head">Config Key</th> <th class="head">Type</th><th class="head"> Description</th></tr>
<tr><td>target_power</td><td>Float</td>
<td>This is a floating point value specifying the target sustained power level
for the test.</td></tr>
<tr><td>ramp_interval</td><td>Integer</td>
<td>This is a time interval, specified in milliseconds, given to the test to
determine the compute load that will sustain the target power. The default value
is 5000 (5 seconds). This time is counted against the duration of the test.
</td></tr>
<tr><td>tolerance</td><td>Float</td>
<td>A value indicating how much the target_power can fluctuate after the ramp
period for the test to succeed. The default value is 0 (any violation
immediately fails the test).
</td></tr>
<tr><td>max_violations</td><td>Integer</td>
<td>The number of tolerance violations that can occur after the ramp_interval
for the test to still pass. The default value is 0.</td></tr>
<tr><td>sample_interval</td><td>Integer</td>
<td>The sampling rate for target_power values given in milliseconds. The default
value is 1000 (1 second).
</td></tr>
<tr><td>log_interval</td><td>Integer</td>
<td>This is a positive integer, given in milliseconds, that specifies an
interval over which the moving average of the bandwidth will be calculated and
logged.</td></tr>

<tr><td>cp_workload</td><td>Bool</td>
<td>If true, enables the GEMM compute workload to drive GPU power. This is the
primary workload used to reach the target power level. The default value is
true.</td></tr>

<tr><td>bw_workload</td><td>Bool</td>
<td>If true, enables a memory bandwidth kernel workload in addition to or
instead of the GEMM workload. The default value is false.</td></tr>

<tr><td>wg_count</td><td>Integer</td>
<td>Number of GPU workgroups used for the bandwidth kernel. Only applicable
when <b>bw_workload: true</b>. The default value is 80.</td></tr>

<tr><td>nt_loads</td><td>Bool</td>
<td>If true, the bandwidth kernel uses non-temporal loads that bypass the L2
cache, exercising memory bandwidth more directly. Only applicable when
<b>bw_workload: true</b>. The default value is false.</td></tr>

<tr><td>matrix_size</td><td>Integer</td>
<td>Sets all three GEMM matrix dimensions (M, N, and K) to the same value.
Equivalent to setting <b>matrix_size_a</b>, <b>matrix_size_b</b>, and
<b>matrix_size_c</b> to the same value. The default value is 5760.</td></tr>

<tr><td>matrix_size_a</td><td>Integer</td>
<td>Number of rows of matrix A (the M dimension in GEMM). Overrides
<b>matrix_size</b> for this dimension. Default is 0 (uses
<b>matrix_size</b>).</td></tr>

<tr><td>matrix_size_b</td><td>Integer</td>
<td>Number of columns of matrix B (the N dimension in GEMM). Overrides
<b>matrix_size</b> for this dimension. Default is 0 (uses
<b>matrix_size</b>).</td></tr>

<tr><td>matrix_size_c</td><td>Integer</td>
<td>Inner (shared) dimension K of the GEMM operation. Overrides
<b>matrix_size</b> for this dimension. Default is 0 (uses
<b>matrix_size</b>).</td></tr>

<tr><td>ops_type</td><td>String</td>
<td>GEMM operation type. Accepted values: <b>sgemm</b>, <b>dgemm</b>,
<b>hgemm</b>. If neither <b>ops_type</b> nor <b>data_type</b> is set, the
module defaults to <b>sgemm</b>. Mutually exclusive with
<b>data_type</b>.</td></tr>

<tr><td>data_type</td><td>String</td>
<td>Data type for the GEMM computation. Accepted values include:
<b>fp4_r</b>, <b>fp6_r</b>, <b>fp8_r</b>, <b>bf16_r</b>, <b>fp16_r</b>,
<b>fp32_r</b>, <b>tf32_r</b> (<b>xf32_r</b>), <b>i8_r</b>. If not specified
the module falls back to the type implied by <b>ops_type</b>.</td></tr>

<tr><td>out_data_type</td><td>String</td>
<td>Output (result) data type, e.g. <b>fp16_r</b>, <b>fp32_r</b>. Only
applicable when using hipBLASLt. If not specified the default matches the
compute type.</td></tr>

<tr><td>compute_type</td><td>String</td>
<td>Accumulation/compute type used internally by the BLAS library. If not
specified the default is <b>fp32_r</b>.</td></tr>

<tr><td>blas_source</td><td>String</td>
<td>BLAS library backend to use. Accepted values:
<b>rocblas</b> (default), <b>hipblaslt</b>.</td></tr>

<tr><td>hot_calls</td><td>Integer</td>
<td>Number of GEMM kernel invocations per measurement window used to amortise
launch overhead. The default value is 1.</td></tr>

<tr><td>matrix_init</td><td>String</td>
<td>Matrix initialization method. Accepted values:
<b>default</b> – Initialize with default pattern (default).
<b>trig</b> – Initialize with trigonometric (sine/cosine) values.
<b>random</b> – Initialize with random values.</td></tr>

<tr><td>transa</td><td>Integer</td>
<td>Transpose operation applied to matrix A before the GEMM call.
0 = no transpose (default), 1 = transpose.</td></tr>

<tr><td>transb</td><td>Integer</td>
<td>Transpose operation applied to matrix B before the GEMM call.
0 = no transpose, 1 = transpose (default).</td></tr>

<tr><td>alpha</td><td>Float</td>
<td>Scalar multiplier applied to the product of matrices A and B
(C = alpha * A * B + beta * C). The default value is 1.</td></tr>

<tr><td>beta</td><td>Float</td>
<td>Scalar multiplier applied to matrix C in the GEMM operation. The default
value is 1.</td></tr>

<tr><td>lda</td><td>Integer</td>
<td>Leading dimension offset for matrix A. The default value is 0.</td></tr>

<tr><td>ldb</td><td>Integer</td>
<td>Leading dimension offset for matrix B. The default value is 0.</td></tr>

<tr><td>ldc</td><td>Integer</td>
<td>Leading dimension offset for matrix C. The default value is 0.</td></tr>

<tr><td>ldd</td><td>Integer</td>
<td>Leading dimension offset for matrix D (output). The default value is
0.</td></tr>

<tr><td>gemm_mode</td><td>String</td>
<td>GEMM execution mode. Accepted values:
<b>""</b> or unset – Standard (single) GEMM (default).
<b>batched</b> – Batched GEMM; use with <b>batch_size</b>.
<b>strided_batched</b> – Strided batched GEMM; use with <b>batch_size</b>
and <b>stride_*</b> keys.</td></tr>

<tr><td>batch_size</td><td>Integer</td>
<td>Number of GEMM operations in a batched or strided-batched call. Only
applicable when <b>gemm_mode</b> is <b>batched</b> or
<b>strided_batched</b>. The default value is 0.</td></tr>

<tr><td>stride_a</td><td>Integer</td>
<td>Stride (in elements) between consecutive matrices A in a
strided-batched GEMM. The default value is 0.</td></tr>

<tr><td>stride_b</td><td>Integer</td>
<td>Stride (in elements) between consecutive matrices B in a
strided-batched GEMM. The default value is 0.</td></tr>

<tr><td>stride_c</td><td>Integer</td>
<td>Stride (in elements) between consecutive matrices C in a
strided-batched GEMM. The default value is 0.</td></tr>

<tr><td>stride_d</td><td>Integer</td>
<td>Stride (in elements) between consecutive output matrices D in a
strided-batched GEMM. The default value is 0.</td></tr>
</table>
</div>


### Output

Module specific output keys are described in the table below:

<div class="pst-scrollable-table-container">
<table class="table table--middle-left">
<tr><th class="head">Output Key</th> <th class="head">Type</th><th class="head"> Description</th></tr>
<tr><td>current_power</td><td>Time Series Floats</td>
<td>The current measured power of the GPU.</td></tr>
<tr><td>power_violations</td><td>Integer</td>
<td>The number of power readings that violated the tolerance of the test after
the ramp interval.
</td></tr>
<tr><td>pass</td><td>Bool</td>
<td>'true' if the GPU achieves its desired sustained power level in the ramp
interval.</td></tr>
</table>
</div>

### Examples

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

```{note}
- when setting the `device` configuration key to `all`, the RVS will detect all the AMD compatible GPUs and run the test on all of them
- the test will run 2 times on each GPU (count = 2)
- only one power violation is allowed. If the total number of violations is bigger than 1 the IET test result will be marked as `failed`
```

When the RVS tool runs against such a configuration file, it will do the following:

- run the test on all AMD compatible GPUs

- log a start message containing the GPU ID and the `target_power`, for example:

    ```
    [INFO ] [167316.308057] action_1 iet 50599 start 135.000000
    ```

- emit, each log_interval (for example: 500ms), a message containing the power for the current GPU

    ```
    [INFO ] [167319.266707] action_1 iet 50599 current power 136.878342
    ```

- log a message as soon as the current GPU reaches the given `target_power`

    ```
    [INFO ] [167318.793062] action_1 iet 50599 target achieved 135.000000
    ```

- log a 'ramp time exceeded' message if the GPU was not able to reach the `target_power` in the `ramp_interval` time frame (for example: 5000ms). In such a case, the test will also terminate

    ```
    [INFO ] [167648.832413] action_1 iet 50599 ramp time exceeded 5000
    ```

- log a 'power violation message' when the current power (for the last `sample_interval`, for example: 500ms) violates the bounds set by the tolerance configuration key (for example: 0.1). Please note that this message is never logged during the `ramp_interval` time frame

    ```
    [INFO ] [161251.971277] action_1 iet 3254 power violation 73.783211
    ```

- log the test result, when the stress test completes.

    ```
    [RESULT] [167305.260051] action_1 iet 33367 pass: TRUE
    ```

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


- All the missing configuration keys (if any) will have their default values. For more information about the default values please consult the dedicated sections (3.3 Common Configuration Keys and 13.1 Module specific keys).
- If a mandatory configuration key is missing, the RVS tool will log an error message and terminate the execution of the current module. For example, if the target_power is missing, the RVS to terminate with the following error message: "RVS-IET: action: action_1 key 'target_power' was not found"
- All the configuration keys will be adjusted/fine-tuned according to the actual GPUs and HW platform capabilities.
    - For example, a matrix size of 5760 should fit the VEGA 10 GPUs while 8640 should work with the VEGA 20 GPUs
    - For small` target_power` values (for example: 30-40W), the `sample_interval` should be increased, otherwise the IET may fail either to achieve the given `target_power` or to sustain it (for example: `ramp_interval = 1500` for `target_power = 40`). In case there are problems reaching/sustaining the given `target_power`: 
        - Increase the `ramp_interval` and/or the tolerance value(s) and try again (in case of a 'ramp time exceeded' message)
        - Increase the tolerance value (in case too many 'power violation message' are logged out)


## Pulse Module

```{warning}
This module is in beta and is not intended for production use. Pass/fail criteria (especially around power-delta enforcement, clock-pinning verification, and throttle detection) are still being tuned and may change between releases.
```

The **Pulse** module is intended for **time-varying** GPU power stress: it alternates **high** phases (maximum clocks plus continuous GEMM) and **low** phases (minimum clocks plus idle/sleep), repeating for the action **duration**. That produces periodic power swings useful for exercising PSU transient response and platform power delivery, complementing **IET** (which targets a sustained power level).

GEMM type, matrix size, and BLAS backend follow the same concepts as **GST** / **IET** (see those sections and `rvs_blas`). Sample reference configuration: `rvs/conf/pulse_single.conf`.

### Behavior summary

- Each **pulse cycle** is one high phase followed by one low phase. Phase lengths derive from `pulse_rate` (Hz) and `high_phase_ratio`; each phase is at least 10 ms after rounding.
- **High phase:** sets GPU clocks high, runs GEMM in a loop (`workload_iterations` per inner batch) until the high-phase time budget elapses, samples power, checks junction temperature (**failure if above 105°C**).
- After the high phase, the module calls `hipDeviceSynchronize` so work drains before the low phase (clearer separation of high vs. low power).
- **Low phase:** sets clocks low, sleeps briefly between `sample_interval`-style sampling (5 ms sleep steps in the implementation), samples power.
- With `parallel: true` and more than one GPU, threads coordinate with a `std::barrier` and a small GPU kernel using fine-grained coherent host memory so all GPUs synchronize across the pulse loop (avoids deadlock on shutdown via a shared done flag).
- **Pass:** for primary MCM dies, the action passes if at least one pulse completed, there was **no** BLAS enqueue/sync failure (unless `halt_on_error` stops earlier), and no thermal violation; secondary MCM GPUs are treated as pass by default. Fail otherwise.

### Prerequisites

- ROCm with hipBLASLt and rocBLAS available as for the rest of RVS (the `rvs` binary links hipblaslt; GEMM code lives in rvslib).
- AMD SMI initialized for power and temperature queries (same family of requirements as other SMI-based modules). Elevated privileges are often required for clock control and power metrics.
- For hipBLASLt, `data_type` must be valid (for example, `fp32_r` with `sgemm`, `fp16_r` with `hgemm` and `compute_type` such as `fp32_r`). If `data_type` is omitted with `blas_source: hipblaslt`, the module infers `fp32_r`/`fp64_r`/`fp16_r` from `ops_type`/`sgemm`/`dgemm`/`hgemm` respectively; for `dgemm` with hipBLASLt, if `compute_type` is still the default `fp32_r`, it is adjusted to `fp64_r`.

### Module specific keys

Keys below are in addition to common keys (`name`, `module`, `device`, `duration`, `parallel`, `log_interval`, `count`, `wait` and so on. For more information, see Common configuration keys.

<div class="pst-scrollable-table-container">
<table class="table table--middle-left">
<tr><th class="head">Config Key</th class="head"> <th class="head">Type</th><th class="head">Description</th></tr>
<tr><td>pulse_rate</td><td>Integer</td><td>Pulse frequency in **Hertz** (cycles per second). Default <b>2</b>. Use roughly **1–5 Hz** if you want power readings to show clear high/low separation; very high rates shorten phases below typical GPU power-state and SMI averaging windows, so reported averages may converge.</td></tr>
<tr><td>high_phase_ratio</td><td>Float</td><td>Fraction of each cycle spent in the **high** (GEMM) phase, **0.0–1.0**. Default <b>0.5</b>. Lower values spend more time in the low phase.</td></tr>
<tr><td>matrix_size</td><td>Integer</td><td>Square GEMM dimension **M = N = K**. Default <b>4096</b>. Larger matrices increase compute and power draw during the high phase.</td></tr>
<tr><td>ops_type</td><td>String</td><td>GEMM operation flavor (e.g. <b>sgemm</b>, <b>dgemm</b>, <b>hgemm</b>). Default <b>sgemm</b>. Must match **data_type** / **compute_type** when using hipBLASLt.</td></tr>
<tr><td>data_type</td><td>String</td><td>Matrix data type string for hipBLASLt (e.g. <b>fp32_r</b>, <b>fp16_r</b>). Default empty (rocBLAS can infer from **ops_type** alone).</td></tr>
<tr><td>out_data_type</td><td>String</td><td>Optional output type for GEMM; default empty (same as input type where applicable).</td></tr>
<tr><td>compute_type</td><td>String</td><td>hipBLASLt compute type (e.g. <b>fp32_r</b>). Default <b>fp32_r</b>.</td></tr>
<tr><td>blas_source</td><td>String</td><td><b>rocblas</b> or <b>hipblaslt</b>. Default <b>rocblas</b>.</td></tr>
<tr><td>alpha</td><td>Float</td><td>GEMM scalar α. Default <b>2.0</b>.</td></tr>
<tr><td>beta</td><td>Float</td><td>GEMM scalar β. Default <b>-1.0</b>.</td></tr>
<tr><td>transa</td><td>Integer</td><td>Transpose A: **0** no transpose, **1** transpose. Default <b>0</b>.</td></tr>
<tr><td>transb</td><td>Integer</td><td>Transpose B: **0** no transpose, **1** transpose. Default <b>1</b>.</td></tr>
<tr><td>lda</td><td>Integer</td><td>Leading dimension offset A (0 = use minimum). Defaults <b>0</b>.</td></tr>
<tr><td>ldb</td><td>Integer</td><td>Leading dimension offset B. Default <b>0</b>.</td></tr>
<tr><td>ldc</td><td>Integer</td><td>Leading dimension offset C. Default <b>0</b>.</td></tr>
<tr><td>ldd</td><td>Integer</td><td>Leading dimension offset D. Default <b>0</b>.</td></tr>
<tr><td>matrix_init</td><td>String</td><td>Host matrix initialization (e.g. <b>default</b>, <b>hiprand</b>). Default <b>default</b>.</td></tr>
<tr><td>workload_iterations</td><td>Integer</td><td>GEMM calls per inner batch in the high phase before re-checking time and power. Default <b>128</b>. Increase for heavier per-iteration work; tune with **pulse_rate** and phase length.</td></tr>
<tr><td>sample_interval</td><td>Integer</td><td>Reserved sampling interval (ms) for pulse-specific logic; values below **50** are raised to **50**. Default <b>100</b>.</td></tr>
<tr><td>tolerance</td><td>Float</td><td>Default <b>10.0</b>. Parsed and passed to the worker; **not** currently used in pass/fail logic (reserved for future checks).</td></tr>
<tr><td>verify_mode</td><td>String</td><td>e.g. <b>diff</b> / <b>crc</b>. Default <b>diff</b>. Parsed; **not** currently used in pass/fail logic (reserved for future GEMM verification).</td></tr>
<tr><td>halt_on_error</td><td>Bool</td><td>If true, stop the GPU thread on first BLAS or thermal error. Default <b>false</b>.</td></tr>
<tr><td>hot_calls</td><td>Integer</td><td>BLAS “hot call” / warmup-related parameter forwarded to **rvs_blas**. Default <b>1</b>.</td></tr>
<tr><td>gpu_sync_wait</td><td>Integer</td><td>Default <b>10000</b>. Parsed from configuration; **not** referenced by the current barrier implementation (placeholder for future timeout behavior).</td></tr>
<tr><td>max_temp_c</td><td>Float</td><td>Junction temperature ceiling in degrees Celsius. If the GPU junction temperature exceeds this threshold during the run, the worker logs a thermal-violation error and, when <b>halt_on_error</b> is true, terminates that GPU thread. Default <b>105.0</b>.</td></tr>
</table>
</div>

### Output

Log lines use the action name, module tag pulse, and GPU ID.

<div class="pst-scrollable-table-container">
<table class="table table--middle-left">
<tr><th class="head">Output / log</th> <th class="head">Description</th></tr>
<tr><td>Start</td><td><code>[INFO] ... pulse &lt;gpu_id&gt; start pulse_rate=&lt;hz&gt;</code></td></tr>
<tr><td>Parameters</td><td><code>[INFO] ... pulse_rate=... Hz period=...ms high=...ms low=...ms</code></td></tr>
<tr><td>Thermal error</td><td><code>[ERROR] ... thermal violation: &lt;temp&gt;C</code> (junction &gt; 105°C)</td></tr>
<tr><td>Periodic summary</td><td>At each <b>log_interval</b>, moving averages and extrema: <code>pulse #N avg_high=...W avg_low=...W max_high=...W min_low=...W delta=...W</code></td></tr>
<tr><td>Completion</td><td>Summary over the run: pulse count, average high/low power, delta, max high, min low.</td></tr>
<tr><td>pass</td><td><code>[RESULT] ... pass: TRUE</code> or <code>FALSE</code> per GPU.</td></tr>
</table>
</div>

With JSON logging enabled, per-pulse records can include `power_high_w`, `power_low_w`, `power_delta_w`, phase durations, `temp_c`, `gemm_count`, and a final summary with pass.

### Example

Minimal illustration (see `pulse_single.conf` for full examples including hipblaslt + `fp16_r`):

    actions:
    - name: pulse_stress_basic
      device: all
      module: pulse
      parallel: true
      duration: 60000
      log_interval: 5000
      pulse_rate: 2
      high_phase_ratio: 0.5
      ops_type: sgemm
      matrix_size: 8192
      blas_source: hipblaslt
      data_type: fp32_r
      compute_type: fp32_r
      workload_iterations: 128
      halt_on_error: false

Run from the build or package `bin` directory, for example:

    ./rvs -c conf/pulse_single.conf -d 3


```{note}
- Tune `matrix_size`, `pulse_rate`, and `high_phase_ratio` to match GPU class and the transient behavior you want to stress.
- hipBLASLt often delivers higher GEMM throughput than rocBLAS on supported GPUs; `fp16_r`/`hgemm` with `compute_type`: `fp32_r` is a common high-throughput choice (as in the `pulse_stress_fp16` action in `pulse_single.conf`).
```

## MEM module

The Memory module tests GPU memory for hardware errors and soft errors using
HIP. It executes a configurable suite of memory test algorithms that exercise
various data patterns and access sequences. Each test can be individually
included or excluded. The module reports errors found per test and passes only
if no memory errors are detected.

The following tests are available (referenced by index in `exclude`):

| Index | Test Name |
|---|---|
| 0 | Walking 1 bit |
| 1 | Own address test |
| 2 | Moving inversions, ones & zeros |
| 3 | Moving inversions, 8-bit pattern |
| 4 | Moving inversions, random pattern |
| 5 | Block move, 64 moves |
| 6 | Moving inversions, 32-bit pattern |
| 7 | Random number sequence |
| 8 | Modulo 20, random pattern |
| 9 | Bit fade test |
| 10 | Memory stress test |

### Module specific keys

<div class="pst-scrollable-table-container">
<table class="table table--middle-left">
<tr><th class="head">Config Key</th> <th class="head">Type</th><th class="head"> Description</th></tr>
<tr><td>mem_blocks</td><td>Integer</td>
<td>Number of GPU memory blocks used per test iteration. The default value is
256.</td></tr>
<tr><td>num_passes</td><td>Integer</td>
<td>Number of passes (repeats) per block in each test. The default value is
1.</td></tr>
<tr><td>thrds_per_blk</td><td>Integer</td>
<td>Number of HIP threads per block launched for each test kernel. The default
value is 128.</td></tr>
<tr><td>stress</td><td>Bool</td>
<td>If true, enables the memory stress test (Test 10) in addition to the
standard tests. The default value is false.</td></tr>
<tr><td>mapped_memory</td><td>Bool</td>
<td>If true, uses host-mapped (pinned) memory instead of device memory for the
test buffers. The default value is false.</td></tr>
<tr><td>num_iter</td><td>Integer</td>
<td>Number of iterations to run per test. The default value is 1.</td></tr>
<tr><td>exclude</td><td>Collection of Integers</td>
<td>Space-separated list of test indices (0–10) to skip. All tests not listed
here will be executed. For example, <b>exclude: 9 10</b> skips the bit fade
and memory stress tests.</td></tr>
</table>
</div>

### Output

<div class="pst-scrollable-table-container">
<table class="table table--middle-left">
<tr><th class="head">Output Key</th> <th class="head">Type</th><th class="head"> Description</th></tr>
<tr><td>Test</td><td>String</td>
<td>Name of the memory test that was executed.</td></tr>
<tr><td>Time Taken</td><td>Float</td>
<td>Time in seconds taken to complete the test on this GPU.</td></tr>
<tr><td>errors</td><td>Integer</td>
<td>Number of memory errors detected during the test. A value of 0 indicates
no errors.</td></tr>
<tr><td>pass</td><td>Bool</td>
<td>True if no memory errors were detected for this test.</td></tr>
</table>
</div>

### Examples

**Example 1:**

Run all tests in parallel on all GPUs, excluding bit fade (9) and stress (10):

    actions:
    - name: action_1
      device: all
      module: mem
      parallel: true
      count: 1
      wait: 100
      mapped_memory: false
      mem_blocks: 128
      num_passes: 500
      thrds_per_blk: 64
      stress: true
      num_iter: 50000
      exclude: 9 10

Run with:

    ./rvs -c conf/mem.conf -d 3

The test passes if no memory errors are detected on any GPU. A typical result
message looks like:

    [RESULT][<timestamp>][action_1] mem <gpu_id> Test 1  [Walking 1 bit] : pass: true  errors: 0 Time Taken: 0.652 s
    [RESULT][<timestamp>][action_1] mem <gpu_id> Test 2  [Own address test] : pass: true  errors: 0 Time Taken: 0.423 s


## BABEL module

The BABEL module executes BabelStream benchmark tests that measure GPU memory
bandwidth. BabelStream is a synthetic benchmark based on the STREAM benchmark
for CPUs. It runs configurable memory kernels (Copy, Mul, Add, Triad, Dot,
Read, Write) using HIP and reports the achieved bandwidth in GB/s or GiB/s for
each kernel. The benchmark is useful for characterizing peak memory bandwidth
and detecting bandwidth regressions.

### Module specific keys

<div class="pst-scrollable-table-container">
<table class="table table--middle-left">
<tr><th class="head">Config Key</th> <th class="head">Type</th><th class="head"> Description</th></tr>
<tr><td>array_size</td><td>Integer</td>
<td>Size of the test buffer in bytes (or in mebibytes if <b>mibibytes: true</b>).
The default value is 33554432 (32 MB).</td></tr>
<tr><td>test_type</td><td>Integer</td>
<td>Data precision used for the benchmark. Accepted values:
<b>1</b> – Float (32-bit, default).
<b>2</b> – Double (64-bit).
<b>3</b> – Triad float.
<b>4</b> – Triad double.</td></tr>
<tr><td>num_iter</td><td>Integer</td>
<td>Number of kernel launch iterations. Used when <b>duration</b> is 0. The
default value is 100.</td></tr>
<tr><td>duration</td><td>Integer</td>
<td>Duration of the test in milliseconds. When greater than 0, the test runs
for this time instead of a fixed number of iterations. A value of 0 means use
<b>num_iter</b>. The default value is 0.</td></tr>
<tr><td>mibibytes</td><td>Bool</td>
<td>If true, <b>array_size</b> is interpreted in mebibytes (MiB) and bandwidth
is reported in GiB/s. If false, bytes and GB/s are used. The default value is
false.</td></tr>
<tr><td>o/p_csv</td><td>Bool</td>
<td>If true, outputs results in CSV format in addition to the standard log.
The default value is false.</td></tr>
<tr><td>read</td><td>Bool</td>
<td>If true, enables the Read kernel (measures read-only bandwidth). The
default value is false.</td></tr>
<tr><td>write</td><td>Bool</td>
<td>If true, enables the Write kernel (measures write-only bandwidth). The
default value is false.</td></tr>
<tr><td>copy</td><td>Bool</td>
<td>If true, enables the Copy kernel (a[i] = b[i]). The default value is
false.</td></tr>
<tr><td>mul</td><td>Bool</td>
<td>If true, enables the Mul kernel (a[i] = scalar * b[i]). The default value
is false.</td></tr>
<tr><td>add</td><td>Bool</td>
<td>If true, enables the Add kernel (a[i] = b[i] + c[i]). The default value
is false.</td></tr>
<tr><td>dot</td><td>Bool</td>
<td>If true, enables the Dot kernel (sum of a[i] * b[i]). The default value
is false.</td></tr>
<tr><td>triad</td><td>Bool</td>
<td>If true, enables the Triad kernel (a[i] = b[i] + scalar * c[i]). The
default value is false.</td></tr>
<tr><td>data_init</td><td>String</td>
<td>Data initialization method for the test arrays. Accepted values:
<b>default</b> – Initialize with default constant values (default).
Other values may be supported depending on the build.</td></tr>
<tr><td>nontemporal</td><td>String</td>
<td>Non-temporal (streaming) memory access mode for the kernels. Non-temporal
stores bypass the cache and can be useful for measuring true memory bandwidth.
Accepted values:
<b>all</b> – Apply non-temporal accesses to all kernels (default).
<b>none</b> – Disable non-temporal accesses.
<b>read</b> – Apply only to read accesses.
<b>write</b> – Apply only to write accesses.</td></tr>
<tr><td>dwords_per_lane</td><td>Integer</td>
<td>Number of 32-bit words processed per GPU lane per kernel invocation. The
default value is 4.</td></tr>
<tr><td>chunks_per_block</td><td>Integer</td>
<td>Number of data chunks processed per GPU thread block. The default value
is 2.</td></tr>
<tr><td>tb_size</td><td>Integer</td>
<td>Thread block size (number of threads per block). The default value is
1024.</td></tr>
</table>
</div>

### Output

<div class="pst-scrollable-table-container">
<table class="table table--middle-left">
<tr><th class="head">Output Key</th> <th class="head">Type</th><th class="head"> Description</th></tr>
<tr><td>Array size</td><td>Integer</td>
<td>Size of the test array used for the measurement, in bytes or MiB depending
on the <b>mibibytes</b> setting.</td></tr>
<tr><td>Function</td><td>String</td>
<td>Name of the BabelStream kernel executed (e.g., Copy, Mul, Add, Triad,
Dot, Read, Write).</td></tr>
<tr><td>bandwidth</td><td>Float</td>
<td>Measured memory bandwidth for this kernel in GB/s (or GiB/s if
<b>mibibytes: true</b>).</td></tr>
<tr><td>pass</td><td>Bool</td>
<td>True if the kernel completed successfully.</td></tr>
</table>
</div>

### Examples

**Example 1:**

Run all BABEL kernels with a 32 MB buffer, 5000 iterations, double precision:

    actions:
    - name: action_1
      device: all
      module: babel
      parallel: true
      count: 1
      num_iter: 5000
      duration: 0
      array_size: 33554432
      test_type: 2
      mibibytes: false
      o/p_csv: false
      copy: true
      read: true
      write: true
      mul: true
      add: true
      dot: true
      triad: true

Run with:

    ./rvs -c conf/babel.conf -d 3
