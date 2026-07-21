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
Overrides the <b>duration</b> value set in all actions of the configuration file.
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

Override the duration for all actions to 30 seconds:
```bash
./rvs -c conf/gst_single.conf -t 30
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

<tr><td>device_index</td><td>Integer</td><td>This is an optional parameter that
restricts the action to a GPU identified by its SMI index. Can also be
overridden via the <b>-i</b> CLI option.</td></tr>

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
wait between executions, in milliseconds. The default value is 500 ms.
Some modules will ignore this parameter.</td></tr>

<tr><td>duration</td><td>Integer</td><td>This indicates how long the test
should run, given in milliseconds. When specified, it takes precedence
over the <b>count</b> key for modules that support it. Some modules will
ignore this parameter.</td></tr>


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
indicating the device index of the GPU. Use the `device_index` common key to
target specific GPUs by their topology node index.

Note: GPUP ignores the `count`, `duration`, `wait`, `parallel`, and
`log_interval` common keys — each action performs a single-pass property dump.

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

**Example:**

Run:

    ./rvs -c conf/gpup_single.conf

Configuration (`conf/gpup_single.conf`, first action):

    actions:
    - name: RVS-GPUP-TC1
      device: all
      module: gpup
      properties:
        all:
      io_links-properties:
        all:

Sample output (first action, abridged):

    [RESULT] [302733.913494] Action name :RVS-GPUP-TC1
    [RESULT] [302733.975662] Module name :gpup
    [RESULT] [302733.975835] [RVS-GPUP-TC1] gpup 42583 cpu_cores_count 0
    [RESULT] [302733.975836] [RVS-GPUP-TC1] gpup 42583 simd_count 1024
    [RESULT] [302733.975836] [RVS-GPUP-TC1] gpup 42583 mem_banks_count 1
    [RESULT] [302733.975836] [RVS-GPUP-TC1] gpup 42583 caches_count 550
    [RESULT] [302733.975837] [RVS-GPUP-TC1] gpup 42583 io_links_count 8
    [RESULT] [302733.975837] [RVS-GPUP-TC1] gpup 42583 p2p_links_count 1
    [RESULT] [302733.975837] [RVS-GPUP-TC1] gpup 42583 cpu_core_id_base 0
    [RESULT] [302733.975838] [RVS-GPUP-TC1] gpup 42583 simd_id_base 2147487744
    ...
    [RESULT] [302733.976401] [RVS-GPUP-TC1] gpup 57875 7 version_major 0
    [RESULT] [302733.976401] [RVS-GPUP-TC1] gpup 57875 7 version_minor 0
    [RESULT] [302733.976402] [RVS-GPUP-TC1] gpup 57875 7 node_from 9
    [RESULT] [302733.976402] [RVS-GPUP-TC1] gpup 57875 7 node_to 8
    [RESULT] [302733.976402] [RVS-GPUP-TC1] gpup 57875 7 weight 15
    [RESULT] [302733.976403] [RVS-GPUP-TC1] gpup 57875 7 min_latency 0
    [RESULT] [302733.976403] [RVS-GPUP-TC1] gpup 57875 7 max_latency 0
    [RESULT] [302733.976403] [RVS-GPUP-TC1] gpup 57875 7 min_bandwidth 0
    [RESULT] [302733.976403] [RVS-GPUP-TC1] gpup 57875 7 recommended_transfer_size 0
    [RESULT] [302733.976404] [RVS-GPUP-TC1] gpup 57875 7 flags 1

    +=====================================================================+
    |                 ROCm Validation Suite (RVS) Summary                 |
    +=====================================================================+
    |                           System Overview                           |
    +---------------------------------------------------------------------+
    | Operating System                 | Ubuntu 22.04.5 LTS               |
    | RVS version                      | 1.6.75                           |
    | ROCm version                     | 7.2.1-81                         |
    | amdgpu version                   | 6.16.13                          |
    | GPUs                             | 8                                |
    +---------------------------------------------------------------------+
    | AMD Instinct MI355X - 42583      | AMD Instinct MI355X - 27226      |
    | 0 - 2 - 0000:05:00.0             | 1 - 3 - 0000:15:00.0             |
    +---------------------------------------------------------------------+
    | AMD Instinct MI355X - 36479      | AMD Instinct MI355X - 17010      |
    | 2 - 4 - 0000:65:00.0             | 3 - 5 - 0000:75:00.0             |
    +---------------------------------------------------------------------+
    | AMD Instinct MI355X - 1590       | AMD Instinct MI355X - 51771      |
    | 4 - 6 - 0000:85:00.0             | 5 - 7 - 0000:95:00.0             |
    +---------------------------------------------------------------------+
    | AMD Instinct MI355X - 11806      | AMD Instinct MI355X - 57875      |
    | 6 - 8 - 0000:e5:00.0             | 7 - 9 - 0000:f5:00.0             |
    +=====================================================================+
    | Action Name                      | Module         | Result          |
    +=====================================================================+
    | RVS-GPUP-TC1                     | GPUP           | PASS            |
    +---------------------------------------------------------------------+


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
        the sample_interval are milliseconds. The default value is 500.
      </td>
    </tr>
    <tr>
      <td>log_interval</td>
      <td>Integer</td>
      <td>
        If this key is specified informational messages will be emitted at the given
        interval, providing the current values of all parameters specified. This parameter
        must be equal to or greater than the sample_interval. The default value is 1000 ms
        (logging is ON by default). Set to 0 to disable periodic logging.
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
be sampled at every interval specified by the `sample_interval` key. If a
bounding box violation is discovered during a sampling interval, a warning
message is logged with the following format:

    [INFO ][<timestamp>][<action name>] gm <gpu id> <metric> bounds violation <metric value>

If the log_interval value is set an information message for each metric is
logged at every interval using the following format:

    [INFO ][<timestamp>][<action name>] gm <gpu id> <metric> <metric_value>

When monitoring is stopped for a target GPU, a result message is logged
with the following format:

    [RESULT][<timestamp>][<action name>] gm <gpu id> stopped

The following messages, reporting the number of metric violations that were
sampled over the duration of the monitoring and the average metric value is
reported:

    [RESULT][<timestamp>][<action name>] gm <gpu id> <metric> violations <metric_violations>
    [RESULT][<timestamp>][<action name>] gm <gpu id> <metric> average <metric_average>

### Examples

**Example:**

Run:

    ./rvs -c conf/gm_single.conf

Configuration (`conf/gm_single.conf`, first action):

    actions:
    - name: metrics_monitor
      module: gm
      device: all
      monitor: true
      metrics:
        temp: true 100 0
        fan: true 100 0
        mem_clock: true 1000 0
        clock: true 1000 0
        power: true 750 0
      duration: 10000

Sample output (first action, abridged):

    [RESULT] [302734.420713] Action name :metrics_monitor
    [RESULT] [302734.503180] Module name :gm
    [RESULT] [302734.503340] [metrics_monitor] gm 1590 started
    [RESULT] [302734.503340] [metrics_monitor] gm 11806 started
    [RESULT] [302734.503340] [metrics_monitor] gm 17010 started
    [RESULT] [302734.503340] [metrics_monitor] gm 27226 started
    [RESULT] [302734.503340] [metrics_monitor] gm 36479 started
    [RESULT] [302734.503340] [metrics_monitor] gm 42583 started
    [RESULT] [302734.503340] [metrics_monitor] gm 51771 started
    [RESULT] [302734.503340] [metrics_monitor] gm 57875 started
    ...
    [RESULT] [302744.940440] [metrics_monitor] gm 57875 temp violations 0
    [RESULT] [302744.940440] [metrics_monitor] gm 57875 temp average 0C
    [RESULT] [302744.940440] [metrics_monitor] gm 57875 clock violations 0
    [RESULT] [302744.940440] [metrics_monitor] gm 57875 clock average 128MHz
    [RESULT] [302744.940440] [metrics_monitor] gm 57875 mem_clock violations 0
    [RESULT] [302744.940440] [metrics_monitor] gm 57875 mem_clock average 128MHz
    [RESULT] [302744.940440] [metrics_monitor] gm 57875 fan violations 0
    [RESULT] [302744.940440] [metrics_monitor] gm 57875 fan average 0%
    [RESULT] [302744.940440] [metrics_monitor] gm 57875 power violations 0
    [RESULT] [302744.940440] [metrics_monitor] gm 57875 power average 139992336Watts
    [RESULT] [302745.228794] [metrics_monitor] gm 57875 stopped

    +=====================================================================+
    |                 ROCm Validation Suite (RVS) Summary                 |
    +=====================================================================+
    |                           System Overview                           |
    +---------------------------------------------------------------------+
    | Operating System                 | Ubuntu 22.04.5 LTS               |
    | RVS version                      | 1.6.75                           |
    | ROCm version                     | 7.2.1-81                         |
    | amdgpu version                   | 6.16.13                          |
    | GPUs                             | 8                                |
    +---------------------------------------------------------------------+
    | AMD Instinct MI355X - 42583      | AMD Instinct MI355X - 27226      |
    | 0 - 2 - 0000:05:00.0             | 1 - 3 - 0000:15:00.0             |
    +---------------------------------------------------------------------+
    | AMD Instinct MI355X - 36479      | AMD Instinct MI355X - 17010      |
    | 2 - 4 - 0000:65:00.0             | 3 - 5 - 0000:75:00.0             |
    +---------------------------------------------------------------------+
    | AMD Instinct MI355X - 1590       | AMD Instinct MI355X - 51771      |
    | 4 - 6 - 0000:85:00.0             | 5 - 7 - 0000:95:00.0             |
    +---------------------------------------------------------------------+
    | AMD Instinct MI355X - 11806      | AMD Instinct MI355X - 57875      |
    | 6 - 8 - 0000:e5:00.0             | 7 - 9 - 0000:f5:00.0             |
    +=====================================================================+
    | Action Name                      | Module         | Result          |
    +=====================================================================+
    | metrics_monitor                  | GM             | PASS            |
    +---------------------------------------------------------------------+


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

Monitoring is performed by polling respective PCIe registers approximately every
1 second.

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

When monitoring starts, an informational message is logged:

    [<action name>] PCIe link speed and power monitoring started ...

When a PCIe link speed or power state change is detected, an informational
message is logged per GPU:

    [<action name>] <gpu id> PCIe link speed changed <state>
    [<action name>] <gpu id> PCIe power state changed <state>

When monitoring stops (triggered by a `monitor: false` action), a summary is
logged:

    [<stop action name>] PCIe monitoring ended after wait duration.
    [<stop action name>] GPU <id> PCIe speed change true/false

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

    [action_1] PCIe link speed and power monitoring started ...
    [action_1] 33367 PCIe link speed changed 8 GT/s
    [action_1] 33367 PCIe power state changed D0
    [action_1] 3254 PCIe link speed changed 8 GT/s
    [action_1] 3254 PCIe power state changed D0
    [action_1] 50599 PCIe link speed changed 8 GT/s
    [action_1] 50599 PCIe power state changed D0
    [action_2] [GPU:: 33367] GFLOPS 6478.066983
    [action_2] [GPU:: 33367] GFLOPS 5189.993529
    [action_2] [GPU:: 33367] GFLOPS 5189.993529
    ...
    [action_2] [GPU:: 33367] GFLOPS 5174.935520
    [action_2] [GPU:: 33367] GFLOPS 6478.066983 Target GFLOPS: 5000.000000 met: TRUE
    [action_3] PCIe monitoring ended after wait duration.
    [action_3] GPU 33367 PCIe speed change true
    [action_3] GPU 3254 PCIe speed change false
    [action_3] GPU 50599 PCIe speed change true

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

    RVS-ERROR [PESM] [act1] invalid 'deviceid' key value: xxx


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
Two types of actions are performed by RCQT, selected by which configuration
key is present — not by the action name:

1) **Metapackage Check** — triggered when the `package` key is present.
Checks the installation of the named metapackages and their dependencies,
verifying the installed versions.

2) **Packages installation check** — triggered when `rpmpackagelist` or
`debpackagelist` is present. Checks whether the listed packages are installed.

Note: the `device` key is ignored by RCQT; checks are platform-wide.

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

**Example:**

Run:

    ./rvs -c conf/rcqt_single.conf

Configuration (`conf/rcqt_single.conf`, first action):

    actions:
    - name: metapackage-validation
      device: all
      module: rcqt
      package: rocm rocm-developer-tools rocm-openmp rocm-opencl-sdk rocm-hip

Sample output (first action, abridged):

    [RESULT] [302695.777282] Action name :metapackage-validation
    [RESULT] [302695.777442] Module name :rcqt
    Meta package rocm :
    json log file is /var/tmp/rvs_1784325458865.json
    Package half installed version is 1.12.0.70201
    Package migraphx installed version is 2.15.0.70201
    Package migraphx-dev installed version is 2.15.0.70201
    Package miopen-hip installed version is 3.5.1.70201
    Package miopen-hip-dev installed version is 3.5.1.70201
    Package mivisionx installed version is 3.5.0.70201
    Package mivisionx-dev installed version is 3.5.0.70201
    Package rocm-cmake installed version is 0.14.0.70201
    Package rocm-core installed version is 7.2.1.70201
    Package rocm-developer-tools installed version is 7.2.1.70201
    Package rocm-hip installed version is 7.2.1.70201
    Package rocm-llvm installed version is 22.0.0.26084.70201
    Package rocm-opencl-sdk installed version is 7.2.1.70201
    Package rocm-openmp installed version is 7.2.1.70201
    Package rocminfo installed version is 1.0.0.70201
    Package rpp installed version is 2.2.1.70201
    Package rpp-dev installed version is 2.2.1.70201
    Meta package validation complete :
        Total packages validated     : 17
        Installed packages           : 17
        Missing packages             : 0
        Version mismatch packages    : 0
    ...
    Package rocm-smi-lib installed version is 7.8.0.70201
    Package rocminfo installed version is 1.0.0.70201
    Package rocprim-dev installed version is 4.2.0.70201
    Package rocrand installed version is 4.2.0.70201
    Package rocrand-dev installed version is 4.2.0.70201
    Package rocsolver installed version is 3.32.0.70201
    Package rocsolver-dev installed version is 3.32.0.70201
    Package rocsparse installed version is 4.2.0.70201
    Package rocsparse-dev installed version is 4.2.0.70201
    Package rocthrust-dev installed version is 4.2.0.70201
    Package rocwmma-dev installed version is 2.2.0.70201
    Meta package validation complete :
        Total packages validated     : 52
        Installed packages           : 52
        Missing packages             : 0
        Version mismatch packages    : 0

    +=====================================================================+
    |                 ROCm Validation Suite (RVS) Summary                 |
    +=====================================================================+
    |                           System Overview                           |
    +---------------------------------------------------------------------+
    | Operating System                 | Ubuntu 22.04.5 LTS               |
    | RVS version                      | 1.6.75                           |
    | ROCm version                     | 7.2.1-81                         |
    | amdgpu version                   | 6.16.13                          |
    | GPUs                             | 8                                |
    +---------------------------------------------------------------------+
    | AMD Instinct MI355X - 42583      | AMD Instinct MI355X - 27226      |
    | 0 - 2 - 0000:05:00.0             | 1 - 3 - 0000:15:00.0             |
    +---------------------------------------------------------------------+
    | AMD Instinct MI355X - 36479      | AMD Instinct MI355X - 17010      |
    | 2 - 4 - 0000:65:00.0             | 3 - 5 - 0000:75:00.0             |
    +---------------------------------------------------------------------+
    | AMD Instinct MI355X - 1590       | AMD Instinct MI355X - 51771      |
    | 4 - 6 - 0000:85:00.0             | 5 - 7 - 0000:95:00.0             |
    +---------------------------------------------------------------------+
    | AMD Instinct MI355X - 11806      | AMD Instinct MI355X - 57875      |
    | 6 - 8 - 0000:e5:00.0             | 7 - 9 - 0000:f5:00.0             |
    +=====================================================================+
    | Action Name                      | Module         | Result          |
    +=====================================================================+
    | metapackage-validation           | RCQT           | PASS            |
    +---------------------------------------------------------------------+

For other cases, mismatched or missing packages are printed with respective counts.

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
<tr><td>version</td><td>String</td>
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

**Example:**

Run:

    ./rvs -c conf/peqt_single.conf

Configuration (`conf/peqt_single.conf`, first action):

    actions:
    - name: pcie_act_1
      device: all
      module: peqt
      capability:
        link_cap_max_speed:
        link_cap_max_width:
        link_stat_cur_speed:
        link_stat_neg_width:
        slot_pwr_limit_value:
        slot_physical_num:
        deviceid:
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

```{note}
- When setting the `device` configuration key to `all`, the RVS will detect all the AMD compatible GPUs and run the test on all of them.
- With no regular expressions specified, RVS reports `TRUE` if at least one AMD compatible GPU is registered within the system. Otherwise it reports `FALSE`.
```

Sample output (first action, abridged):

    [RESULT] [302695.130515] Action name :pcie_act_1
    [RESULT] [302695.251177] Module name :peqt
    [RESULT] [302695.289685] [pcie_act_1] peqt true

    +=====================================================================+
    |                 ROCm Validation Suite (RVS) Summary                 |
    +=====================================================================+
    |                           System Overview                           |
    +---------------------------------------------------------------------+
    | Operating System                 | Ubuntu 22.04.5 LTS               |
    | RVS version                      | 1.6.75                           |
    | ROCm version                     | 7.2.1-81                         |
    | amdgpu version                   | 6.16.13                          |
    | GPUs                             | 8                                |
    +---------------------------------------------------------------------+
    | AMD Instinct MI355X - 42583      | AMD Instinct MI355X - 27226      |
    | 0 - 2 - 0000:05:00.0             | 1 - 3 - 0000:15:00.0             |
    +---------------------------------------------------------------------+
    | AMD Instinct MI355X - 36479      | AMD Instinct MI355X - 17010      |
    | 2 - 4 - 0000:65:00.0             | 3 - 5 - 0000:75:00.0             |
    +---------------------------------------------------------------------+
    | AMD Instinct MI355X - 1590       | AMD Instinct MI355X - 51771      |
    | 4 - 6 - 0000:85:00.0             | 5 - 7 - 0000:95:00.0             |
    +---------------------------------------------------------------------+
    | AMD Instinct MI355X - 11806      | AMD Instinct MI355X - 57875      |
    | 6 - 8 - 0000:e5:00.0             | 7 - 9 - 0000:f5:00.0             |
    +=====================================================================+
    | Action Name                      | Module         | Result          |
    +=====================================================================+
    | pcie_act_1                       | PEQT           | PASS            |
    +---------------------------------------------------------------------+

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
<tr><td>bar5_size</td><td>Integer</td><td>The actual size of BAR5. Note: due to a
source bug, the JSON output for <b>bar5_size</b> is populated from the BAR4
size instead of BAR5; the text log value is correct.</td></tr>
<tr><td>pass</td><td>Bool</td> <td>'true' if all of the BAR properties satisfy
the constraints, 'false' otherwise.</td></tr>
</table>
</div>

The qualification check will query the specified bar properties and check that
they satisfy the give parameters. The pass output key will be true and the test
will pass if all of the BAR properties satisfy the constraints. After the check
is finished, the following informational messages will be generated:

    [INFO  ][<timestamp>][<action name>] smqt <gpu_id> bar1_size <bar1_size>
    [INFO  ][<timestamp>][<action name>] smqt <gpu_id> bar1_base_addr <bar1_base_addr>
    [INFO  ][<timestamp>][<action name>] smqt <gpu_id> bar2_size <bar2_size>
    [INFO  ][<timestamp>][<action name>] smqt <gpu_id> bar2_base_addr <bar2_base_addr>
    [INFO  ][<timestamp>][<action name>] smqt <gpu_id> bar4_size <bar4_size>
    [INFO  ][<timestamp>][<action name>] smqt <gpu_id> bar4_base_addr <bar4_base_addr>
    [INFO  ][<timestamp>][<action name>] smqt <gpu_id> bar5_size <bar5_size>
    [RESULT][<timestamp>][<action name>] smqt <gpu_id> pass: <true|false>


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

    [INFO  ] [257936.568768] [action_1]  smqt 3254 bar1_size      17179869184 (16.00 GB)
    [INFO  ] [257936.568768] [action_1]  smqt 3254 bar1_base_addr 13C0000000C
    [INFO  ] [257936.568768] [action_1]  smqt 3254 bar2_size      2097152 (2.00 MB)
    [INFO  ] [257936.568768] [action_1]  smqt 3254 bar2_base_addr 13B0000000C
    [INFO  ] [257936.568768] [action_1]  smqt 3254 bar4_size      524288 (512.00 KB)
    [INFO  ] [257936.568768] [action_1]  smqt 3254 bar4_base_addr E4B00000
    [INFO  ] [257936.568768] [action_1]  smqt 3254 bar5_size      0 (0.00 B)
    [RESULT] [257936.568920] [action_1]  smqt 3254 pass: false
    [INFO  ] [257936.569234] [action_1]  smqt 50599 bar1_size      17179869184 (16.00 GB)
    [INFO  ] [257936.569234] [action_1]  smqt 50599 bar1_base_addr 1A00000000C
    [INFO  ] [257936.569234] [action_1]  smqt 50599 bar2_size      2097152 (2.00 MB)
    [INFO  ] [257936.569234] [action_1]  smqt 50599 bar2_base_addr 19F0000000C
    [INFO  ] [257936.569234] [action_1]  smqt 50599 bar4_size      524288 (512.00 KB)
    [INFO  ] [257936.569234] [action_1]  smqt 50599 bar4_base_addr E9900000
    [INFO  ] [257936.569234] [action_1]  smqt 50599 bar5_size      0 (0.00 B)
    [RESULT] [257936.569281] [action_1]  smqt 50599 pass: false
    [INFO  ] [257936.570798] [action_1]  smqt 33367 bar1_size      17179869184 (16.00 GB)
    [INFO  ] [257936.570798] [action_1]  smqt 33367 bar1_base_addr 16C0000000C
    [INFO  ] [257936.570798] [action_1]  smqt 33367 bar2_size      2097152 (2.00 MB)
    [INFO  ] [257936.570798] [action_1]  smqt 33367 bar2_base_addr 1710000000C
    [INFO  ] [257936.570798] [action_1]  smqt 33367 bar4_size      524288 (512.00 KB)
    [INFO  ] [257936.570798] [action_1]  smqt 33367 bar4_base_addr E7300000
    [INFO  ] [257936.570798] [action_1]  smqt 33367 bar5_size      0 (0.00 B)
    [RESULT] [257936.570837] [action_1]  smqt 33367 pass: false

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
devices pass the P2P check. Note: setting this to false currently causes the
module to exit with an error in the current implementation; bandwidth testing
must be enabled for a successful run.</td></tr>
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

The P2P capability check logs a result message for each device-peer pair.
Self-pairs (same GPU) are not logged. The message format is:

    [<action name>] p2p [GPU:: <src_node> - <src_id> - <src_bdf>] [GPU:: <dst_node> - <dst_id> - <dst_bdf>] peers:<p2p_result> distance:<distance> <hop_type>:<hop_dist>

If `test_bandwidth` is true, bandwidth testing between the device and each of
its peers will take place in parallel or in sequence, depending on the value of
the `parallel` flag. During bandwidth benchmarking, informational output
providing the moving average of the transfer's bandwidth is logged at every
`log_interval`:

    [<action name>] p2p-bandwidth[<transfer_id>] [GPU:: <src_node> - <src_id> - <src_bdf>] [GPU:: <dst_node> - <dst_id> - <dst_bdf>] bidirectional: <bidirectional> <interval_bandwidth> GBps

At the end of the test, the average bandwidth over the entire test duration is
logged as a result:

    [<action name>] p2p-bandwidth[<transfer_id>] [GPU:: <src_node> - <src_id> - <src_bdf>] [GPU:: <dst_node> - <dst_id> - <dst_bdf>] bidirectional: <bidirectional> <bandwidth> GBps duration: <duration> secs

When `transferbench_test: alltoall` is used, additional aggregate output lines
are emitted:

    [<action name>] a2a-p2p-bandwidth[<transfer_id>] ...
    [<action name>] a2a-gpu-bandwidth ... Aggregate peer bandwidth: <value> GBps
    [<action name>] a2a-bandwidth [<N> GPUs][<M> p2p transfers] Aggregate bandwidth: <value> GBps

### Examples

**Example:**

Run:

    ./rvs -c conf/MI355X/pbqt_single.conf

Configuration (`conf/MI355X/pbqt_single.conf`, first action):

    actions:
    - name: xgmi_d2d_unidir_bandwidth
      device: all
      module: pbqt
      log_interval: 5000
      duration: 30000
      peers: all
      test_bandwidth: true
      bidirectional: false
      parallel: false
      block_size: 1073741824
      device_id: all

Sample output (first action, abridged):

    [RESULT] [302380.159634] Action name :xgmi_d2d_unidir_bandwidth
    [RESULT] [302380.356991] Module name :pbqt
    [RESULT] [302380.357340] [xgmi_d2d_unidir_bandwidth] p2p [GPU:: 2 - 42583 - 0000:05:00.0] [GPU:: 3 - 27226 - 0000:15:00.0] peers:true distance:15 xGMI:15
    [RESULT] [302380.357342] [xgmi_d2d_unidir_bandwidth] p2p [GPU:: 2 - 42583 - 0000:05:00.0] [GPU:: 4 - 36479 - 0000:65:00.0] peers:true distance:15 xGMI:15
    [RESULT] [302380.357344] [xgmi_d2d_unidir_bandwidth] p2p [GPU:: 2 - 42583 - 0000:05:00.0] [GPU:: 5 - 17010 - 0000:75:00.0] peers:true distance:15 xGMI:15
    [RESULT] [302380.357345] [xgmi_d2d_unidir_bandwidth] p2p [GPU:: 2 - 42583 - 0000:05:00.0] [GPU:: 6 -  1590 - 0000:85:00.0] peers:true distance:15 xGMI:15
    [RESULT] [302380.357346] [xgmi_d2d_unidir_bandwidth] p2p [GPU:: 2 - 42583 - 0000:05:00.0] [GPU:: 7 - 51771 - 0000:95:00.0] peers:true distance:15 xGMI:15
    [RESULT] [302380.357347] [xgmi_d2d_unidir_bandwidth] p2p [GPU:: 2 - 42583 - 0000:05:00.0] [GPU:: 8 - 11806 - 0000:e5:00.0] peers:true distance:15 xGMI:15
    [RESULT] [302380.357348] [xgmi_d2d_unidir_bandwidth] p2p [GPU:: 2 - 42583 - 0000:05:00.0] [GPU:: 9 - 57875 - 0000:f5:00.0] peers:true distance:15 xGMI:15
    ...
    [RESULT] [302410.443673] [xgmi_d2d_unidir_bandwidth] p2p-bandwidth[48/56] [GPU:: 8 - 11806 - 0000:e5:00.0] [GPU:: 7 - 51771 - 0000:95:00.0] bidirectional: false 61.370 GBps duration: 0.227452 secs
    [RESULT] [302410.444726] [xgmi_d2d_unidir_bandwidth] p2p-bandwidth[49/56] [GPU:: 8 - 11806 - 0000:e5:00.0] [GPU:: 9 - 57875 - 0000:f5:00.0] bidirectional: false 61.369 GBps duration: 0.227453 secs
    [RESULT] [302410.445779] [xgmi_d2d_unidir_bandwidth] p2p-bandwidth[50/56] [GPU:: 9 - 57875 - 0000:f5:00.0] [GPU:: 2 - 42583 - 0000:05:00.0] bidirectional: false 61.368 GBps duration: 0.227456 secs
    [RESULT] [302410.446832] [xgmi_d2d_unidir_bandwidth] p2p-bandwidth[51/56] [GPU:: 9 - 57875 - 0000:f5:00.0] [GPU:: 3 - 27226 - 0000:15:00.0] bidirectional: false 61.373 GBps duration: 0.227441 secs
    [RESULT] [302410.447885] [xgmi_d2d_unidir_bandwidth] p2p-bandwidth[52/56] [GPU:: 9 - 57875 - 0000:f5:00.0] [GPU:: 4 - 36479 - 0000:65:00.0] bidirectional: false 59.704 GBps duration: 0.233798 secs
    [RESULT] [302410.448939] [xgmi_d2d_unidir_bandwidth] p2p-bandwidth[53/56] [GPU:: 9 - 57875 - 0000:f5:00.0] [GPU:: 5 - 17010 - 0000:75:00.0] bidirectional: false 61.371 GBps duration: 0.227448 secs
    [RESULT] [302410.449991] [xgmi_d2d_unidir_bandwidth] p2p-bandwidth[54/56] [GPU:: 9 - 57875 - 0000:f5:00.0] [GPU:: 6 -  1590 - 0000:85:00.0] bidirectional: false 61.367 GBps duration: 0.227463 secs
    [RESULT] [302410.451044] [xgmi_d2d_unidir_bandwidth] p2p-bandwidth[55/56] [GPU:: 9 - 57875 - 0000:f5:00.0] [GPU:: 7 - 51771 - 0000:95:00.0] bidirectional: false 61.378 GBps duration: 0.227422 secs
    [RESULT] [302410.452097] [xgmi_d2d_unidir_bandwidth] p2p-bandwidth[56/56] [GPU:: 9 - 57875 - 0000:f5:00.0] [GPU:: 8 - 11806 - 0000:e5:00.0] bidirectional: false 61.368 GBps duration: 0.227458 secs

    +=====================================================================+
    |                 ROCm Validation Suite (RVS) Summary                 |
    +=====================================================================+
    |                           System Overview                           |
    +---------------------------------------------------------------------+
    | Operating System                 | Ubuntu 22.04.5 LTS               |
    | RVS version                      | 1.6.75                           |
    | ROCm version                     | 7.2.1-81                         |
    | amdgpu version                   | 6.16.13                          |
    | GPUs                             | 8                                |
    +---------------------------------------------------------------------+
    | AMD Instinct MI355X - 42583      | AMD Instinct MI355X - 27226      |
    | 0 - 2 - 0000:05:00.0             | 1 - 3 - 0000:15:00.0             |
    +---------------------------------------------------------------------+
    | AMD Instinct MI355X - 36479      | AMD Instinct MI355X - 17010      |
    | 2 - 4 - 0000:65:00.0             | 3 - 5 - 0000:75:00.0             |
    +---------------------------------------------------------------------+
    | AMD Instinct MI355X - 1590       | AMD Instinct MI355X - 51771      |
    | 4 - 6 - 0000:85:00.0             | 5 - 7 - 0000:95:00.0             |
    +---------------------------------------------------------------------+
    | AMD Instinct MI355X - 11806      | AMD Instinct MI355X - 57875      |
    | 6 - 8 - 0000:e5:00.0             | 7 - 9 - 0000:f5:00.0             |
    +=====================================================================+
    | Action Name                      | Module         | Result          |
    +=====================================================================+
    | xgmi_d2d_unidir_bandwidth        | PBQT           | PASS            |
    +---------------------------------------------------------------------+

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
<td>Controls whether transfers run in parallel across all CPU–GPU paths or one
by one.\n
- true – Run all test transfers in parallel.\n
- false – Run test transfers one by one.

</td></tr>
<tr><td>duration</td><td>Integer</td>
<td>This key specifies the duration a transfer test should run, given in
milliseconds. If this key is not specified, the default value is 10000 (10
seconds).
</td></tr>
<tr><td>log_interval</td><td>Integer</td>
<td>This is a positive integer, given in milliseconds, that specifies an interval
over which the moving average of the bandwidth will be calculated and logged. The
default value is 1000 (1 second). It must be smaller than the duration key.\n
If this key is 0 (zero), results are displayed as soon as the test transfer
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

    [<action name>] pcie-bandwidth [CPU:: <cpu_node>] [GPU:: <gpu_node> - <gpu_id> - <bdf>] distance:<distance> <hop_type>:<hop_dist>

During the execution of the benchmark, informational output providing the moving
average of the bandwidth of the transfer will be calculated and logged at every
`log_interval`:

    [<action name>] pcie-bandwidth [<transfer_id>] [CPU:: <cpu_node>] [GPU:: <gpu_node> - <gpu_id> - <bdf>] h2d::<host_to_device> d2h::<device_to_host> <interval_bandwidth> GBps

At the end of test, the average bandwidth over the entire test duration is
logged as a result:

    [<action name>] pcie-bandwidth [<transfer_id>] [CPU:: <cpu_node>] [GPU:: <gpu_node> - <gpu_id> - <bdf>] h2d::<host_to_device> d2h::<device_to_host> <bandwidth> GBps duration: <duration> secs



### Examples

**Example:**

Run:

    ./rvs -c conf/MI355X/pebb_single.conf

Configuration (`conf/MI355X/pebb_single.conf`, first action):

    actions:
    - name: pcie_h2d_bandwidth
      device: all
      module: pebb
      duration: 30000
      device_to_host: false
      host_to_device: true
      parallel: false
      block_size: 1073741824
      link_type: 2

Sample output (first action, abridged):

    [RESULT] [302349.492047] Action name :pcie_h2d_bandwidth
    [RESULT] [302349.671479] Module name :pebb
    [RESULT] [302349.671823] [pcie_h2d_bandwidth] pcie-bandwidth [CPU:: 0] [GPU:: 2 - 42583 - 0000:05:00.0] distance:20 PCIe:20
    [RESULT] [302349.671837] [pcie_h2d_bandwidth] pcie-bandwidth [CPU:: 1] [GPU:: 2 - 42583 - 0000:05:00.0] distance:52 PCIe:52
    [RESULT] [302349.671839] [pcie_h2d_bandwidth] pcie-bandwidth [CPU:: 0] [GPU:: 3 - 27226 - 0000:15:00.0] distance:20 PCIe:20
    [RESULT] [302349.671840] [pcie_h2d_bandwidth] pcie-bandwidth [CPU:: 1] [GPU:: 3 - 27226 - 0000:15:00.0] distance:52 PCIe:52
    [RESULT] [302349.671841] [pcie_h2d_bandwidth] pcie-bandwidth [CPU:: 0] [GPU:: 4 - 36479 - 0000:65:00.0] distance:20 PCIe:20
    [RESULT] [302349.671842] [pcie_h2d_bandwidth] pcie-bandwidth [CPU:: 1] [GPU:: 4 - 36479 - 0000:65:00.0] distance:52 PCIe:52
    [RESULT] [302349.671843] [pcie_h2d_bandwidth] pcie-bandwidth [CPU:: 0] [GPU:: 5 - 17010 - 0000:75:00.0] distance:20 PCIe:20
    [RESULT] [302349.671844] [pcie_h2d_bandwidth] pcie-bandwidth [CPU:: 1] [GPU:: 5 - 17010 - 0000:75:00.0] distance:52 PCIe:52
    [RESULT] [302349.671846] [pcie_h2d_bandwidth] pcie-bandwidth [CPU:: 0] [GPU:: 6 -  1590 - 0000:85:00.0] distance:52 PCIe:52
    [RESULT] [302349.671847] [pcie_h2d_bandwidth] pcie-bandwidth [CPU:: 1] [GPU:: 6 -  1590 - 0000:85:00.0] distance:20 PCIe:20
    [RESULT] [302349.671848] [pcie_h2d_bandwidth] pcie-bandwidth [CPU:: 0] [GPU:: 7 - 51771 - 0000:95:00.0] distance:52 PCIe:52
    [RESULT] [302349.671849] [pcie_h2d_bandwidth] pcie-bandwidth [CPU:: 1] [GPU:: 7 - 51771 - 0000:95:00.0] distance:20 PCIe:20
    [RESULT] [302349.671850] [pcie_h2d_bandwidth] pcie-bandwidth [CPU:: 0] [GPU:: 8 - 11806 - 0000:e5:00.0] distance:52 PCIe:52
    [RESULT] [302349.671851] [pcie_h2d_bandwidth] pcie-bandwidth [CPU:: 1] [GPU:: 8 - 11806 - 0000:e5:00.0] distance:20 PCIe:20
    [RESULT] [302349.671852] [pcie_h2d_bandwidth] pcie-bandwidth [CPU:: 0] [GPU:: 9 - 57875 - 0000:f5:00.0] distance:52 PCIe:52
    [RESULT] [302349.671853] [pcie_h2d_bandwidth] pcie-bandwidth [CPU:: 1] [GPU:: 9 - 57875 - 0000:f5:00.0] distance:20 PCIe:20
    [RESULT] [302379.847423] [pcie_h2d_bandwidth] pcie-bandwidth [ 1/16] [CPU:: 0] [GPU:: 2 - 42583 - 0000:05:00.0] h2d::true d2h::false 57.678 GBps duration: 0.223392 secs
    [RESULT] [302379.847438] [pcie_h2d_bandwidth] pcie-bandwidth [ 2/16] [CPU:: 1] [GPU:: 2 - 42583 - 0000:05:00.0] h2d::true d2h::false 54.034 GBps duration: 0.238461 secs
    ...
    [RESULT] [302379.847449] [pcie_h2d_bandwidth] pcie-bandwidth [ 9/16] [CPU:: 0] [GPU:: 6 -  1590 - 0000:85:00.0] h2d::true d2h::false 57.708 GBps duration: 0.204669 secs
    [RESULT] [302379.847451] [pcie_h2d_bandwidth] pcie-bandwidth [10/16] [CPU:: 1] [GPU:: 6 -  1590 - 0000:85:00.0] h2d::true d2h::false 55.223 GBps duration: 0.213882 secs
    [RESULT] [302379.847453] [pcie_h2d_bandwidth] pcie-bandwidth [11/16] [CPU:: 0] [GPU:: 7 - 51771 - 0000:95:00.0] h2d::true d2h::false 57.708 GBps duration: 0.204671 secs
    [RESULT] [302379.847454] [pcie_h2d_bandwidth] pcie-bandwidth [12/16] [CPU:: 1] [GPU:: 7 - 51771 - 0000:95:00.0] h2d::true d2h::false 57.708 GBps duration: 0.204671 secs
    [RESULT] [302379.847472] [pcie_h2d_bandwidth] pcie-bandwidth [13/16] [CPU:: 0] [GPU:: 8 - 11806 - 0000:e5:00.0] h2d::true d2h::false 57.708 GBps duration: 0.204672 secs
    [RESULT] [302379.847474] [pcie_h2d_bandwidth] pcie-bandwidth [14/16] [CPU:: 1] [GPU:: 8 - 11806 - 0000:e5:00.0] h2d::true d2h::false 57.708 GBps duration: 0.204670 secs
    [RESULT] [302379.847475] [pcie_h2d_bandwidth] pcie-bandwidth [15/16] [CPU:: 0] [GPU:: 9 - 57875 - 0000:f5:00.0] h2d::true d2h::false 57.709 GBps duration: 0.204668 secs
    [RESULT] [302379.847476] [pcie_h2d_bandwidth] pcie-bandwidth [16/16] [CPU:: 1] [GPU:: 9 - 57875 - 0000:f5:00.0] h2d::true d2h::false 57.710 GBps duration: 0.204666 secs

    +=====================================================================+
    |                 ROCm Validation Suite (RVS) Summary                 |
    +=====================================================================+
    |                           System Overview                           |
    +---------------------------------------------------------------------+
    | Operating System                 | Ubuntu 22.04.5 LTS               |
    | RVS version                      | 1.6.75                           |
    | ROCm version                     | 7.2.1-81                         |
    | amdgpu version                   | 6.16.13                          |
    | GPUs                             | 8                                |
    +---------------------------------------------------------------------+
    | AMD Instinct MI355X - 42583      | AMD Instinct MI355X - 27226      |
    | 0 - 2 - 0000:05:00.0             | 1 - 3 - 0000:15:00.0             |
    +---------------------------------------------------------------------+
    | AMD Instinct MI355X - 36479      | AMD Instinct MI355X - 17010      |
    | 2 - 4 - 0000:65:00.0             | 3 - 5 - 0000:75:00.0             |
    +---------------------------------------------------------------------+
    | AMD Instinct MI355X - 1590       | AMD Instinct MI355X - 51771      |
    | 4 - 6 - 0000:85:00.0             | 5 - 7 - 0000:95:00.0             |
    +---------------------------------------------------------------------+
    | AMD Instinct MI355X - 11806      | AMD Instinct MI355X - 57875      |
    | 6 - 8 - 0000:e5:00.0             | 7 - 9 - 0000:f5:00.0             |
    +=====================================================================+
    | Action Name                      | Module         | Result          |
    +=====================================================================+
    | pcie_h2d_bandwidth               | PEBB           | PASS            |
    +---------------------------------------------------------------------+

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
for the test to still pass. The default value is 0. Note: this key is parsed
but violation counting is not active in the current implementation; pass/fail
is determined solely by whether the peak GFLOPS meets the target.</td></tr>
<tr><td>log_interval</td><td>Integer</td>
<td>This is a positive integer, given in milliseconds, that specifies an
interval over which the moving average of GFLOPS will be calculated and
logged.</td></tr>
<tr><td>matrix_size_a</td><td>Integer</td>
<td>Number of rows of matrix A (the M dimension in GEMM). The default value is 5760.</td></tr>
<tr><td>matrix_size_b</td><td>Integer</td>
<td>Number of columns of matrix B (the N dimension in GEMM). The default value is 5760.</td></tr>
<tr><td>matrix_size_c</td><td>Integer</td>
<td>Inner (shared) dimension K of the GEMM operation (columns of A / rows of B).
The default value is 5760.</td></tr>

<tr><td>ops_type</td><td>String</td>
<td>GEMM operation type. Accepted values: <b>sgemm</b>, <b>dgemm</b>,
<b>hgemm</b>. If neither <b>ops_type</b> nor <b>data_type</b> is set, the
module defaults to <b>sgemm</b>. Mutually exclusive with <b>data_type</b>.</td></tr>

<tr><td>data_type</td><td>String</td>
<td>Data type for the GEMM computation. Accepted values include:
<b>fp4_r</b>, <b>fp6_r</b>, <b>fp8_r</b>, <b>bf16_r</b>, <b>fp16_r</b>,
<b>fp32_r</b>, <b>i8_r</b>. If not specified
the module falls back to the type implied by <b>ops_type</b>.</td></tr>

<tr><td>out_data_type</td><td>String</td>
<td>Output (result) data type, e.g. <b>fp16_r</b>, <b>fp32_r</b>. Only
applicable when using hipBLASLt. If not specified the default matches the
compute type.</td></tr>

<tr><td>compute_type</td><td>String</td>
<td>Accumulation/compute type used internally by the BLAS library. Accepted
values include <b>fp32_r</b> (default) and <b>xf32_r</b> (TF32 fast
compute).</td></tr>

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
<b>rand</b> – Initialize with random values.</td></tr>

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

During the execution of the test, a result message reporting the GFLOPS
achieved by the GPU is logged at each `log_interval`:

    [<action name>] [GPU:: <gpu id>] GFLOPS <interval_gflops>

When the test completes, the final result message is printed:

    [<action name>] [GPU:: <gpu id>] GFLOPS <max_gflops> Target GFLOPS: <target_stress> met: TRUE

The test passes if `max_gflops >= target_stress * (1 - tolerance)` at the end
of the run. Otherwise the result is `met: FALSE`.

### Examples

**Example:**

Run:

    ./rvs -c conf/MI355X/gst_single.conf

Configuration (`conf/MI355X/gst_single.conf`, first action):

    actions:
    - name: gst-Tflops-2K2K2K-trig-fp4
      device: all
      module: gst
      log_interval: 3000
      ramp_interval: 5000
      duration: 15000
      hot_calls: 1000
      copy_matrix: false
      target_stress: 0
      matrix_size_a: 2048
      matrix_size_b: 2048
      matrix_size_c: 2048
      scale_a: block
      scale_b: block
      matrix_init: trig
      data_type: fp4_r
      out_data_type: fp16_r
      compute_type: fp32_r
      transa: 1
      transb: 0
      alpha: 1.5
      beta: 2
      blas_source: hipblaslt

Sample output (first action, abridged):

    [RESULT] [301995.91241 ] Action name :gst-Tflops-2K2K2K-trig-fp4
    [RESULT] [301995.175394] Module name :gst
    [RESULT] [301995.794012] [gst-Tflops-2K2K2K-trig-fp4] [GPU:: 42583] Start of GPU ramp up
    [RESULT] [302001.161732] [gst-Tflops-2K2K2K-trig-fp4] [GPU:: 42583] GFLOPS 536870
    [RESULT] [302002.161726] [gst-Tflops-2K2K2K-trig-fp4] [GPU:: 42583] End of GPU ramp up
    [RESULT] [302005.176647] [gst-Tflops-2K2K2K-trig-fp4] [GPU:: 42583] GFLOPS 1059931
    [RESULT] [302008.187367] [gst-Tflops-2K2K2K-trig-fp4] [GPU:: 42583] GFLOPS 1055687
    [RESULT] [302011.194700] [gst-Tflops-2K2K2K-trig-fp4] [GPU:: 42583] GFLOPS 1056873
    [RESULT] [302014.199261] [gst-Tflops-2K2K2K-trig-fp4] [GPU:: 42583] GFLOPS 1063560
    [RESULT] [302017.167195] [gst-Tflops-2K2K2K-trig-fp4] [GPU:: 42583] GFLOPS 1063560 Target GFLOPS: 0 met: TRUE
    [RESULT] [302017.168168] [gst-Tflops-2K2K2K-trig-fp4] [GPU:: 27226] Start of GPU ramp up
    [RESULT] [302022.421856] [gst-Tflops-2K2K2K-trig-fp4] [GPU:: 27226] GFLOPS 536870
    [RESULT] [302023.421863] [gst-Tflops-2K2K2K-trig-fp4] [GPU:: 27226] End of GPU ramp up
    [RESULT] [302026.422111] [gst-Tflops-2K2K2K-trig-fp4] [GPU:: 27226] GFLOPS 1065100
    [RESULT] [302029.437823] [gst-Tflops-2K2K2K-trig-fp4] [GPU:: 27226] GFLOPS 1065326
    [RESULT] [302032.447003] [gst-Tflops-2K2K2K-trig-fp4] [GPU:: 27226] GFLOPS 1067645
    [RESULT] [302035.454318] [gst-Tflops-2K2K2K-trig-fp4] [GPU:: 27226] GFLOPS 1068301
    [RESULT] [302038.430577] [gst-Tflops-2K2K2K-trig-fp4] [GPU:: 27226] GFLOPS 1068301 Target GFLOPS: 0 met: TRUE
    [RESULT] [302038.431686] [gst-Tflops-2K2K2K-trig-fp4] [GPU:: 36479] Start of GPU ramp up
    [RESULT] [302043.682076] [gst-Tflops-2K2K2K-trig-fp4] [GPU:: 36479] GFLOPS 554189
    ...
    [RESULT] [302141.857878] [gst-Tflops-2K2K2K-trig-fp4] [GPU:: 11806] GFLOPS 1072748
    [RESULT] [302144.843455] [gst-Tflops-2K2K2K-trig-fp4] [GPU:: 11806] GFLOPS 1073132 Target GFLOPS: 0 met: TRUE
    [RESULT] [302144.844774] [gst-Tflops-2K2K2K-trig-fp4] [GPU:: 57875] Start of GPU ramp up
    [RESULT] [302150.131420] [gst-Tflops-2K2K2K-trig-fp4] [GPU:: 57875] GFLOPS 554189
    [RESULT] [302151.131405] [gst-Tflops-2K2K2K-trig-fp4] [GPU:: 57875] End of GPU ramp up
    [RESULT] [302154.146042] [gst-Tflops-2K2K2K-trig-fp4] [GPU:: 57875] GFLOPS 1060026
    [RESULT] [302157.154100] [gst-Tflops-2K2K2K-trig-fp4] [GPU:: 57875] GFLOPS 1056631
    [RESULT] [302160.164765] [gst-Tflops-2K2K2K-trig-fp4] [GPU:: 57875] GFLOPS 1061420
    [RESULT] [302163.173056] [gst-Tflops-2K2K2K-trig-fp4] [GPU:: 57875] GFLOPS 1056547
    [RESULT] [302166.136604] [gst-Tflops-2K2K2K-trig-fp4] [GPU:: 57875] GFLOPS 1061420 Target GFLOPS: 0 met: TRUE

    +=====================================================================+
    |                 ROCm Validation Suite (RVS) Summary                 |
    +=====================================================================+
    |                           System Overview                           |
    +---------------------------------------------------------------------+
    | Operating System                 | Ubuntu 22.04.5 LTS               |
    | RVS version                      | 1.6.75                           |
    | ROCm version                     | 7.2.1-81                         |
    | amdgpu version                   | 6.16.13                          |
    | GPUs                             | 8                                |
    +---------------------------------------------------------------------+
    | AMD Instinct MI355X - 42583      | AMD Instinct MI355X - 27226      |
    | 0 - 2 - 0000:05:00.0             | 1 - 3 - 0000:15:00.0             |
    +---------------------------------------------------------------------+
    | AMD Instinct MI355X - 36479      | AMD Instinct MI355X - 17010      |
    | 2 - 4 - 0000:65:00.0             | 3 - 5 - 0000:75:00.0             |
    +---------------------------------------------------------------------+
    | AMD Instinct MI355X - 1590       | AMD Instinct MI355X - 51771      |
    | 4 - 6 - 0000:85:00.0             | 5 - 7 - 0000:95:00.0             |
    +---------------------------------------------------------------------+
    | AMD Instinct MI355X - 11806      | AMD Instinct MI355X - 57875      |
    | 6 - 8 - 0000:e5:00.0             | 7 - 9 - 0000:f5:00.0             |
    +=====================================================================+
    | Action Name                      | Module         | Result          |
    +=====================================================================+
    | gst-Tflops-2K2K2K-trig-fp4       | GST            | PASS            |
    +---------------------------------------------------------------------+

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
is 5000 (5 seconds). Note: this key is parsed but not used by the worker in the
current implementation.
</td></tr>
<tr><td>tolerance</td><td>Float</td>
<td>A value indicating how much the target_power can fluctuate for the test to
succeed. The default value is 0 (any violation immediately fails the test).
</td></tr>
<tr><td>max_violations</td><td>Integer</td>
<td>The number of tolerance violations that can occur for the test to still
pass. The default value is 0. Note: this key is parsed but not used by the
worker in the current implementation.</td></tr>
<tr><td>sample_interval</td><td>Integer</td>
<td>The interval between power samples, specified in seconds. The default
value is 1 (1 second). If a value less than 1 is specified it is raised to 1.
</td></tr>
<tr><td>log_interval</td><td>Integer</td>
<td>This is a positive integer, given in milliseconds, that specifies a logging
interval. Note: this key is parsed but not used by the worker in the current
implementation.</td></tr>

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
<b>fp32_r</b>, <b>i8_r</b>. If not specified
the module falls back to the type implied by <b>ops_type</b>.</td></tr>

<tr><td>out_data_type</td><td>String</td>
<td>Output (result) data type, e.g. <b>fp16_r</b>, <b>fp32_r</b>. Only
applicable when using hipBLASLt. If not specified the default matches the
compute type.</td></tr>

<tr><td>compute_type</td><td>String</td>
<td>Accumulation/compute type used internally by the BLAS library. Accepted
values include <b>fp32_r</b> (default) and <b>xf32_r</b> (TF32 fast
compute).</td></tr>

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
<b>rand</b> – Initialize with random values.</td></tr>

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

During the execution of the test, a result message reporting the measured GPU
power is logged at each `sample_interval`:

    [<action name>] [GPU:: <gpu id>] Power(W) <current_power>

When the test completes, the pass/fail result is printed:

    [<action name>] [GPU:: <gpu id>] pass: TRUE

The test passes if the peak power measured during the run meets or exceeds
`target_power * (1 - tolerance)`. When JSON output is enabled (`-j`), an
`average power` field is also included in the result record.

### Examples

**Example:**

Run:

    ./rvs -c conf/MI355X/iet_stress.conf

Configuration (`conf/MI355X/iet_stress.conf`, first action):

    actions:
    - name: iet-stress-1400W-true
      device: all
      module: iet
      parallel: true
      duration: 600000
      ramp_interval: 1000
      sample_interval: 5000
      log_interval: 5000
      target_power: 1400
      tolerance: 0.01
      bw_workload: true
      cp_workload: false
      wg_count: 256
      nt_loads: true

Sample output (first action, abridged):

    [RESULT] [302166.984877] Action name :iet-stress-1400W-true
    [RESULT] [302167.47707 ] Module name :iet
    [RESULT] [302167.678963] [iet-stress-1400W-true] [GPU:: 42583] Power(W) 241.000000
    [RESULT] [302167.679266] [iet-stress-1400W-true] [GPU:: 36479] Power(W) 238.000000
    [RESULT] [302167.679363] [iet-stress-1400W-true] [GPU:: 17010] Power(W) 242.000000
    [RESULT] [302167.679366] [iet-stress-1400W-true] [GPU:: 27226] Power(W) 251.000000
    [RESULT] [302167.686277] [iet-stress-1400W-true] [GPU::  1590] Power(W) 244.000000
    [RESULT] [302167.686419] [iet-stress-1400W-true] [GPU:: 57875] Power(W) 247.000000
    [RESULT] [302167.686591] [iet-stress-1400W-true] [GPU:: 11806] Power(W) 243.000000
    [RESULT] [302167.686605] [iet-stress-1400W-true] [GPU:: 51771] Power(W) 239.000000
    [RESULT] [302172.680300] [iet-stress-1400W-true] [GPU:: 36479] Power(W) 1398.000000
    [RESULT] [302172.680303] [iet-stress-1400W-true] [GPU:: 42583] Power(W) 1398.000000
    [RESULT] [302172.680570] [iet-stress-1400W-true] [GPU:: 27226] Power(W) 1398.000000
    [RESULT] [302172.680606] [iet-stress-1400W-true] [GPU:: 17010] Power(W) 1398.000000
    [RESULT] [302172.687273] [iet-stress-1400W-true] [GPU:: 57875] Power(W) 1399.000000
    [RESULT] [302172.687390] [iet-stress-1400W-true] [GPU::  1590] Power(W) 1399.000000
    [RESULT] [302172.687638] [iet-stress-1400W-true] [GPU:: 11806] Power(W) 1400.000000
    [RESULT] [302172.687845] [iet-stress-1400W-true] [GPU:: 51771] Power(W) 1399.000000
    [RESULT] [302177.681560] [iet-stress-1400W-true] [GPU:: 36479] Power(W) 1395.000000
    [RESULT] [302177.681561] [iet-stress-1400W-true] [GPU:: 42583] Power(W) 1393.000000
    [RESULT] [302177.681698] [iet-stress-1400W-true] [GPU:: 27226] Power(W) 1394.000000
    [RESULT] [302177.681724] [iet-stress-1400W-true] [GPU:: 17010] Power(W) 1395.000000
    ...
    [RESULT] [302347.724033] [iet-stress-1400W-true] [GPU:: 42583] Power(W) 1399.000000
    [RESULT] [302347.724905] [iet-stress-1400W-true] [GPU:: 27226] pass: TRUE
    [RESULT] [302347.727817] [iet-stress-1400W-true] [GPU:: 17010] pass: TRUE
    [RESULT] [302347.729796] [iet-stress-1400W-true] [GPU:: 42583] pass: TRUE
    [RESULT] [302347.730984] [iet-stress-1400W-true] [GPU:: 11806] Power(W) 1400.000000
    [RESULT] [302347.730984] [iet-stress-1400W-true] [GPU::  1590] Power(W) 1401.000000
    [RESULT] [302347.730997] [iet-stress-1400W-true] [GPU:: 57875] Power(W) 1400.000000
    [RESULT] [302347.731021] [iet-stress-1400W-true] [GPU:: 51771] Power(W) 1399.000000
    [RESULT] [302347.732052] [iet-stress-1400W-true] [GPU:: 36479] pass: TRUE
    [RESULT] [302347.737108] [iet-stress-1400W-true] [GPU::  1590] pass: TRUE
    [RESULT] [302347.739075] [iet-stress-1400W-true] [GPU:: 11806] pass: TRUE
    [RESULT] [302347.740868] [iet-stress-1400W-true] [GPU:: 51771] pass: TRUE
    [RESULT] [302347.742768] [iet-stress-1400W-true] [GPU:: 57875] pass: TRUE

    +=====================================================================+
    |                 ROCm Validation Suite (RVS) Summary                 |
    +=====================================================================+
    |                           System Overview                           |
    +---------------------------------------------------------------------+
    | Operating System                 | Ubuntu 22.04.5 LTS               |
    | RVS version                      | 1.6.75                           |
    | ROCm version                     | 7.2.1-81                         |
    | amdgpu version                   | 6.16.13                          |
    | GPUs                             | 8                                |
    +---------------------------------------------------------------------+
    | AMD Instinct MI355X - 42583      | AMD Instinct MI355X - 27226      |
    | 0 - 2 - 0000:05:00.0             | 1 - 3 - 0000:15:00.0             |
    +---------------------------------------------------------------------+
    | AMD Instinct MI355X - 36479      | AMD Instinct MI355X - 17010      |
    | 2 - 4 - 0000:65:00.0             | 3 - 5 - 0000:75:00.0             |
    +---------------------------------------------------------------------+
    | AMD Instinct MI355X - 1590       | AMD Instinct MI355X - 51771      |
    | 4 - 6 - 0000:85:00.0             | 5 - 7 - 0000:95:00.0             |
    +---------------------------------------------------------------------+
    | AMD Instinct MI355X - 11806      | AMD Instinct MI355X - 57875      |
    | 6 - 8 - 0000:e5:00.0             | 7 - 9 - 0000:f5:00.0             |
    +=====================================================================+
    | Action Name                      | Module         | Result          |
    +=====================================================================+
    | iet-stress-1400W-true            | IET            | PASS            |
    +---------------------------------------------------------------------+


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
<tr><td>max_temp_c</td><td>Float</td><td>Junction temperature ceiling in degrees Celsius. If the GPU junction temperature exceeds this threshold during the run, the worker logs a thermal-violation error and, when <b>halt_on_error</b> is true, terminates that GPU thread. Default <b>105.0</b>. A value of <b>0</b> disables thermal checking entirely.</td></tr>
</table>
</div>

### Output

Log lines use the action name, module tag pulse, and GPU ID.

<div class="pst-scrollable-table-container">
<table class="table table--middle-left">
<tr><th class="head">Output / log</th> <th class="head">Description</th></tr>
<tr><td>Start</td><td><code>[INFO] ... pulse &lt;gpu_id&gt; start pulse_rate=&lt;hz&gt;</code></td></tr>
<tr><td>Parameters</td><td><code>[INFO] ... pulse_rate=... Hz period=...ms high=...ms low=...ms</code></td></tr>
<tr><td>Thermal error</td><td><code>[ERROR] ... thermal violation: &lt;temp&gt;C (limit &lt;max_temp_c&gt;C)</code> (emitted only when <b>max_temp_c</b> is &gt; 0)</td></tr>
<tr><td>Periodic summary</td><td>At each <b>log_interval</b>, moving averages and extrema: <code>pulse #N avg_high=...W avg_low=...W max_high=...W min_low=...W delta=...W</code></td></tr>
<tr><td>Completion</td><td>Summary over the run: pulse count, average high/low power, delta, max high, min low.</td></tr>
<tr><td>pass</td><td><code>[RESULT] ... pass: true</code> or <code>false</code> per GPU.</td></tr>
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
<td>If true, intended to enable the memory stress test (Test 10) in addition to
the standard tests. The default value is false. Note: this key is parsed but
does not affect which tests run in the current implementation; all tests are
always executed.</td></tr>
<tr><td>mapped_memory</td><td>Bool</td>
<td>If true, uses host-mapped (pinned) memory instead of device memory for the
test buffers. The default value is false.</td></tr>
<tr><td>num_iter</td><td>Integer</td>
<td>Number of iterations to run per test. The default value is 1.</td></tr>
<tr><td>exclude</td><td>Collection of Integers</td>
<td>Space-separated list of test indices (0–10) to skip. For example,
<b>exclude: 9 10</b> is intended to skip the bit fade and memory stress tests.
Note: this key is parsed but does not affect which tests run in the current
implementation; all tests are always executed.</td></tr>
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

**Example:**

Run:

    ./rvs -c conf/mem.conf

Configuration (`conf/mem.conf`, first action):

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

Sample output (first action, abridged):

    [RESULT] [302697.357999] Action name :action_1
    [RESULT] [302697.415572] Module name :mem
    [RESULT] [302697.637264] [action_1] mem   The following memory tests will run
    [RESULT] [302697.637269] =============== Test 1  [Walking 1 bit]
    [RESULT] [302697.637270] =============== Test 2  [Own address test]
    [RESULT] [302697.637270] =============== Test 3  [Moving inversions, ones&zeros]
    [RESULT] [302697.637271] =============== Test 4  [Moving inversions, 8 bit pat]
    [RESULT] [302697.637271] =============== Test 5  [Moving inversions, random pattern]
    [RESULT] [302697.637271] =============== Test 6  [Block move, 64 moves]
    [RESULT] [302697.637271] =============== Test 7  [Moving inversions, 32 bit pat]
    [RESULT] [302697.637272] =============== Test 8  [Random number sequence]
    [RESULT] [302697.637272] =============== Test 9  [Modulo 20, random pattern]
    [RESULT] [302697.869004] [action_1] mem Test 1: Change one bit memory addresss
    [RESULT] [302697.872232] [action_1] mem Test 1: Change one bit memory addresss
    [RESULT] [302697.890031] [action_1] mem Test 1: Change one bit memory addresss
    [RESULT] [302697.893155] [action_1] mem Test 1: Change one bit memory addresss
    [RESULT] [302697.895754] [action_1] mem Test 1 : PASS
    [RESULT] [302697.895771] [action_1] mem Test 2: Each Memory location is filled with its own address
    [RESULT] [302697.896227] [action_1] mem Test 1: Change one bit memory addresss
    [RESULT] [302697.900155] [action_1] mem Test 1 : PASS
    [RESULT] [302697.900173] [action_1] mem Test 2: Each Memory location is filled with its own address
    [RESULT] [302697.903915] [action_1] mem Test 1: Change one bit memory addresss
    ...
    [RESULT] [302732.976817] [action_1] mem Test 11: elapsedtime = 1986.538940 bandwidth = 6443.431641GB/s
    [RESULT] [302732.978980] [action_1] mem Test 11 : PASS
    [RESULT] [302733.171129] [action_1] mem Test 11: elapsedtime = 1989.459717 bandwidth = 6433.972168GB/s
    [RESULT] [302733.173334] [action_1] mem Test 11 : PASS
    [RESULT] [302733.208421] [action_1] mem Test 11: elapsedtime = 1974.268921 bandwidth = 6483.477539GB/s
    [RESULT] [302733.210793] [action_1] mem Test 11 : PASS
    [RESULT] [302730.722708] [action_1] mem Test 11: elapsedtime = 1964.402588 bandwidth = 6516.041016GB/s
    [RESULT] [302730.724818] [action_1] mem Test 11 : PASS
    [RESULT] [302731.757998] [action_1] mem Test 11: elapsedtime = 2002.477539 bandwidth = 6392.145508GB/s
    [RESULT] [302731.760200] [action_1] mem Test 11 : PASS

    +=====================================================================+
    |                 ROCm Validation Suite (RVS) Summary                 |
    +=====================================================================+
    |                           System Overview                           |
    +---------------------------------------------------------------------+
    | Operating System                 | Ubuntu 22.04.5 LTS               |
    | RVS version                      | 1.6.75                           |
    | ROCm version                     | 7.2.1-81                         |
    | amdgpu version                   | 6.16.13                          |
    | GPUs                             | 8                                |
    +---------------------------------------------------------------------+
    | AMD Instinct MI355X - 42583      | AMD Instinct MI355X - 27226      |
    | 0 - 2 - 0000:05:00.0             | 1 - 3 - 0000:15:00.0             |
    +---------------------------------------------------------------------+
    | AMD Instinct MI355X - 36479      | AMD Instinct MI355X - 17010      |
    | 2 - 4 - 0000:65:00.0             | 3 - 5 - 0000:75:00.0             |
    +---------------------------------------------------------------------+
    | AMD Instinct MI355X - 1590       | AMD Instinct MI355X - 51771      |
    | 4 - 6 - 0000:85:00.0             | 5 - 7 - 0000:95:00.0             |
    +---------------------------------------------------------------------+
    | AMD Instinct MI355X - 11806      | AMD Instinct MI355X - 57875      |
    | 6 - 8 - 0000:e5:00.0             | 7 - 9 - 0000:f5:00.0             |
    +=====================================================================+
    | Action Name                      | Module         | Result          |
    +=====================================================================+
    | action_1                         | MEM            | PASS            |
    +---------------------------------------------------------------------+


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
<td>Number of elements in the test array (element count, not bytes). The actual
memory size depends on the data type selected by <b>test_type</b>. The default
value is 33554432 (32 M elements).</td></tr>
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
<td>If true, bandwidth is reported in GiB/s. If false, GB/s are used. This
key affects the reporting unit only; it does not change how <b>array_size</b>
is interpreted. The default value is false.</td></tr>
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
<b>gpu_norm_dist</b> – Initialize on GPU with a normal distribution.
<b>cpu_norm_dist</b> – Initialize on CPU with a normal distribution.
<b>zero_init</b> – Initialize all elements to zero.</td></tr>
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
<tr><td>Function</td><td>String</td>
<td>Name of the BabelStream kernel executed (e.g., Copy, Mul, Add, Triad,
Dot, Read, Write).</td></tr>
<tr><td>MBytes/sec</td><td>Float</td>
<td>Measured memory bandwidth for this kernel in MB/s (or MiB/s if
<b>mibibytes: true</b>).</td></tr>
<tr><td>Max_MBytes/sec</td><td>Float</td>
<td>Maximum memory bandwidth observed across all iterations for this
kernel.</td></tr>
<tr><td>pass</td><td>Bool</td>
<td>True if the kernel completed successfully.</td></tr>
</table>
</div>

### Examples

**Example:**

Run:

    ./rvs -c conf/MI355X/babel.conf

Configuration (`conf/MI355X/babel.conf`, first action):

    actions:
    - name: babel-double-825MiB
      device: all
      module: babel
      parallel: false
      count: 1
      num_iter: 2000
      duration: 0
      array_size: 865075200
      test_type: 2
      mibibytes: false
      o/p_csv: false
      read: true
      write: true
      copy: true
      mul: true
      add: true
      dot: true
      triad: true
      dwords_per_lane: 4
      chunks_per_block: 1
      tb_size: 512

Sample output (first action, abridged):

    [RESULT] [302411.15197 ] Action name :babel-double-825MiB
    [RESULT] [302411.96856 ] Module name :babel
    [RESULT] [302411.735270] [babel-double-825MiB] [GPU:: 42583] Starting the Babel memory stress test
    [RESULT] [302411.735337] Running kernels 2000 times, Precision: double
    [RESULT] [302411.735378] Array size: 6920.6 MB (=6.9 GB), Total size: 20761.8 MB (=20.8 GB)

    [RESULT] [302446.903905]
    ---------------------------------------------------------------------------------
    GPU Id      Function    MBytes/sec     Max MB/s       Min MB/s       Avg MB/s
    ---------------------------------------------------------------------------------
    42583       Read        7090191.900    7090191.900    6638148.386    7019091.298
    42583       Write       6645670.223    6645670.223    5618652.956    6106108.680
    42583       Copy        6224488.986    6224488.986    6037686.546    6150313.849
    42583       Mul         6238685.230    6238685.230    6056284.950    6152982.512
    42583       Add         6008723.700    6008723.700    5485279.487    5919255.626
    42583       Triad       6031416.066    6031416.066    5879289.313    5950804.403
    42583       Dot         5844542.436    5844542.436    4912455.933    5806381.609
    ---------------------------------------------------------------------------------
    ...

    +=====================================================================+
    |                 ROCm Validation Suite (RVS) Summary                 |
    +=====================================================================+
    |                           System Overview                           |
    +---------------------------------------------------------------------+
    | Operating System                 | Ubuntu 22.04.5 LTS               |
    | RVS version                      | 1.6.75                           |
    | ROCm version                     | 7.2.1-81                         |
    | amdgpu version                   | 6.16.13                          |
    | GPUs                             | 8                                |
    +---------------------------------------------------------------------+
    | AMD Instinct MI355X - 42583      | AMD Instinct MI355X - 27226      |
    | 0 - 2 - 0000:05:00.0             | 1 - 3 - 0000:15:00.0             |
    +---------------------------------------------------------------------+
    | AMD Instinct MI355X - 36479      | AMD Instinct MI355X - 17010      |
    | 2 - 4 - 0000:65:00.0             | 3 - 5 - 0000:75:00.0             |
    +---------------------------------------------------------------------+
    | AMD Instinct MI355X - 1590       | AMD Instinct MI355X - 51771      |
    | 4 - 6 - 0000:85:00.0             | 5 - 7 - 0000:95:00.0             |
    +---------------------------------------------------------------------+
    | AMD Instinct MI355X - 11806      | AMD Instinct MI355X - 57875      |
    | 6 - 8 - 0000:e5:00.0             | 7 - 9 - 0000:f5:00.0             |
    +=====================================================================+
    | Action Name                      | Module         | Result          |
    +=====================================================================+
    | babel-double-825MiB              | BABEL          | PASS            |
    +---------------------------------------------------------------------+
