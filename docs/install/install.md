
## Installing RVS

RVS can be obtained by building it from source code base or by installing from
pre-built package.

### Building from Source Code

RVS has been developed as open source solution. Its source code and belonging documentation can be found at AMD's GitHub page.

To build RVS from source code, refer to
[ROCm Validation Suite GitHubsite](https://github.com/ROCm/ROCmValidationSuite) and follow instructions in README file.

### Installing from Package
Based on the OS, use the appropriate package manager to install the **rocm-validation-suite** package.

For more details, refer to [ROCm Validation Suite GitHub site](https://github.com/ROCm/ROCmValidationSuite).

RVS package components are installed in `/opt/rocm`. Package contains:
- executable binary (located in _install-base_/bin/rvs)
- public shared libraries (located in _install-base_/lib)
- module specific shared libraries (located in _install-base_/lib/rvs)
- configuration files (located in _install-base_/share/rocm-validation-suite/conf)
- testscripts (located in _install-base_/share/rocm-validation-suite/testscripts)
- user guide (located in _install-base_/share/rocm-validation-suite/userguide)
- man page (located in _install-base_/share/man)

### Running RVS

#### Run version built from source code

    cd <source folder>/build/bin

    Command examples
    ./rvs --help ; Lists all options to run RVS test suite
    ./rvs -g ; Lists supported GPUs available in the machine
    ./rvs -d 3 ; Run set of RVS sanity tests (in rvs.conf) with verbose level 3
    ./rvs -c conf/gst_single.conf ; Run GST module tests

#### Run version pre-complied and packaged with ROCm release

    cd /opt/rocm/bin

    Command examples
    ./rvs --help ; Lists all options to run RVS test suite
    ./rvs -g ; Lists supported GPUs available in the machine
    ./rvs -d 3 ; Run set of RVS sanity tests (in rvs.conf) with verbose level 3
    ./rvs -c conf/gst_single.conf ; Run GST module tests

Similarly, all RVS module tests can be run using scripts present in folder "/opt/rocm/share/rocm-validation-suite/testscripts/".

## Reporting

Test results, errors and verbose logs are printed as terminal output. To enable json logging use "-j" command line option.
The json output file is stored in /var/tmp folder and the name of the file will be printed.

