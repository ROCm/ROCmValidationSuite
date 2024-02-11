.. meta::
    :description: ROCm validation Suite The ROCm Validation Suite (RVS) is a system validation and diagnostics tool for monitoring, stress testing, detecting, and troubleshooting issues.
    :keywords: rocm validation suite, ROCm, SPARSE, library, API, HIP

.. _what-is-ROCm-Validation-Suite:

*******************************
What is ROCm Validation Suite
*******************************
The ROCm Validation Suite (RVS) is a system validation and diagnostics tool for monitoring, stress testing, detecting, and troubleshooting issues that
affect the functionality and performance of AMD GPU(s) operating in a high-performance/AI/ML computing environment. RVS is enabled using the ROCm
software stack on a compatible software and hardware platform.

RVS is a collection of tests, benchmarks, and qualification tools, each targeting a specific sub-system of the ROCm platform. The tools are
implemented in software and share a common command line interface. Each set of tests is implemented in a “module, " a library encapsulating the
functionality specific to the tool. The CLI can specify the directory containing modules when searching for libraries to load. Each module may have a set
of options that it defines and a configuration file that supports its execution.
