# ROCmValidationSuite Modules


## GPU Properties – GPUP
The GPU Properties module queries the configuration of a target device and returns the device’s static characteristics. These static values can be used to debug issues such as device support, performance and firmware problems. 

## GPU Monitor – GM module
The GPU monitor tool is capable of running on one, some or all of the GPU(s) installed and will report various information at regular intervals. The module can be configured to halt another RVS modules execution if one of the quantities exceeds a specified boundary value.

## PCI Express State Monitor – PESM module
The PCIe State Monitor tool is used to actively monitor the PCIe interconnect between the host platform and the GPU. The module will register a “listener” on a target GPU’s PCIe interconnect, and log a message whenever it detects a state change. The PESM will be able to detect the following state changes:

1.	PCIe link speed changes
2.	GPU power state changes

## ROCm Configuration Qualification Tool - RCQT module
The ROCm Configuration Qualification Tool ensures the platform is capable of running ROCm applications and is configured correctly. It checks the installed versions of the ROCm components and the platform configuration of the system. This includes checking the dependencies corresponding to the ROCm meta-packages are installed correctly.

## PCI Express Qualification Tool – PEQT module
The PCIe Qualification Tool is used to qualify the PCIe bus on which the GPU is connected. The qualification test will be capable of determining the following characteristics of the PCIe bus interconnect to a GPU:

1.	Support for Gen 3 atomic completers
2.	DMA transfer statistics
3.	PCIe link speed
4.	PCIe link width

## SBIOS Mapping Qualification Tool – SMQT module
The GPU SBIOS mapping qualification tool is designed to verify that a platform’s SBIOS has satisfied the BAR mapping requirements for VDI and Radeon Instinct products for ROCm support.

Refer to the “ROCm Use of Advanced PCIe Features and Overview of How BAR Memory is Used In ROCm Enabled System” web page for more information about how BAR memory is initialized by VDI and Radeon products.

## P2P Benchmark and Qualification Tool – PBQT module
The P2P Benchmark and Qualification Tool  is designed to provide the list of all GPUs that support P2P and characterize the P2P links between peers. In addition to testing for P2P compatibility, this test will perform a peer-to-peer throughput test between all P2P pairs for performance evaluation. The P2P Benchmark and Qualification Tool will allow users to pick a collection of two or more GPUs on which to run. The user will also be able to select whether or not they want to run the throughput test on each of the pairs.

Please see the web page “ROCm, a New Era in Open GPU Computing” to find out more about the P2P solutions available in a ROCm environment.

## PCI Express Bandwidth Benchmark – PEBB module
The PCIe Bandwidth Benchmark attempts to saturate the PCIe bus with DMA transfers between system memory and a target GPU card’s memory. The maximum bandwidth obtained is reported to help debug low bandwidth issues. The benchmark should be capable of  targeting one, some or all of the GPUs installed in a platform, reporting individual benchmark statistics for each.

## GPU Stress Test - GST module
The GPU Stress Test runs a Graphics Stress test or SGEMM/DGEMM/HGEMM (Single/Double/Half-precision General Matrix Multiplication) workload on one, some or all GPUs. The GPUs can be of the same or different types. The duration of the benchmark should be configurable, both in terms of time (how long to run) and iterations (how many times to run).

The test should be capable driving the power level equivalent to the rated TDP of the card, or levels below that. The tool must be capable of driving cards at TDP-50% to TDP-100%, in 10% incremental jumps. This should be controllable by the user.

## Input EDPp Test - IET module
The Input EDPp Test generates EDP peak power on all input rails. This test is used to verify if the system PSU is capable of handling the worst case power spikes of the board.  Peak Current at defined period  =  1 minute moving average power.

## Memory Test - MEM module
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

## BABEL benchmark Test - BABEL module
The Babel module executes BabelStream (synthetic GPU benchmark based on the original STREAM benchmark for CPUs) benchmark that measures memory transfer rates (bandwidth) to and from global device memory. Various benchmark tests are implemented using GPU kernels in HIP (Heterogeneous Interface for Portability) programming language.
