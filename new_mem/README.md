# cuda_memtest

This software tests GPU memory for hardware errors and soft errors using CUDA (or OpenCL).

## Note for this Fork

This is a fork of the original, yet long-time unmaintained project at https://sourceforge.net/projects/cudagpumemtest/ .

After our fork in 2013 (v1.2.3), we primarily focused on support for newer CUDA versions and support of newer Nvidia hardware.
Pull-requests maintaining the OpenCL versions are nevertheless still welcome.

## License

Illinois Open Source License

University of Illinois/NCSA  
Open Source License

Copyright 2009-2012,    University of Illinois.  All rights reserved.  
Copyright 2013-2019,    The developers of PIConGPU at Helmholtz-Zentrum Dresden-Rossendorf

Developed by:

  Innovative Systems Lab  
  National Center for Supercomputing Applications  
  http://www.ncsa.uiuc.edu/AboutUs/Directorates/ISL.html

Forked and maintained for newer Nvidia GPUs since 2013 by:

  Axel Huebl and Rene Widera  
  Computational Radiation Physics Group  
  Helmholtz-Zentrum Dresden-Rossendorf  
  https://www.hzdr.de/crp

Permission is hereby granted, free of charge, to any person obtaining a copy of 
this software and associated documentation files (the "Software"), to deal with 
the Software without restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the 
Software, and to permit persons to whom the Software is furnished to do so, subject
to the following conditions:

* Redistributions of source code must retain the above copyright notice, this list 
  of conditions and the following disclaimers.

* Redistributions in binary form must reproduce the above copyright notice, this list
  of conditions and the following disclaimers in the documentation and/or other materials
  provided with the distribution.

* Neither the names of the Innovative Systems Lab, the National Center for Supercomputing
  Applications, nor the names of its contributors may be used to endorse or promote products
  derived from this Software without specific prior written permission.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT 
OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS WITH THE SOFTWARE.

## Compile and Run

### Compile

Inside the source directory, run:
```bash
mkdir build
cd build
cmake ..
make
```

Note: In CMake, `..` is the path to the source directory.

We also provide the package `cuda-memtest` in the [Spack package manager](https://spack.io) .

### Run

```
cuda_memtest
```
The default behavior is running the test on all the GPUs available infinitely.
There are options to change the default behavior. 

```
cuda_memtest --disable_all --enable_test 10
cuda_memtest --stress
```
This runs test 10 (the stress test). `--stress` is equivalent to `--disable_all --enable_test 10 --exit_on_error`

```
cuda_memtest --stress --num_iterations 100 --num_passes 1
```
This one does a quick sanity check for GPUs with a short run of test 10. More on this later.

See help message by 

```
cuda_memtest --help
```

### Sanity Check

There is a simple script `sanity_check.sh` in the directory. 
This script does a quick check if one GPU or all GPUs are in bad health.

Example usage: 
```bash
# copy the cuda_memtest binary first into the same location as this script, e.g.
cd ..
mv build/cuda_memtest .
```
```
./sanity_check.sh 0   //check GPU 0
./sanity_check.sh 1   //check GPU 1 
./sanity_check.sh     //check All GPUs in the system
```

Fork note: We just run the `cuda_memtest` binary directly.
Consider this script as a source for inspiration, or so.

### Known Issues

* If your machine is cuda 2.2, killing the program while it is running test 10 (the memory stress test) could result 
  in your GPUs in bad state. This is a bug from the nvidia driver. A detailed description can be found in 
  http://forums.nvidia.com/index.php?showtopic=97379. We have filed a bug report to nvidia.
  Rebooting or reloading the nvidia driver will put the GPUs back to clean state.

Note: You are not using CUDA 2.2 anymore, are you? ;-)

* We are not maintaining the OpenCL version of this code base.
  Pull requests restoring and updating the OpenCL capabilities are welcome.

## Test Descriptions

### List of all Tests

Running 
```
cuda_memtest --list_tests
```
will print out all tests and their short descriptions, as of 6/18/2009, we implemented 11 tests

```
Test0 [Walking 1 bit] 
Test1 [Own address test] 
Test2 [Moving inversions, ones&zeros] 
Test3 [Moving inversions, 8 bit pat] 
Test4 [Moving inversions, random pattern] 
Test5 [Block move, 64 moves] 
Test6 [Moving inversions, 32 bit pat] 
Test7 [Random number sequence] 
Test8 [Modulo 20, random pattern] 
Test9 [Bit fade test]  ==disabled by default==
Test10 [Memory stress test] 
```

### The General Algorithm

First a kernel is launched to write a pattern.
Then we exit the kernel so that the memory can be flushed. Then we start a new kernel to read
and check if the value matches the pattern. An error is recorded if it does not match for each 
memory location. In the same kernel, the compliment of the pattern is written after the checking. 
The third kernel is launched to read the value again and checks against the compliment of the pattern. 

### Detailed Description

Test 0 `[Walking 1 bit]`  
	This test changes one bit a time in memory address to see it
	goes to a different memory location. It is designed to test
	the address wires. 

Test 1 `[Own address test]`  
	Each Memory location is filled with its own address. The next kernel checks if the 
	value in each memory location still agrees with the address.

Test 2 `[Moving inversions, ones&zeros]`  
	This test uses the moving inversions algorithm with patterns of all
	ones and zeros. 

Test 3 `[Moving inversions, 8 bit pat]`  
	This is the same as test 1 but uses a 8 bit wide pattern of
	"walking" ones and zeros.  This test will better detect subtle errors
	in "wide" memory chips. 

Test 4 `[Moving inversions, random pattern]`  
	Test 4 uses the same algorithm as test 1 but the data pattern is a
	random number and it's complement. This test is particularly effective
	in finding difficult to detect data sensitive errors. The random number 
	sequence is different with each pass so multiple passes increase effectiveness.

Test 5 `[Block move, 64 moves]`  
	This test stresses memory by moving block memories. Memory is initialized
	with shifting patterns that are inverted every 8 bytes.  Then blocks
	of memory are moved around.  After the moves
	are completed the data patterns are checked.  Because the data is checked
	only after the memory moves are completed it is not possible to know
	where the error occurred.  The addresses reported are only for where the
	bad pattern was found.

Test 6 `[Moving inversions, 32 bit pat]`  
	This is a variation of the moving inversions algorithm that shifts the data
	pattern left one bit for each successive address. The starting bit position
	is shifted left for each pass. To use all possible data patterns 32 passes
	are required.  This test is quite effective at detecting data sensitive
	errors but the execution time is long.

Test 7 `[Random number sequence]`  
	This test writes a series of random numbers into memory.  A block (1 MB) of memory
	is initialized with random patterns. These patterns and their complements are
	used in moving inversions test with rest of memory.

Test 8 `[Modulo 20, random pattern]`  
	A random pattern is generated. This pattern is used to set every 20th memory location
	in memory. The rest of the memory location is set to the complimemnt of the pattern.
	Repeat this for 20 times and each time the memory location to set the pattern is shifted right.

Test 9 `[Bit fade test, 90 min, 2 patterns]`  
	The bit fade test initializes all of memory with a pattern and then
	sleeps for 90 minutes. Then memory is examined to see if any memory bits
	have changed. All ones and all zero patterns are used. This test takes
	3 hours to complete. The Bit Fade test is disabled by default

Test 10 `[memory stress test]`  
	Stress memory as much as we can. A random pattern is generated and a kernel of large grid size
	and block size is launched to set all memory to the pattern. A new read and write kernel is launched
	immediately after the previous write kernel to check if there is any errors in memory and set the
	memory to the compliment. This process is repeated for 1000 times for one pattern. The kernel is 
	written as to achieve the maximum bandwidth between the global memory and GPU.
	This will increase the chance of catching software error. In practice, we found this test quite useful 
	to flush hardware errors as well.
