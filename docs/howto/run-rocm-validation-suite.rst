


Running ROCm Validation Suite (RVS)
************************************

Run version built from the source code
---------------------------------------
```
        cd <source folder>/build/bin
```
Command examples

```
        ./rvs --help ; Lists all options to run RVS test suite
        ./rvs -g ; Lists supported GPUs available in the machine
        ./rvs -d 3 ; Run set of RVS sanity tests (in rvs.conf) with verbose level 3
        ./rvs -c conf/gst_single.conf ; Run GST module tests

```

Run version pre-complied and packaged with ROCm release
---------------------------------------------------------
``  
        cd /opt/rocm/bin
```

Command examples

```
        ./rvs --help ; Lists all options to run RVS test suite
        ./rvs -g ; Lists supported GPUs available in the machine
        ./rvs -d 3 ; Run set of RVS sanity tests (in rvs.conf) with verbose level 3
        ./rvs -c conf/gst_single.conf ; Run GST module tests
```

Similarly, all RVS module tests can be run using scripts present in folder "/opt/rocm/share/rocm-validation-suite/testscripts/".
