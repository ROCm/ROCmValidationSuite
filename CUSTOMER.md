# ROCmValidationSuite tests

RVS supports different tests, some of the test that can be run by the customer are listed below: 

- rvs/conf/customer/  This directory contains stress and perforance tests
- rvs/conf/           This directory contains stress and perforance tests and other tests like state monitoring tests
- rvs/conf/archive    This directory contains example tests that can be referred to create new tests


## Listing out the customer directory tests (rvs/conf/customer/)
1.	GPU properties                               : gpup.conf
2.	GST SGEMM stress/performance test            : gst_sgemm.conf 
3.	GST DGEMM stress/performance test            : gst_dgemm.conf 
4.	GST HGEMM stress/performance test            : gst_hgemm.conf 
5.	GPU2GPU uni-directional bandwidth            : gpu2gpu_unidir_bw.conf
6.	GPU2GPU bi-directional bandwidth             : gpu2gpu_bidir_bw.conf
7.	Power 200W test                              : iet_200W.conf
8.	Power 220W test                              : iet_220W.conf
9.	Memory Test                                  : mem.conf
10.	Memory stress test                           : memory_stress.conf
11.	PCIe unidirectional bandwidth tests          : PCIE_unidir_BW_test.conf 
12.	PCIe bidirectional bandwidth tests           : PCIE_bidir_BW_test.conf

### Running Long duration stress test
    cd build/bin
    ./rvs-stress-long.sh (defaults to 1 hour stress test)
    ./rvs-stress-long.sh 3 (user specified no of hours for stress test)
