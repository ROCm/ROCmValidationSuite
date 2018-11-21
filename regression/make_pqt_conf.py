#!/usr/bin/env python

# This script only creates valid combinations, invalid ones will be created as special cases

import os

from random import seed
from random import random
from random import sample

# global variables
module_name = "pqt"
cmake_file_name = "rand_tests.cmake"

#gpu_ids        = [3254, 33367, 50599]
gpu_ids        = [3254]
log_interval   = [1000]
duration       = [10000]
device_id      = [-1, 26720]
test_bandwidth = ['true', 'false']
bidirectional  = ['true', 'false']
parralel       = ['true', 'false']

# RVS build folder
build_location = os.path.dirname(os.path.realpath(__file__))
build_location = build_location + "/.."
print build_location

# location of configuration files
conf_location = build_location + "/rvs/conf/"
cmake_file_location = build_location + "/" + module_name + ".so/" + cmake_file_name
cmake_file = open(cmake_file_location, "w")

cmake_file.write("################################################################################\n")
cmake_file.write("##\n")
cmake_file.write("## Copyright (c) 2018 ROCm Developer Tools\n")
cmake_file.write("##\n")
cmake_file.write("## MIT LICENSE:\n")
cmake_file.write("## Permission is hereby granted, free of charge, to any person obtaining a copy of\n")
cmake_file.write("## this software and associated documentation files (the \"Software\"), to deal in\n")
cmake_file.write("## the Software without restriction, including without limitation the rights to\n")
cmake_file.write("## use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies\n")
cmake_file.write("## of the Software, and to permit persons to whom the Software is furnished to do\n")
cmake_file.write("## so, subject to the following conditions:\n")
cmake_file.write("##\n")
cmake_file.write("## The above copyright notice and this permission notice shall be included in all\n")
cmake_file.write("## copies or substantial portions of the Software.\n")
cmake_file.write("##\n")
cmake_file.write("## THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n")
cmake_file.write("## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n")
cmake_file.write("## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE\n")
cmake_file.write("## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n")
cmake_file.write("## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n")
cmake_file.write("## OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\n")
cmake_file.write("## SOFTWARE.\n")
cmake_file.write("##\n")
cmake_file.write("################################################################################\n")

cmake_file.write("\n\n")

counter = 0
total_iterations = (len(gpu_ids) + 1) * len(log_interval) * len(duration) * len(test_bandwidth) * len(bidirectional) * len(parralel) * len(device_id)
print "Total number of combinations (including invalid) is " + str(total_iterations)

gpu_ids_size = len(gpu_ids)

for test_bandwidth_f in test_bandwidth:
    for log_interval_f in log_interval:
        for duration_f in duration:
            for bidirectional_f in bidirectional:
                for parralel_f in parralel:
                    for device_id_f in device_id:
                        # crete several combinations of gpu_ids
                        sample_size = 0
                        while True:
                            # skip invalid combinations of test_bandwidth x (other bandwidth calculation parameters)
                            if test_bandwidth_f == 'false' and (bidirectional_f == 'true' or parralel_f == 'true'):
                                # skip
                                break

                            # for each combination create the conf file
                            filename = conf_location + "rand_" + module_name + str(counter) + ".conf"
                            print 'Iteration is %d' % (counter) + ", working on conf file " + filename
                            f = open(filename, "w")

                            # for each conf file add the new test in the cmake unit test list
                            cmake_file.write("add_test(NAME rand." + module_name + "." + str(counter) + "\n")
                            cmake_file.write("  WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}\n")
                            cmake_file.write("  COMMAND rvs -d 3 -c conf/" + module_name + "_module_" + str(counter) + ".conf" + "\n")
                            cmake_file.write(")\n\n")

                            # go to the next conf file
                            counter = counter + 1
                            
                            f.write("actions:" + "\n")
                            f.write("- name: action_1 " + "\n")
                            f.write("  device: all" + "\n")
                            f.write("  module: pqt" + "\n")
                            f.write("  log_interval: " + str(log_interval_f) + "\n")
                            f.write("  duration: " + str(duration_f) + "\n")

                            if sample_size == 0:
                                f.write("  peers: all" + "\n")
                            else:
                                sample_gpus = sample(gpu_ids, sample_size)
                                f.write("  peers:")
                                for p in sample_gpus:
                                    f.write(" " + str(p))
                                f.write("\n")

                            # device id is optional parameter
                            if device_id_f > -1:
                                f.write("  peer_deviceid: " + str(device_id_f) + "\n")
                            f.write("  test_bandwidth: " + str(test_bandwidth_f) + "\n")
                            f.write("  bidirectional: " + str(bidirectional_f) + "\n")
                            f.write("  parallel: " + str(parralel_f) + "\n")
                            f.close()

                            sample_size += 1
                            if sample_size == gpu_ids_size + 1:
                                break
cmake_file.close()
