#!/usr/bin/env python

# This script only creates valid combinations, invalid ones will be created as special cases
from __future__ import print_function

import os
import itertools
import sys

from random import seed
from random import random
from random import sample

# global variables
module_name = "pqt"
cmake_file_name = "rand_tests.cmake"

# collect gpu IDs of registered devices
gpu_ids   = set()
device_id = set()
for root, dirs, files in os.walk('/sys/class/kfd/kfd/topology/nodes'):
	for name in files:
		if name == 'gpu_id':
			gpuid = os.popen('cat {}'.format(os.path.join(root, name))).read().rstrip()
			if gpuid not in ['0', '']:
				devid = os.popen("grep 'device_id' {} | cut -f 2 -d ' '".format(os.path.join(root, 'properties'))).read().rstrip()
				device_id.add(int(devid))
				gpu_ids.add(int(gpuid))

log_interval   = [1000]
duration       = [10000]
test_bandwidth = [True, False]
bidirectional  = [True, False]
parralel       = [True, False]

# RVS build folder
build_location = os.path.dirname(os.path.realpath(__file__))
build_location = build_location + "/.."
print(build_location)

# location of configuration files
conf_location = build_location + "/rvs/conf/"
cmake_file_location = build_location + "/" + module_name + ".so/" + cmake_file_name

try:
	cmake_file = open(cmake_file_location, "w")
except OSError:
	print('Could not open file {} for writing. Terminating...'.format(cmake_file_location))
	sys.exit(1)

cmake_file_header = \
"""###################################################################################
##
## Copyright (c) 2018 ROCm Developer Tools
##
## MIT LICENSE:
## Permission is hereby granted, free of charge, to any person obtaining a copy of
## this software and associated documentation files (the "Software"), to deal in
## the Software without restriction, including without limitation the rights to
## use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
## of the Software, and to permit persons to whom the Software is furnished to do
## so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included in all
## copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
## OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
## SOFTWARE.
##
###################################################################################


"""

cmake_file.write(cmake_file_header)

counter = 0
total_iterations = (len(gpu_ids) + 1) * len(log_interval) * len(duration) * len(test_bandwidth) * len(bidirectional) * len(parralel) * len(device_id)
print('Total number of combinations (including invalid) is {}'.format(total_iterations))

gpu_ids_size = len(gpu_ids)

combos = itertools.product(test_bandwidth, log_interval, duration, bidirectional, parralel, device_id)

# go through all combinations
for test_bandwidth_f, log_interval_f, duration_f, bidirectional_f, parralel_f, device_id_f in combos:

    # create several combinations of gpu_ids
    sample_size = 0
    while True:
        # skip invalid combinations of test_bandwidth x (other bandwidth calculation parameters)
        if not test_bandwidth_f and (bidirectional_f or parralel_f):
            # skip
            break

        # for each conf file add the new test in the cmake unit test list
        cmake_file.write('add_test(NAME rand.{}.{}\n'.format(module_name, counter))
        cmake_file.write('  WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}\n')
        cmake_file.write('  COMMAND rvs -d 3 -c conf/{}_module_{}.conf\n'.format(module_name, counter))
        cmake_file.write(')\n\n')

        # for each combination create the conf file
        filename = conf_location + "rand_" + module_name + str(counter) + ".conf"
        print('Iteration is {}, working on conf file {}'.format(counter, filename))
        try:
        	f = open(filename, "w")
        except OSError:
        	print('Could not open file {} for writing. Terminating...'.format(filename))
        	sys.exit(1)

        f.write('actions:\n')
        f.write('- name: action_1 \n')
        f.write('  device: all\n')
        f.write('  module: pqt\n')
        f.write('  log_interval: {}\n'.format(log_interval_f))
        f.write('  duration: {}\n'.format(duration_f))

        if sample_size:
            sample_gpus = sample(gpu_ids, sample_size)
            f.write('  peers:')
            for p in sample_gpus:
                f.write(' {}'.format(p))
            f.write('\n')
        else:
        	f.write('  peers: all\n')

        # device id is optional parameter
        if device_id_f > -1:
            f.write('  peer_deviceid: {}\n'.format(device_id_f))
        f.write('  test_bandwidth: {}\n'.format(str(test_bandwidth_f).lower()))
        f.write('  bidirectional: {}\n'.format(str(bidirectional_f).lower()))
        f.write('  parallel: {}\n'.format(str(parralel_f).lower()))
        f.close()

        sample_size += 1
        if sample_size == gpu_ids_size + 1:
            break

        # go to the next conf file
        counter += 1

cmake_file.close() 
