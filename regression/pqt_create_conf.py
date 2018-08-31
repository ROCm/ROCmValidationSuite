#!/usr/bin/env python

# This script only creates valid combinations, invalid ones will be created as special cases

import os

from random import seed
from random import random
from random import sample

# global variables
module_name = "demofile"

#gpu_ids        = [3254, 33367, 50599]
gpu_ids        = [3254]
log_interval   = [1000]
duration       = [10000]
device_id      = [-1, 26720]
test_bandwidth = ['true', 'false']
bidirectional  = ['true', 'false']
parralel       = ['true', 'false']

# RVS build folder
build_location = os.environ['RVS_BUILD']
# location of configuration files
conf_location = build_location + "/regression/conf/"
if not os.path.exists(conf_location):
    os.makedirs(conf_location)

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
                            filename = conf_location + module_name + str(counter) + ".conf"
                            print 'Iteration is %d' % (counter) + ", working on conf file " + filename
                            f = open(filename, "w")
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
