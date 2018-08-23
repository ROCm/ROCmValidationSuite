#!/usr/bin/env python

from random import seed
from random import random
from random import sample

# global variables
module_name = "demofile"

gpu_ids        = [3254, 33367, 50599]
log_interval   = [100]
duration       = [1000]
device_id      = [-1, 26720]
test_bandwidth = ['true', 'false']
bidirectional  = ['true', 'false']
parralel       = ['true', 'false']

counter = 0
total_iterations = (len(gpu_ids) + 1) * len(log_interval) * len(duration) * len(test_bandwidth) * len(bidirectional) * len(parralel) * len(device_id) + 1
print "Total number of combinations is " + str(total_iterations)

while True:
    for gpu_ids_f in gpu_ids:
        for log_interval_f in log_interval:
            for duration_f in duration:
                for test_bandwidth_f in test_bandwidth:
                    for bidirectional_f in bidirectional:
                        for parralel_f in parralel:
                            for device_id_f in device_id:
                                # crete several combinations of gpu_ids
                                gpu_ids_size = len(gpu_ids)
                                sample_size = 0
                                while True:
                                    # for each combination create the conf file
                                    filename = module_name + str(counter) + ".conf"
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

                                    sample_size += 1
                                    if sample_size == gpu_ids_size + 1:
                                        break
                                        
                                    # device id is optional parameter
                                    if device_id_f > -1:
                                        f.write("  peer_device_id:" + str(device_id_f) + "\n")
                                    f.write("  test_bandwidth: " + str(test_bandwidth_f) + "\n")
                                    f.write("  bidirectional: " + str(bidirectional_f) + "\n")
                                    f.write("  parralel: " + str(parralel_f) + "\n")
    #if counter == total_iterations:    
    if counter > 0:
        break

#gpu_ids_size = len(gpu_ids)
#sample_size = 0
#while True:
    #print "Sample size is %d" % (sample_size)
    #sample_gpus = sample(gpu_ids, sample_size)
    #print sample_gpus
    #sample_size += 1
    #if sample_size == gpu_ids_size + 1:
        #break

#for x in gpu_ids:
    #print x

#for x in xrange(1, 11):
    #for y in xrange(1, 11):
        #string = '%d * %d = %d' % (x, y, x*y)
        ##print string
        #f.write(string + "\n")
