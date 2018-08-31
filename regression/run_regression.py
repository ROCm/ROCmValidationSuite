#!/usr/bin/env python

import subprocess
import os
import mmap
import sys

from shutil import copyfile

curr_location = subprocess.check_output(["pwd", ""])
curr_location_size = len(curr_location)
curr_location = curr_location[0:curr_location_size-1]
print curr_location

# set paths to build and ROCmValidationSuite folders
build_location = os.environ['RVS_BUILD']
rocm_location = os.environ['RVS']
# set output folder and configuration file format
conf_files = "demofile"
regression_directory = build_location + "/regression"

# location of configuration files
conf_location = build_location + "/regression/conf/"
if not os.path.exists(conf_location):
    os.makedirs(conf_location)

# copy run and build files to build_location
copyfile("build", build_location + "/build")
copyfile("run", build_location + "/run")
# make them executable
os.chdir(build_location)
subprocess.call(["chmod", "+x", "build"])
subprocess.call(["chmod", "+x", "run"])

# create tests
subprocess.call([rocm_location + "/regression/pqt_create_conf.py", ""])

if not os.path.exists(regression_directory):
    os.makedirs(regression_directory)

# open file for regression result
res = open(regression_directory + "/regression_res", "w")

# count tests and run them
number_of_tests = 0
count_fail = 0
max_tries = 20
while True:
    restart = 0
    num_tries = 0
    confname = conf_files + str(number_of_tests)
    filename = confname + ".conf"
    print filename
    os.chdir(conf_location)
    if os.path.isfile(filename):
        print "Found file with name : " + filename + "\n"
    else:
        number_of_tests = number_of_tests - 1
        break
    
    # run test
    os.chdir(build_location)
    os.system("./run %s %s" % (conf_location + confname + ".conf", regression_directory + "/json_" + confname + ".txt"))
    
    # check json output
    result_json = regression_directory + "/json_" + confname + ".txt"
    if os.path.isfile(result_json):
        print "Found json file " + conf_files + "\n"
        f = open(result_json)
        s = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        if s.find('INFO') != -1:
            print "Found INFO"
        if s.find('RESULT') != -1:
            print "Found RESULT"
        else:
            if (num_tries < max_tries):
                restart = 1
                num_tries = num_tries + 1
            else:
                restart = 0
                if s.find('No GPU/peer combination matches criteria from test configuation') != -1:
                    res.write("WARN: Testname " + confname + " : No GPU/peer - maybe a infrastucture issue\n")
                else:
                    res.write("FAIL: Testname " + confname + " does not have RESULT field" + "\n")
                    count_fail = count_fail + 1
        if s.find('ERROR') != -1:
            print "Found ERROR"
            res.write("FAIL: Testname " + confname + " not ERROR field" + "\n")
            count_fail = count_fail + 1
    else:
        print "No json file with number  : " + str(number_of_tests) + "\n"
    # should we restart the test
    if (restart == 0):
        number_of_tests = number_of_tests + 1
        print number_of_tests

# finish output
res.write("\n")
res.write("REGRESSION_RESULTS: passing " + str(number_of_tests - count_fail) + " / " + str(number_of_tests) + "\n")
res.close()
