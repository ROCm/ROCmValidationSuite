#!/usr/bin/env python3

import subprocess
import os
import mmap
import sys

from shutil import copyfile

curr_location = subprocess.check_output(["pwd", ""])
curr_location_size = len(curr_location)
curr_location = curr_location[0:curr_location_size-1]
print('curr_location',curr_location)

# set paths to build and ROCmValidationSuite folders
build_location = os.environ['RVS_BUILD']
rocm_location = os.environ['RVS']
# single conf files
single_conf = ['gst_single.conf', 'pqt_single.conf', 'pebb_single.conf',
               'iet_single.conf', 'peqt_single.conf', 'rand_single.conf']
# set output folder and configuration file format
conf_files = []
regression_directory = build_location + "/regression"

# location of configuration files
conf_location = build_location + "/regression/conf/"
if not os.path.exists(conf_location):
    os.makedirs(conf_location)

# copy run and build files to build_location
copyfile("build", build_location + "/build")
copyfile("run", build_location + "/run")
for conf in single_conf:
    if os.path.exists("../rvs/conf/" + conf):
        copyfile("../rvs/conf/" + conf, conf_location + "/" + conf)
        conf_files.append(conf)
    else:
        print('copy of conf file ' + conf + 'from rvs/conf folder')
print('conf_files',conf_files)

# make them executable
os.chdir(build_location)
subprocess.call(["chmod", "+x", "build"])
subprocess.call(["chmod", "+x", "run"])

if not os.path.exists(regression_directory):
    os.makedirs(regression_directory)

# open file for regression result
res = open(regression_directory + "/regression_res", "w")

# count tests and run them
number_of_tests = 0
count_fail = 0
max_tries = 2
while True:
    restart = 0
    num_tries = 0
    confname = conf_files[number_of_tests]
    os.chdir(conf_location)
    print('conf_location',conf_location)
    print('build_location',build_location)
    # run test
    os.chdir(build_location)
    os.system("./run %s %s" % (conf_location + confname , regression_directory + "/log_" + confname + ".txt"))
    
    # check json output
    result_json = regression_directory + "/log_" + confname + ".txt"
    if os.path.isfile(result_json):
        print("Found log file " + confname[-5] + "\n")
        f = open(result_json)
        s = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        if s.find('INFO') != -1:
            print("Found INFO")
        elif s.find('RESULT') != -1:
            print("Found RESULT")
        elif s.find('ERROR') != -1:
            print("Found ERROR")
            res.write("FAIL: Testname " + confname + " not ERROR field" + "\n")
            count_fail = count_fail + 1
        else:
            if (num_tries < max_tries):
                restart = 1
                num_tries = num_tries + 1
            else:
                restart = 0
                if s.find('No GPU/peer combination matches criteria from the test configuration') != -1:
                    res.write("WARN: Testname " + confname + " : No GPU/peer - maybe a infrastucture issue\n")
                else:
                    res.write("FAIL: Testname " + confname + " does not have RESULT field" + "\n")
                    count_fail = count_fail + 1
    else:
        print("No json file with number  : " + str(number_of_tests) + "\n")
    # should we restart the test
    if (restart == 0):
        if (number_of_tests < len(conf_files)-1):
            number_of_tests = number_of_tests + 1
            print('number_of_tests',number_of_tests)
        else:
            print('All conf files are run is done, Exiting now')
            break

# finish output
res.write("\n")
res.write("REGRESSION_RESULTS: passing " + str(number_of_tests - count_fail) + " / " + str(number_of_tests) + "\n")
res.close()
