#!/usr/bin/env python3

import subprocess  # nosec B404 # argument lists only; no shell
import os
import mmap
import sys

# global variables
test_output_file_name = "tmp_log_result.txt"

#print("Number of arguments: ", len(sys.argv))
#print("The arguments are: " , str(sys.argv))

# --------------------
# passed arguments:
# --------------------
#   rvs bin path
#   rvs path
#   conf
#   num_runs
#   debug level
# --------------------

bin_path       = sys.argv[1]
rvs_path       = sys.argv[2]
conf_name      = sys.argv[3]
num_runs       = int(sys.argv[4])
debug_level    = sys.argv[5] # only 0,1,2,3,4,5

# check input values
if num_runs < 2:
   print("number of runs (argument 4) should be at least 2")
   sys.exit(1)

if not debug_level in ['0', '1', '2', '3', '4', '5']:
   print("debug_level (argument 5) should be inside true /false")
   sys.exit(1)

# ./multi_run_and_check_json.py /work/igorhdl/ROCm2/build/bin /work/igorhdl/ROCm2/ROCmValidationSuite  /work/igorhdl/ROCm2/ROCmValidationSuite/rvs/conf/rand_pbqt0.conf 5 3

# get current location
curr_location = os.getcwd()
print(curr_location)

# run test commands
result_json = bin_path + "/" + test_output_file_name

rvs_cmd = [os.path.abspath(os.path.join(bin_path, "rvs")), "-d", debug_level, "-c", conf_name, "-l", result_json, "-j"]
rvs_cmd_append = rvs_cmd + ["-a"]

# start running tests
os.chdir(rvs_path + "/regression")

for i in range(0, num_runs):
   print("Iteration %d" % (i))
   if i == 0:
      tst_result = subprocess.call(rvs_cmd)  # nosec B603
   else:
      tst_result = subprocess.call(rvs_cmd_append)  # nosec B603
   # also check test result
   print("Test result is : %s" % (tst_result))
   if tst_result != 0:
      print("Test is expected to pass with value 0, but return value is %s" %(tst_result))
      print(conf_name + " - FAIL")
      sys.exit(1)

os.chdir(curr_location)

# result test pass/fail
test_result = True

json_checker = os.path.join(curr_location, "check_json_file.py")
json_result = subprocess.call([json_checker, result_json])  # nosec B603
if json_result == 1:
   print("Json file is invalid")
   test_result = False

# return result
if test_result == True:
   print(conf_name + " - PASS")
   sys.exit(0)
else:
   print(conf_name + " - FAIL")
   sys.exit(1)
