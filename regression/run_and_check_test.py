#!/usr/bin/env python

import subprocess
import os
import mmap
import sys
import json

# global variables
test_result_file_name = "tmp_test_result.txt"
test_json_file_name = "tmp_json_result.txt"

#print "Number of arguments: ", len(sys.argv)
#print "The arguments are: " , str(sys.argv)

# --------------------
# passed arguments:
# --------------------
#   rvs bin path
#   rvs path
#   conf
#   log_usage
#   json usage
# --------------------

bin_path    = sys.argv[1]
rvs_path    = sys.argv[2]
conf_name   = sys.argv[3]
log_usage   = sys.argv[4]
json_usage  = sys.argv[5]

# ./run_single_test /work/igorhdl/ROCm2/build/bin /work/igorhdl/ROCm2/ROCmValidationSuite/rvs/conf/rand_pqt0.conf json_tmp.txt tmp.txt

# get current location
curr_location = subprocess.check_output(["pwd", ""])
curr_location_size = len(curr_location)
curr_location = curr_location[0:curr_location_size-1]
print curr_location

# run test (with json)
if json_usage == 'true':
   test_cmd = "./run_single_test %s %s %s %s" % (bin_path, conf_name,  bin_path + "/" + test_json_file_name, bin_path + "/" + test_result_file_name)
else:
   test_cmd = "./run_single_test %s %s %s %s" % (bin_path, conf_name,  "no_json", bin_path + "/" + test_result_file_name)

os.chdir(rvs_path + "/regression")
os.system(test_cmd)
os.chdir(curr_location)

# result test pass/fail
test_result = True

# check log output
if log_usage == 'true':
   print "log_usage is True"
   result_log = bin_path + "/" + test_result_file_name
   if os.path.isfile(result_log):
      f = open(result_log)
      s = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
      if s.find('RESULT') == -1:
         print "No found RESULT"
         test_result = False
      if s.find('ERROR') != -1:
         print "Found ERROR"
         test_result = False
      f.close()
   else:
      print "No file found"
      test_result = False

# check json output
if json_usage == 'true':
   print "json_usage is True"
   result_json = bin_path + "/" + test_json_file_name
   if os.path.isfile(result_json):
      f = open(result_json)
      s = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
      if s.find('RESULT') == -1:
         print "No found RESULT"
         test_result = False
      if s.find('ERROR') != -1:
         print "Found ERROR"
         test_result = False

      # validate json format
      print "check json format"
      try:
         json.load(f)
      except ValueError as e:
         print('Invalid json: %s' % e)
         test_result = False

      f.close()
   else:
      print "No file found"
      test_result = False

# return result
if test_result == True:
   print conf_name + " - PASS"
   exit(0)
else:
   print conf_name + " - FAIL"
   exit(1)
