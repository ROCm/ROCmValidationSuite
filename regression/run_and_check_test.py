#!/usr/bin/env python

import subprocess
import os
import mmap
import sys

# global variables
test_console_file_name = "tmp_console_result.txt"
test_output_file_name = "tmp_log_result.txt"

#print "Number of arguments: ", len(sys.argv)
#print "The arguments are: " , str(sys.argv)

# --------------------
# passed arguments:
# --------------------
#   rvs bin path
#   rvs path
#   conf
#   console usage
#   log usage
#   json usage
#   ttp / ttf
#   debug level
# --------------------

bin_path       = sys.argv[1]
rvs_path       = sys.argv[2]
conf_name      = sys.argv[3]
console_usage  = sys.argv[4] # only true / false
log_usage      = sys.argv[5] # only true / false
json_usage     = sys.argv[6] # only true / false
test_pass_fail = sys.argv[7] # only ttp / ttf
debug_level    = sys.argv[8] # only 0,1,2,3,4,5

# check input values
if not console_usage in ['true', 'false']:
   print "console_usage (argument 4) should be inside true /false"
   sys.exit(1)

if not log_usage in ['true', 'false']:
   print "log_usage (argument 5) should be inside true /false"
   sys.exit(1)

if not json_usage in ['true', 'false']:
   print "json_usage (argument 6) should be inside true /false"
   sys.exit(1)

if not test_pass_fail in ['ttp', 'ttf']:
   print "test_pass_fail (argument 7) should be inside true /false"
   sys.exit(1)

if not debug_level in ['0', '1', '2', '3', '4', '5']:
   print "debug_level (argument 8) should be inside true /false"
   sys.exit(1)

# ./run_and_check_test.py /work/igorhdl/ROCm2/build/bin /work/igorhdl/ROCm2/ROCmValidationSuite  /work/igorhdl/ROCm2/ROCmValidationSuite/rvs/conf/rand_pqt0.conf true true true ttp 3

# ./run_single_test /work/igorhdl/ROCm2/build/bin /work/igorhdl/ROCm2/ROCmValidationSuite/rvs/conf/rand_pqt0.conf 3 [tmp_output_file.txt|no_log] [true|false] tmp_console_file.txt
# ./run_single_test /work/igorhdl/ROCm2/build/bin /work/igorhdl/ROCm2/ROCmValidationSuite/rvs/conf/rand_pqt0.conf 3 tmp_output_file.txt true tmp_console_file.txt

# get current location
curr_location = subprocess.check_output(["pwd", ""])
curr_location_size = len(curr_location)
curr_location = curr_location[0:curr_location_size-1]
print curr_location

# run test command
if log_usage == 'true':
   pass_log = bin_path + "/" + test_output_file_name
else:
   pass_log = "no_log"

test_cmd = "./run_single_test %s %s %s %s %s %s" % (bin_path, conf_name, debug_level, pass_log, json_usage, bin_path + "/" + test_console_file_name)

os.chdir(rvs_path + "/regression")
tst_result = os.system(test_cmd)
print "Test result is : %s" % (tst_result)
os.chdir(curr_location)

# check test to pass/fail first
if test_pass_fail == 'ttp' and tst_result > 0:
   print "Test is expected to pass with value 0, but return value is %s" %(tst_result)
   print conf_name + " - FAIL"
   sys.exit(1)

if test_pass_fail == 'ttf':
   if tst_result == 0:
      print "Test is expected to fail with value different than 0, but return value is %s" %(tst_result)
      print conf_name + " - FAIL"
      sys.exit(1)
   else:
      print "Test is expected to fail and return value is non 0"
      print conf_name + " - PASS"
      sys.exit(0)

# result test pass/fail
test_result = True

# check console output
if console_usage == 'true':
   print "console_usage is True"
   result_log = bin_path + "/" + test_console_file_name
   if os.path.isfile(result_log):
      if os.path.getsize(result_log) > 0:
         f = open(result_log)
         s = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
         if s.find('RESULT') == -1 and s.find('ERROR') == -1:
            print "No found RESULT/ERROR"
            test_result = False
         f.close()
      else:
         print "Empty file"
         test_result = False
   else:
      print "No file found"
      test_result = False

# check json output file
if json_usage == 'true' and log_usage == 'true':
   print "json_usage is True and log_usage is True"
   result_json = bin_path + "/" + test_output_file_name

   json_result = os.system("./check_json_file.py " + result_json)
   if json_result == 1:
      print "Json file is invalid"
      test_result = False

# check console output file
else:
   if log_usage == 'true':
      print "log_usage is True"
      result_log = bin_path + "/" + test_output_file_name
      if os.path.isfile(result_log):
         if os.path.getsize(result_log) > 0:
            f = open(result_log)
            s = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            if s.find('RESULT') == -1 and s.find('ERROR') == -1:
               print "No found RESULT/ERROR"
               test_result = False
            f.close()
         else:
            print "Empty file"
            test_result = False
      else:
         print "No file found"
         test_result = False

# return result
if test_result == True:
   print conf_name + " - PASS"
   sys.exit(0)
else:
   print conf_name + " - FAIL"
   sys.exit(1)
