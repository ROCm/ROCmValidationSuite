#!/usr/bin/env python3

import subprocess  # nosec B404 # argument lists only; no shell
import os
import mmap
import sys

# global variables
test_console_file_name = "tmp_console_result.txt"
test_output_file_name = "tmp_log_result.txt"

#print("Number of arguments: ", len(sys.argv))
#print("The arguments are: " , str(sys.argv))

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
expected_result = sys.argv[7] # only ttp / ttf
debug_level    = sys.argv[8] # only 0,1,2,3,4,5

# check input values
if not console_usage in ['true', 'false']:
   print("console_usage (argument 4) should be inside true /false")
   sys.exit(1)

if not log_usage in ['true', 'false']:
   print("log_usage (argument 5) should be inside true /false")
   sys.exit(1)

if not json_usage in ['true', 'false']:
   print("json_usage (argument 6) should be inside true /false")
   sys.exit(1)

if not expected_result in ['ttp', 'ttf']:
   print("expected_result (argument 7) should be inside true /false")
   sys.exit(1)

if not debug_level in ['0', '1', '2', '3', '4', '5']:
   print("debug_level (argument 8) should be inside true /false")
   sys.exit(1)

# ./run_and_check_test.py /work/igorhdl/ROCm2/build/bin /work/igorhdl/ROCm2/ROCmValidationSuite  /work/igorhdl/ROCm2/ROCmValidationSuite/rvs/conf/rand_pbqt0.conf true true true ttp 3

# ./run_single_test /work/igorhdl/ROCm2/build/bin /work/igorhdl/ROCm2/ROCmValidationSuite/rvs/conf/rand_pbqt0.conf 3 [tmp_output_file.txt|no_log] [true|false] tmp_console_file.txt
# ./run_single_test /work/igorhdl/ROCm2/build/bin /work/igorhdl/ROCm2/ROCmValidationSuite/rvs/conf/rand_pbqt0.conf 3 tmp_output_file.txt true tmp_console_file.txt

# get current location
curr_location = os.getcwd()
print(curr_location)

# run test command
if log_usage == 'true':
   log_path = bin_path + "/" + test_output_file_name
else:
   log_path = "no_log"

os.chdir(rvs_path + "/regression")
run_single_test = os.path.join(rvs_path, "regression", "run_single_test")
tst_result = subprocess.call([  # nosec B603
   run_single_test,
   bin_path,
   conf_name,
   debug_level,
   log_path,
   json_usage,
   bin_path + "/" + test_console_file_name,
])
print("Test result is : %s" % (tst_result))
os.chdir(curr_location)

# check test to pass/fail first
if expected_result == 'ttp' and tst_result != 0:
   print("Test is expected to pass with value 0, but return value is %s" %(tst_result))
   print(conf_name + " - FAIL")
   sys.exit(1)

if expected_result == 'ttf':
   if tst_result == 0:
      print("Test is expected to fail with value different than 0, but return value is %s" %(tst_result))
      print(conf_name + " - FAIL")
      sys.exit(1)
   else:
      print("Test is expected to fail and return value is non 0")
      print(conf_name + " - PASS")
      sys.exit(0)

# result test pass/fail
test_result = True

# check console output
if console_usage == 'true':
   print("console_usage is True")
   result_log = bin_path + "/" + test_console_file_name
   if os.path.isfile(result_log):
      if os.path.getsize(result_log) > 0:
         f = open(result_log)
         s = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
         if s.find(b'RESULT') == -1 and s.find(b'ERROR') == -1:
            print("No found RESULT/ERROR")
            test_result = False
         f.close()
      else:
         print("Empty file")
         test_result = False
   else:
      print("No file found")
      test_result = False

# check json output file
if json_usage == 'true' and log_usage == 'true':
   print("json_usage is True and log_usage is True")
   result_json = bin_path + "/" + test_output_file_name

   json_checker = os.path.join(curr_location, "check_json_file.py")
   json_result = subprocess.call([json_checker, result_json])  # nosec B603
   if json_result == 1:
      print("Json file is invalid")
      test_result = False

# check console output file
else:
   if log_usage == 'true':
      print("log_usage is True")
      result_log = bin_path + "/" + test_output_file_name
      if os.path.isfile(result_log):
         if os.path.getsize(result_log) > 0:
            f = open(result_log)
            s = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            if s.find('RESULT') == -1 and s.find('ERROR') == -1:
               print("No found RESULT/ERROR")
               test_result = False
            f.close()
         else:
            print("Empty file")
            test_result = False
      else:
         print("No file found")
         test_result = False

# return result
if test_result == True:
   print(conf_name + " - PASS")
   sys.exit(0)
else:
   print(conf_name + " - FAIL")
   sys.exit(1)
