#!/usr/bin/env python

import os
import mmap
import sys
import json

#print "Number of arguments: ", len(sys.argv)
#print "The arguments are: " , str(sys.argv)

# --------------------
# passed arguments:
# --------------------
#   json log path
# --------------------

json_log_path  = sys.argv[1]

# result test pass/fail
test_result = True

# check json output file
if os.path.isfile(json_log_path):
   f = open(json_log_path)
   # validate json format
   print "check json format"
   try:
      data = json.load(f)
      json_has_res_err = False
      for d in data:
        json_line = d['loglevelname']
        print json_line
        if json_line == 'RESULT':
          print "JSON Found RESULT"
          json_has_res_err = True
          break
        if json_line == 'ERROR ':
          print "JSON Found ERROR"
          json_has_res_err = True
          break
      if json_has_res_err == False:
         print "JSON No found RESULT/ERROR"
         test_result = False
   except ValueError as e:
      print('Invalid json: %s' % e)
      test_result = False
   f.close()
else:
   print "No file found"
   test_result = False

# return result
if test_result == True:
   print json_log_path + " - PASS"
   sys.exit(0)
else:
   print json_log_path + " - FAIL"
   sys.exit(1)
