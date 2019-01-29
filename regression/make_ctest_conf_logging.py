#!/usr/bin/env python
"""
###################################################################################
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

# This script only creates valid combinations, invalid ones will be created as special cases
from __future__ import print_function

import os, fnmatch
import itertools
import sys

from random import sample

# global variables
src_root = sys.argv[1]
bin_folder = sys.argv[2]
ttpf = sys.argv[3]
cmake_file_name = sys.argv[4]
pattern = sys.argv[5]
module_name = sys.argv[6]

family = ""
# print('num args: {}'.format(len(sys.argv)))
if len(sys.argv) > 7:
  family = sys.argv[7]

# RVS build folder
#src_root = os.path.dirname(os.path.realpath(__file__))
#src_root = src_root + "/.."
# print(src_root)

# location of configuration files
conf_location = src_root + "/rvs/conf/"
#cmake_file_location = src_root + "/" + module_name + ".so/" + cmake_file_name
cmake_file_location = cmake_file_name

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

listOfFiles = os.listdir(conf_location)
#pattern = '{}*'.format(module_name)

combos = itertools.product(listOfFiles)

# go through all files
for conf_file in listOfFiles:

    if fnmatch.fnmatch(conf_file, pattern):
#        print('conf_file: {}'.format(conf_file))
        # find tail
        if not family:
          if ttpf == "ttp":
            index = len(module_name)
          else:
            index = len("ttf_" + module_name)
        else:
          if ttpf == "ttp":
            index = len(module_name+"_"+family)
          else:
            index = len("ttf_" + module_name+"_"+family)

        confindex = conf_file.find(".conf")
        tail = conf_file[index+1:confindex]

        # construct test name
        test_name = ""
        if not family:
          test_name = '{}.conf.log.{}.{}'.format(ttpf, module_name, tail)
        else:
          test_name = '{}.conf.log.{}.{}.{}'.format(ttpf, module_name, family, tail)

        cmake_file.write('add_test(NAME {}'.format(test_name))
        print('conf test: {}'.format(test_name))

        cmake_file.write('  WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}\n')
#        cmake_file.write('  COMMAND rvs -d 3 -c conf/{} -j -l {}\n'.format(conf_file, log_file))
        cmake_file.write('  COMMAND {}/regression/run_and_check_test.py {} {} conf/{} true true true {} 3\n'.format(src_root, bin_folder, src_root, conf_file, ttpf))
        cmake_file.write(')\n\n')

cmake_file.close() 
