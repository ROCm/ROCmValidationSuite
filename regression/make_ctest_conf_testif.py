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
ttpf = sys.argv[1]
cmake_file_name = sys.argv[2]
pattern = sys.argv[3]
module_name = sys.argv[4]

family = ""
# print('num args: {}'.format(len(sys.argv)))
if len(sys.argv) > 5:
  family = sys.argv[5]

# RVS build folder
build_location = os.path.dirname(os.path.realpath(__file__))
build_location = build_location + "/.."
# print(build_location)

# location of configuration files
conf_location = build_location + "/rvs/conf/"
#cmake_file_location = build_location + "/" + module_name + ".so/" + cmake_file_name
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
          test_name = '{}.conf.{}.{}'.format(ttpf, module_name, tail)
        else:
          test_name = '{}.conf.{}.{}.{}'.format(ttpf, module_name, family, tail)

        cmake_file.write('add_test(NAME {}'.format(test_name))
        print('conf test: {}'.format(test_name))

        log_file = conf_file.replace(".conf", ".log")
        log_file = "zz_" + log_file

        if ttpf == "ttp":
          cmake_file.write('  WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}\n')
          cmake_file.write('  COMMAND ${{CMAKE_SOURCE_DIR}}/regression/run_config_testif.sh testif.config ./rvs conf/{}\n'.format(conf_file))
          cmake_file.write(')\n\n')
        else:
          cmake_file.write('  WORKING_DIRECTORY ${RVS_BINTEST_FOLDER}\n')
          cmake_file.write('  COMMAND ${{CMAKE_SOURCE_DIR}}/regression/run_config_testif.sh testif.config ./rvsfail conf/{} \n'.format(conf_file))
          cmake_file.write(')\n\n')

cmake_file.close() 
