# Copyright (c) 2018-2020 Advanced Micro Devices, Inc. All rights reserved.

HERE := $(dir $(lastword $(MAKEFILE_LIST)))

define newline


endef

ifeq ($(CC),cc)
CC=gcc
endif

ifeq ($(CXX),c++)
CXX=g++
endif

ifeq ($(CC),gcc)
GCC_GTE_48 := $(shell expr `gcc -dumpversion | cut -f1,2 -d.` \>= 4.8)
ifeq "$(GCC_GTE_48)" "1"
PEDANTIC_FLAGS=-Wpedantic -pedantic-errors
endif
endif
WARN_FLAGS=-Wall -Wextra $(PEDANTIC_FLAGS) -Wpacked -Wundef
CONLY_WARN=-Wold-style-definition

ifndef DEBUG
C_AND_CXX_FLAGS=-g3 -ggdb -DBASE_FILE_NAME=\"$(<F)\" -DLINUX $(WERROR_FLAG) $(WARN_FLAGS) $(INCLUDE_FLAGS) -O3 -march=native -DNDEBUG
else
C_AND_CXX_FLAGS=-g3 -ggdb -fno-omit-frame-pointer -DBASE_FILE_NAME=\"$(<F)\" -DLINUX $(WARN_FLAGS) $(INCLUDE_FLAGS) -O0 -DDEBUG
endif

CFLAGS=$(C_AND_CXX_FLAGS) -std=gnu11 $(CONLY_WARN)
CXXFLAGS=$(C_AND_CXX_FLAGS) -std=c++11 -I$(HERE).. -fno-strict-aliasing -Wformat -Werror=format-security -fwrapv
LDFLAGS=-Wl,--as-needed -lpciaccess -pthread

%.o: %.c
	$(CC) $(CFLAGS) $(LOCAL_INCLUDES) -c -MMD -o $@ $<

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(LOCAL_INCLUDES) -c -MMD -o $@ $<

%.d: ;
.PRECIOUS: %.d
