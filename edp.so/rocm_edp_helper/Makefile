# Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.

FILE_DIR := $(dir $(lastword $(MAKEFILE_LIST)))

-include ./make/master.mk

.PHONY: all
all: rocm_edp_helper

ROCM_EDP_HELPER_CSRC=$(shell find $(FILE_DIR) -maxdepth 1 -name '*.c')
ROCM_EDP_HELPER_CPPSRC=$(shell find $(FILE_DIR) -maxdepth 1 -name '*.cpp')
ROCM_EDP_HELPER_COBJECTS=$(ROCM_EDP_HELPER_CSRC:.c=.o)
ROCM_EDP_HELPER_CPPOBJECTS=$(ROCM_EDP_HELPER_CPPSRC:.cpp=.o)
ROCM_EDP_HELPER_CDEPS=$(ROCM_EDP_HELPER_COBJECTS:.o=.d)
ROCM_EDP_HELPER_CPPDEPS=$(ROCM_EDP_HELPER_CPPOBJECTS:.o=.d)

rocm_edp_helper: $(ROCM_EDP_HELPER_COBJECTS) $(ROCM_EDP_HELPER_CPPOBJECTS)
	$(CXX) $^ $(LDFLAGS) -o $@

.PHONY: clean
clean:
	$(RM) -rf rocm_edp_helper $(ROCM_EDP_HELPER_COBJECTS) $(ROCM_EDP_HELPER_CDEPS) $(ROCM_EDP_HELPER_CPPOBJECTS) $(ROCM_EDP_HELPER_CPPDEPS)

-include $(ROCM_EDP_HELPER_CDEPS)
-include $(ROCM_EDP_HELPER_CPPDEPS)
