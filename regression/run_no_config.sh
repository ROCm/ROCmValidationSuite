#!/bin/bash

mv .rvsmodules.config .rvsmodules.config.old
./rvsfail
set retval=$
mv .rvsmodules.config.old .rvsmodules.config
exit $retval

