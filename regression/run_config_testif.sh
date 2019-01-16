#!/bin/bash

mv .rvsmodules.config .rvsmodules.config.old
rm -f .rvsmodules.config
cp $1 .rvsmodules.config
$2 -c $3
set retval=$
rm -f .rvsmodules.config
mv .rvsmodules.config.old .rvsmodules.config
exit $retval

