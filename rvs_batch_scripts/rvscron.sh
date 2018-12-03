#!/bin/bash


export RVS_BATCH_SCRIPTS=/work/rvs_batch_scripts
export RVS_BATCH_BUILD=/work/rvs_batch_build

rm -rf $RVS_BATCH_BUILD
mkdir $RVS_BATCH_BUILD

# run Ubuntu batch builds
cd $RVS_BATCH_SCRIPTS
$RVS_BATCH_SCRIPTS/rvs_ubuntu.sh

