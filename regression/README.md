## Regression script

This is a simple regression script for testing PQT.
It will call prq_create_conf.py to generate valid configuration files then run each of the generated configurations and report results in the regression_dir folder.
If you need different tests modifty the prq_create_conf.py script to generate them.


## Set environment variables

Before running the run_regression.py you first need to set the following environment variables for location of build and ROCM folders:

    export WB=/work/yourworkfolder
    export RVS=$WB/ROCmValidationSuite
    export RVS_BUILD=$RVS/../build
