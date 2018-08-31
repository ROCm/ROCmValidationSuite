# Regression

Regression is currently implemented for PQT module only.
It comes in the form of a Python script `run_regression.py`.

The script will first create valid configuration files on
`$RVS_BUILD/regression` folder. It is done by invoking `prq_create_conf.py`
script to generate valid configuration files. If you need different tests,
modify the `prq_create_conf.py` script to generate them.

Then, it will iterate through generated files and invoke RVS to specifying also
JSON output and `-d 3` logging level.

Finally, it will iterate over generated JSON output files and search for `ERROR`
string. Results are written into `$RVS_BUILD/regression/regression_res`
file.

Results are written into $RVS_BUILD/regression/


## Environment variables

Before running the run_regression.py you first need to set the following
environment variables for location of RVS source tree and build folders
(ajdust for your particular clone):

    export WB=/work/yourworkfolder
    export RVS=$WB/ROCmValidationSuite
    export RVS_BUILD=$RVS/../build

## Running the script

Just do:

    cd $RVS/regression
    ./run_regression.py

