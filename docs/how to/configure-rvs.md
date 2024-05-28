
# Configuration Files

The ROCm Validation Suite (RVS) tool allows users to indicate a configuration file, adhering to the YAML 1.2 specification, which details the validation tests to run and the
expected results of a test, benchmark or configuration check.

The configuration file used for an execution is specified using the `--config` option. The default configuration file used for a run is `rvs.conf`, which will include default
values for all defined tests, benchmarks and configurations checks, as well as device specific configuration values. The format of the configuration files
determines the order in which actions are executed, and can provide the number of times the test will be executed as well.

Configuration file is, in YAML terms, mapping of 'actions' keyword into sequence of action items. Action items are themselves YAML keyed lists. Each list consists of several _key:value_ pairs. Some keys may have values which
are keyed lists themselves (nested mappings).

Action item (or action for short) uses keys to define nature of validation test to be performed. Each action has some common keys -- like 'name', 'module', 'deviceid' -- and test specific keys which depend on the module being used.

An example of a RVS configuration file is given here:

```
    actions:
    - name: action_1
      device: all
      module: gpup
      properties:
        mem_banks_count:
      io_links-properties:
        version_major:
    - name: action_2
      module: gpup
      device: all
      properties:
        mem_banks_count:
    - name: action_3
    ...
```

## Common Configuration Keys

Common configuration keys applicable to most modules are summarized in the following table.

| Config Key | Type                 |  Description                                                                                                                                                                                                                                                                                                                                                                                             |
|------------|----------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| name       | String               | The name of the defined action.                                                                                                                                                                                                                                                                                                                                                                          |
| device     | Collection of String | This is a list of device indexes (gpu ids), or the keyword “all”. The defined actions will be executed on the specified device, as long as the action targets a device specifically (some are platform actions). If an invalid device id value or no value is specified the tool will report that the device was not found and terminate execution, returning an error regarding the configuration file. |
| deviceid   | Integer              | This is an optional parameter, but if specified it restricts the action to a specific device type corresponding to the deviceid.                                                                                                                                                                                                                                                                         |
| parallel   | Bool                 | If this key is false, actions will be run on one device at a time, in the order specified in the device list, or the natural ordering if the device value is “all”. If this parameter is true, actions will be run on all specified devices in parallel. If a value isn’t specified the default value is false.                                                                                          |
| count      | Integer              | This specifies number of times to execute the action. If the value is 0, execution will continue indefinitely. If a value isn’t specified the default is 1. Some modules will ignore this parameter.                                                                                                                                                                                                     |
| wait       | Integer              | This indicates how long the test should wait between executions, in milliseconds. Some modules will ignore this parameter. If the count key is not specified, this key is ignored. duration Integer This parameter overrides the count key, if specified. This indicates how long the test should run, given in milliseconds. Some modules will ignore this parameter.                                   |



### Command Line Options

Command line options are summarized in the table below:

| Config Key | Type                 |  Description                                                                                                                                                                                                                                                                                                                                                                                             |
|------------|----------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| name       | String               | The name of the defined action.                                                                                                                                                                                                                                                                                                                                                                          |
| device     | Collection of String | This is a list of device indexes (gpu ids), or the keyword “all”. The defined actions will be executed on the specified device, as long as the action targets a device specifically (some are platform actions). If an invalid device id value or no value is specified the tool will report that the device was not found and terminate execution, returning an error regarding the configuration file. |
| deviceid   | Integer              | This is an optional parameter, but if specified it restricts the action to a specific device type corresponding to the deviceid.                                                                                                                                                                                                                                                                         |
| parallel   | Bool                 | If this key is false, actions will be run on one device at a time, in the order specified in the device list, or the natural ordering if the device value is “all”. If this parameter is true, actions will be run on all specified devices in parallel. If a value isn’t specified the default value is false.                                                                                          |
| count      | Integer              | This specifies number of times to execute the action. If the value is 0, execution will continue indefinitely. If a value isn’t specified the default is 1. Some modules will ignore this parameter.                                                                                                                                                                                                     |
| wait       | Integer              | This indicates how long the test should wait between executions, in milliseconds. Some modules will ignore this parameter. If the count key is not specified, this key is ignored. duration Integer This parameter overrides the count key, if specified. This indicates how long the test should run, given in milliseconds. Some modules will ignore this parameter.                                   |


