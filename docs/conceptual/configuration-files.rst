.. meta::
  :description: rocm validation suite documentation 
  :keywords: rocm validation suite, ROCm, documentation

.. _configuration:



Configuration files
----------------------

The RVS tool will allow the user to indicate a configuration file, adhering to the YAML 1.2 specification, which details the validation tests to run and the
expected results of a test, benchmark or configuration check.

The configuration file used for an execution is specified using the `--config` option. The default configuration file used for a run is `rvs.conf`, which will include default
values for all defined tests, benchmarks and configurations checks, as well as device specific configuration values. The format of the configuration files
determines the order in which actions are executed, and can provide the number of times the test will be executed as well.

Configuration file is, in YAML terms, mapping of 'actions' keyword into sequence of action items. Action items are themselves YAML keyed lists. Each
list consists of several _key:value_ pairs. Some keys may have values, which are keyed lists themselves (nested mappings).

Action item (or action for short) uses keys to define nature of validation test to be performed. Each action has some common keys -- like 'name', 'module',
'deviceid' -- and test specific keys which depend on the module being used.

An example of RVS configuration file is given here:



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
  
