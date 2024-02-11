
.. meta::
  :description: rocm validation suite documentation 
  :keywords: rocm validation suite, ROCm, documentation

.. _architecture:


ROCm Validation Suite Architecture
***********************************

RVS is implemented as a set of modules, each implementing a particular test functionality. Modules are invoked from one central place (aka Launcher), responsible for reading input (command line and test configuration file), loading and running appropriate modules, and providing test output. 

RVS architecture is built around the concept of Linux-shared objects, thus allowing for the easy addition of new modules in the future.
