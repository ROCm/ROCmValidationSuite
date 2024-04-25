.. meta::
  :description: ROCm Validation Suite documentation 
  :keywords: ROCm Validation Suite, RVS, ROCm, documentation

********************************************************************
ROCm Validation Suite documentation
********************************************************************

The ROCm Validation Suite (RVS) is a system validation and diagnostics tool for monitoring, stress testing, detecting and troubleshooting issues that affects the functionality and performance of AMD GPU(s) operating in a high-performance/AI/ML computing environment. RVS is enabled using the ROCm software stack on a compatible software and hardware platform.

RVS is a collection of tests, benchmarks and qualification tools each targeting a specific sub-system of the ROCm platform. All of the tools are implemented in software and share a common command line interface. Each set of tests are implemented in a “module” which is a library encapsulating the functionality specific to the tool. The CLI can specify the directory containing modules to use when searching for libraries to load. Each module may have a set of options that it defines and a configuration file that supports its execution.

For different RVS modules and their description, refer to [the documentation on features](./features.md).

For module configuration files description and examples, refer to [the user guide](./ug1main.md).

The documentation is structured as follows:

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

       * :doc:`RVS installation <./install/installation>`

  .. grid-item-card:: API library

    * :doc:`API library <../doxygen/html/files>`
    * :doc:`Functions <../doxygen/html/globals>`
    * :doc:`Data structures <../doxygen/html/annotated>`

  .. grid-item-card:: How to

    * :doc:`Use ROCm Validation Suite <how to/use-rvs>`
    * :doc:`Configure ROCm Validation Suite <how to/configure-rvs>`

  .. grid-item-card:: Conceptual

     * :doc:`ROCm Validation Suite modules <./conceptual/rvs-modules>`


To contribute to the documentation, refer to
`Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the
`Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.

    
    
    

