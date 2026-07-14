.. meta::
   :description lang=en: Install ROCm Validation Suite (RVS)
   :keywords: rocm, core, sdk, rvs, validation, suite, install

*****************************
Install ROCm Validation Suite
*****************************

ROCm Validation Suite (RVS) is supported on AMD Instinct and Radeon GPUs
supported by ROCm. See the `ROCm compatibilty matrix
<https://rocm.docs.amd.com/en/docs-7.14.0/compatibility/compatibility-matrix.html>`__ for support information.

For advanced workflows, source builds, or custom configurations, see
`<https://github.com/ROCm/ROCmValidationSuite#rocmvalidationsuite>`__.

Install RVS on Linux
====================

ROCm Validation Suite (RVS) is a ROCm Extra requiring the ROCm Core SDK to be
installed.

Install the ROCm Core SDK
-------------------------

For instructions, see `Install AMD ROCm <https://rocm.docs.amd.com/en/docs-7.14.0/install/rocm.html?fam=all&i=pkgman>`__. Use the
selector panel on that page to view instructions appropriate for your system
environment.

Install RVS using package manager
----------------------------------

Use the following steps to install RVS using package manager on top of the ROCm Core SDK.

1. Download the RVS package.

   .. tab-set::

      .. tab-item:: Ubuntu

         .. code-block:: bash

            wget https://repo.amd.com/rocm/rvs/deb/amdrocm7-rvs_1.5.122-579_amd64.deb

      .. tab-item:: RHEL

         .. code-block:: bash

            wget https://repo.amd.com/rocm/rvs/rpm/amdrocm7-rvs-1.5.122-579.el8.x86_64.rpm

2. Install the RVS package.

   .. tab-set::

      .. tab-item:: Ubuntu

         .. code-block:: bash

            sudo apt install ./amdrocm7-rvs_1.5.122-579_amd64.deb

      .. tab-item:: RHEL

         .. code-block:: bash

            sudo dnf install ./amdrocm7-rvs-1.5.122-579.el8.x86_64.rpm

3. Complete the following post-installation step to set up your environment.
   Set ``ROCM_PATH`` to your ROCm Core SDK location.

   .. tab-set::

      .. tab-item:: User setup

         .. code-block:: bash

            tee -a ~/.bashrc << EOF
            export ROCM_PATH=/opt/rocm
            export PATH=\$ROCM_PATH/extras-7/bin:\$ROCM_PATH/bin:\$PATH
            export LD_LIBRARY_PATH=\$ROCM_PATH/extras-7/lib:\$ROCM_PATH/lib:\$ROCM_PATH/lib/llvm/lib:\$LD_LIBRARY_PATH
            EOF

            source ~/.bashrc

      .. tab-item:: System-wide setup

         .. code-block:: bash

            sudo tee /etc/profile.d/set-rocm-env.sh << EOF
            export ROCM_PATH=/opt/rocm
            export PATH=\$ROCM_PATH/extras-7/bin:\$ROCM_PATH/bin:\$PATH
            export LD_LIBRARY_PATH=\$ROCM_PATH/extras-7/lib:\$ROCM_PATH/lib:\$ROCM_PATH/lib/llvm/lib:\$LD_LIBRARY_PATH
            EOF

            sudo chmod +x /etc/profile.d/set-rocm-env.sh
            source /etc/profile.d/set-rocm-env.sh

4. Verify your installation.

   .. code-block:: bash

      rvs -h

.. note::
   The ROCm repositories must be setup as part of RVS package manager installation. This repository setup is part of the ROCm Core SDK installation.

Install RVS using tarball
--------------------------

Use the following steps to install RVS using a tarball on top of the ROCm Core SDK.

1. Install system dependencies.

   .. tab-set::

      .. tab-item:: Ubuntu

         .. code-block:: bash

            sudo apt install libpci3 libnuma1

      .. tab-item:: RHEL

         .. code-block:: bash

            sudo dnf install pciutils-libs numactl-libs

2. Download the RVS tarball.

   .. code-block:: bash

      wget https://repo.amd.com/rocm/rvs/tarball/amdrocm7-rvs-1.5.122-579-Linux.tar.gz

3. Extract the tarball to the ROCm Extras location. Set ``ROCM_PATH`` to your
   ROCm Core SDK location, which varies depending on how you installed it. For
   example, if you installed the ROCm Core SDK using your Linux distribution's
   package manager:

   .. code-block:: bash

      sudo mkdir -p /opt/rocm/extras-7
      sudo tar -xzf amdrocm7-rvs-1.5.122-579-Linux.tar.gz -C /opt/rocm/extras-7

4. Complete the following post-installation step to set up your environment.
   Set ``ROCM_PATH`` to your ROCm Core SDK location.

   .. tab-set::

      .. tab-item:: User setup

         .. code-block:: bash

            tee -a ~/.bashrc << EOF
            export ROCM_PATH=/opt/rocm
            export PATH=\$ROCM_PATH/extras-7/bin:\$ROCM_PATH/bin:\$PATH
            export LD_LIBRARY_PATH=\$ROCM_PATH/extras-7/lib:\$ROCM_PATH/lib:\$ROCM_PATH/lib/llvm/lib:\$LD_LIBRARY_PATH
            EOF

            source ~/.bashrc

      .. tab-item:: System-wide setup

         .. code-block:: bash

            sudo tee /etc/profile.d/set-rocm-env.sh << EOF
            export ROCM_PATH=/opt/rocm
            export PATH=\$ROCM_PATH/extras-7/bin:\$ROCM_PATH/bin:\$PATH
            export LD_LIBRARY_PATH=\$ROCM_PATH/extras-7/lib:\$ROCM_PATH/lib:\$ROCM_PATH/lib/llvm/lib:\$LD_LIBRARY_PATH
            EOF

            sudo chmod +x /etc/profile.d/set-rocm-env.sh
            source /etc/profile.d/set-rocm-env.sh

5. Verify your installation.

   .. code-block:: bash

      rvs -h
