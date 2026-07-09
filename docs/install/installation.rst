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

For instructions, see `Install AMD ROCm
<https://rocm.docs.amd.com/en/docs-7.14.0/install/rocm.html?fam=all&i=pkgman>`__. Use the
selector panel on that page to view instructions appropriate for your system
environment.

Install RVS using tarball
-------------------------

Use the following steps to install RVS using tarball on top of the ROCm Core SDK.

1. Install system dependencies using your Linux distribution's package manager.

   .. tab-set::

      .. tab-item:: Ubuntu

         .. code-block:: bash

            sudo apt install libpci3 libgomp1

      .. tab-item:: RHEL

         .. code-block:: bash

            sudo dnf install pciutils-libs libgomp

2. Download the RVS tarball.

   .. code-block:: bash

      wget https://repo.amd.com/rocm/rvs/tarball/amdrocm7-rvs-1.4.21-288-Linux.tar.gz

3. Extract the tarball to the ROCm Extras location. Set ``ROCM_PATH`` to your
   ROCm Core SDK location, which varies depending on how you installed it. For
   example, if you installed the ROCm Core SDK using your Linux distribution's
   package manager:

   .. code-block:: bash

      export ROCM_PATH=/opt/rocm
      sudo mkdir -p $ROCM_PATH/extras-7
      sudo tar -xzf amdrocm7-rvs-1.4.21-288-Linux.tar.gz -C $ROCM_PATH/extras-7

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

Install RVS using native packages
---------------------------------

Use the following steps to install RVS using native DEB or RPM packages.

1. Create a working directory and download the package.

   .. tab-set::

      .. tab-item:: Ubuntu (DEB)

         .. code-block:: bash

            mkdir -p ~/RVS && cd ~/RVS
            wget https://d22tya8uodfbu6.cloudfront.net/release/rvs/deb/amdrocm7-rvs_1.5.118-548_amd64.deb

      .. tab-item:: RHEL (RPM)

         .. code-block:: bash

            mkdir -p ~/rpm_rel && cd ~/rpm_rel
            wget https://d22tya8uodfbu6.cloudfront.net/release/rvs/rpm/amdrocm7-rvs-1.5.118-548.el8.x86_64.rpm

2. Install the package.

   .. tab-set::

      .. tab-item:: Ubuntu (DEB)

         .. code-block:: bash

            sudo dpkg -i amdrocm7-rvs_1.5.118-548_amd64.deb

      .. tab-item:: RHEL (RPM)

         .. code-block:: bash

            sudo rpm -ivh amdrocm7-rvs-1.5.118-548.el8.x86_64.rpm

3. Set up the environment variables.

   .. code-block:: bash

      export RVS_PATH=/opt/rocm/extras-7
      export ROCM_PATH=/opt/rocm/core-7.14
      export PATH=$RVS_PATH/bin:$ROCM_PATH/bin:$PATH
      export LD_LIBRARY_PATH=$RVS_PATH/lib:/opt/rocm/lib:$ROCM_PATH/lib:$ROCM_PATH/llvm/lib

4. Verify your installation and run a test configuration.

   .. code-block:: bash

      $RVS_PATH/bin/rvs -g
      $RVS_PATH/bin/rvs -c $RVS_PATH/share/rocm-validation-suite/conf/gfx1201/iet_single.conf -d 3
