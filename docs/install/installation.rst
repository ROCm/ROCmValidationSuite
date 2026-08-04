.. meta::
   :description lang=en: Install ROCm Validation Suite (RVS)
   :keywords: rocm, core, sdk, rvs, validation, suite, install

*****************************
Install ROCm Validation Suite
*****************************

ROCm Validation Suite (RVS) is supported on AMD Instinct and Radeon GPUs
supported by ROCm. See the `ROCm compatibility matrix
<https://rocm.docs.amd.com/en/docs-7.14.0/compatibility/compatibility-matrix.html>`__ for support information.

For advanced workflows, source builds, or custom configurations, see
`<https://github.com/ROCm/ROCmValidationSuite#rocmvalidationsuite>`__.

.. note::
   TransferBench is now a part of the ROCm Validation Suite and is installed with it.
   See the `TransferBench documentation <https://rocm.docs.amd.com/projects/TransferBench/en/latest/>`__ for more information.

Prerequisites
=============

Install the ROCm Core SDK before installing RVS.

For instructions, see :doc:`Install AMD ROCm <rocm:install/rocm>`. Use the
selector panel on that page to view instructions appropriate for your system
environment.

ROCm installation path
======================

The ROCm installation path depends on what installation method was selected during
ROCm installation. After installation, the ROCm Core SDK will be deployed to this
path location and contains all the core ROCm directories such as ``bin``, ``include``,
and ``lib``. When configuring the environment for RVS, you must set the
``ROCM_INSTALL_PATH`` environment variable to the ROCm core installation directory
``<rocm-core-path>`` based on the installation method used:

.. tab-set::

   .. tab-item:: Package manager

      .. code-block:: bash

         ROCM_INSTALL_PATH=/opt/rocm/core-7.14             # <rocm-core-path> = /opt/rocm/core-7.14

   .. tab-item:: pip

      Use within active python virtual environment:

      .. code-block:: bash

         ROCM_INSTALL_PATH=$(rocm-sdk path --root)         # <rocm-core-path> = rocm-sdk path of core installation

   .. tab-item:: Tarball

      .. code-block:: bash

         ROCM_INSTALL_PATH=<path>/therock-tarball/install  # <rocm-core-path> = <path> to default "therock-tarball" installation

   .. tab-item:: Runfile

      .. code-block:: bash

         ROCM_INSTALL_PATH=/opt/rocm/core-7.14             # <rocm-core-path> = default /opt/rocm/core-7.14 installation (no target=)
         ROCM_INSTALL_PATH=<path>/rocm/core-7.14           # <rocm-core-path> = target=<path>

Package manager installation
=============================

Use the following steps to install RVS using your distribution's package manager
on top of the ROCm Core SDK.

1. Register the RVS repository.

   .. tab-set::

      .. tab-item:: Ubuntu
         :sync: ubuntu

         .. code-block:: bash

            sudo mkdir --parents --mode=0755 /etc/apt/keyrings
            wget https://repo.amd.com/rocm/packages-multi-arch/gpg/rocm.gpg -O - | gpg --dearmor | sudo tee /etc/apt/keyrings/amdrocm.gpg > /dev/null
            sudo tee /etc/apt/sources.list.d/rvs.list << EOF
            deb [arch=amd64 signed-by=/etc/apt/keyrings/amdrocm.gpg] https://repo.amd.com/rocm/rvs/packages/deb/ stable main
            EOF

            sudo apt update

      .. tab-item:: RHEL
         :sync: rhel

         .. code-block:: bash

            sudo tee /etc/yum.repos.d/rvs.repo <<EOF
            [rvs]
            name=ROCm Validation Suite
            baseurl=https://repo.amd.com/rocm/rvs/packages/rpm/x86_64/
            enabled=1
            gpgcheck=1
            gpgkey=https://repo.amd.com/rocm/packages-multi-arch/gpg/rocm.gpg
            priority=50
            EOF
            sudo dnf clean all

2. Install the RVS package.

   .. tab-set::

      .. tab-item:: Ubuntu
         :sync: ubuntu

         .. code-block:: bash

            sudo apt install amdrocm7-rvs

      .. tab-item:: RHEL
         :sync: rhel

         .. code-block:: bash

            sudo dnf install amdrocm7-rvs

3. Complete the following post-installation steps.

   Use the following commands to update your shell configuration file
   (``~/.bashrc`` or ``~/.profile``) and add ROCm to your PATH.

   a. Set the ROCm installation path based on the ROCm installation method
      and ``<rocm-core-path>``:

      .. code-block:: bash

         ROCM_INSTALL_PATH=<rocm-core-path>  # ie. /opt/rocm/core-7.14

   b. Configure your environment.

      .. tab-set::

         .. tab-item:: System-wide

            .. code-block:: bash

               sudo tee /etc/profile.d/set-rvs-env.sh << EOF
               export EXTRAS_PATH=/opt/rocm/extras-7
               export ROCM_PATH=$ROCM_INSTALL_PATH
               export PATH=\$EXTRAS_PATH/bin:\$ROCM_PATH/bin:\$PATH
               export LD_LIBRARY_PATH=\$EXTRAS_PATH/lib:\$ROCM_PATH/lib:\$ROCM_PATH/lib/llvm/lib:\$LD_LIBRARY_PATH
               EOF

               sudo chmod +x /etc/profile.d/set-rvs-env.sh
               source /etc/profile.d/set-rvs-env.sh

         .. tab-item:: User

            .. code-block:: bash

               tee --append ~/.bashrc << EOF
               export EXTRAS_PATH=/opt/rocm/extras-7
               export ROCM_PATH=$ROCM_INSTALL_PATH
               export PATH=\$EXTRAS_PATH/bin:\$ROCM_PATH/bin:\$PATH
               export LD_LIBRARY_PATH=\$EXTRAS_PATH/lib:\$ROCM_PATH/lib:\$ROCM_PATH/lib/llvm/lib:\$LD_LIBRARY_PATH
               EOF

               source ~/.bashrc

4. Verify your installation.

   .. code-block:: bash

      rvs -g

.. note::
   The ROCm repositories must be set up before installing RVS. This repository
   setup is part of the ROCm Core SDK installation.

Package manager uninstallation
=============================

1. Remove installed packages.

   .. tab-set::

      .. tab-item:: Ubuntu
         :sync: ubuntu

         .. code-block:: bash

            sudo apt autoremove amdrocm7-rvs

      .. tab-item:: RHEL
         :sync: rhel

         .. code-block:: bash

            sudo dnf remove amdrocm7-rvs

2. Remove RVS repositories.

   .. tab-set::

      .. tab-item:: Ubuntu
         :sync: ubuntu

         .. code-block:: bash

            # Remove RVS repositories
            sudo rm /etc/apt/sources.list.d/rvs.list

            # Clear the cache and clean the system
            sudo rm -rf /var/cache/apt/*
            sudo apt clean all
            sudo apt update

      .. tab-item:: RHEL
         :sync: rhel

         .. code-block:: bash

            # Remove RVS repositories
            sudo rm /etc/yum.repos.d/rvs.repo

            # Clear the cache and clean the system
            sudo rm -rf /var/cache/dnf
            sudo dnf clean all

3. Remove RVS environment configuration.

   .. tab-set::

      .. tab-item:: System-wide

         If you opted for a system-wide setup during the installation
         process, remove the RVS environment variables.

         .. code-block:: bash

            sudo rm -f /etc/profile.d/set-rvs-env.sh

      .. tab-item:: User

         If you opted for a user-specific setup during the installation
         process, remove the RVS environment configuration block from
         your shell configuration file (``~/.bashrc`` or ``~/.profile``).

Tarball installation
====================

Use the following steps to install RVS using a tarball on top of the ROCm Core SDK.

1. Install system dependencies.

   .. tab-set::

      .. tab-item:: Ubuntu
         :sync: ubuntu

         .. code-block:: bash

            sudo apt install libpci3 libnuma1

      .. tab-item:: RHEL
         :sync: rhel

         .. code-block:: bash

            sudo dnf install pciutils-libs numactl-libs

2. Download the RVS tarball.

   .. code-block:: bash

      wget https://repo.amd.com/rocm/rvs/tarball/amdrocm7-rvs-1.5.122-579-Linux.tar.gz

3. Extract the tarball to the ROCm Extras location.

   RVS is part of the ROCm Extras set of tools that work with the ROCm Core SDK.
   The ROCm Extras location (``EXTRAS_INSTALL_PATH``) can be set to a custom
   location ``<extras-path>``, but is typically set based on the ROCm installation
   method used.

   .. code-block:: bash

      EXTRAS_INSTALL_PATH=<extras-path>  # ie. <extras-path> = path to ROCm extras for RVS extract

      sudo mkdir -p $EXTRAS_INSTALL_PATH
      sudo tar -xzf amdrocm7-rvs-1.5.122-579-Linux.tar.gz -C $EXTRAS_INSTALL_PATH

   **Recommended:** Set ``EXTRAS_INSTALL_PATH`` to a location within the root
   install directory for the ROCm Core SDK.

   For example, if you installed the ROCm Core SDK using your Linux
   distribution's package manager:

   .. code-block:: bash

      sudo mkdir -p /opt/rocm/extras-7
      sudo tar -xzf amdrocm7-rvs-1.5.122-579-Linux.tar.gz -C /opt/rocm/extras-7

4. Complete the following post-installation steps.

   Use the following commands to update your shell configuration file
   (``~/.bashrc`` or ``~/.profile``) and add Extras and ROCm to your PATH.

   a. Set the ROCm installation path based on the ROCm installation method
      and ``<rocm-core-path>``:

      .. code-block:: bash

         ROCM_INSTALL_PATH=<rocm-core-path>  # ie. /opt/rocm/core-7.14

   b. Configure your environment.

      .. tab-set::

         .. tab-item:: System-wide

            .. tab-set::

               .. tab-item:: Ubuntu
                  :sync: ubuntu

                  .. code-block:: bash

                     sudo tee /etc/profile.d/set-rvs-env.sh << EOF
                     export EXTRAS_PATH=$EXTRAS_INSTALL_PATH
                     export ROCM_PATH=$ROCM_INSTALL_PATH
                     export PATH=\$EXTRAS_PATH/bin:\$ROCM_PATH/bin:\$PATH
                     export LD_LIBRARY_PATH=\$EXTRAS_PATH/lib:\$ROCM_PATH/lib:\$ROCM_PATH/lib/llvm/lib:\$LD_LIBRARY_PATH
                     EOF

                     sudo chmod +x /etc/profile.d/set-rvs-env.sh
                     source /etc/profile.d/set-rvs-env.sh

               .. tab-item:: RHEL
                  :sync: rhel

                  .. code-block:: bash

                     sudo tee /etc/profile.d/set-rvs-env.sh << EOF
                     export EXTRAS_PATH=/opt/rocm/extras-7
                     export ROCM_PATH=$ROCM_INSTALL_PATH
                     export PATH=\$EXTRAS_PATH/bin:\$ROCM_PATH/bin:\$PATH
                     export LD_LIBRARY_PATH=\$EXTRAS_PATH/lib:\$ROCM_PATH/lib:\$ROCM_PATH/lib/llvm/lib:\$LD_LIBRARY_PATH
                     EOF

                     sudo chmod +x /etc/profile.d/set-rvs-env.sh
                     source /etc/profile.d/set-rvs-env.sh

         .. tab-item:: User

            .. tab-set::

               .. tab-item:: Ubuntu
                  :sync: ubuntu

                  .. code-block:: bash

                     tee --append ~/.bashrc << EOF
                     export EXTRAS_PATH=$EXTRAS_INSTALL_PATH
                     export ROCM_PATH=$ROCM_INSTALL_PATH
                     export PATH=\$EXTRAS_PATH/bin:\$ROCM_PATH/bin:\$PATH
                     export LD_LIBRARY_PATH=\$EXTRAS_PATH/lib:\$ROCM_PATH/lib:\$ROCM_PATH/lib/llvm/lib:\$LD_LIBRARY_PATH
                     EOF

                     source ~/.bashrc

               .. tab-item:: RHEL
                  :sync: rhel

                  .. code-block:: bash

                     tee --append ~/.bashrc << EOF
                     export EXTRAS_PATH=/opt/rocm/extras-7
                     export ROCM_PATH=$ROCM_INSTALL_PATH
                     export PATH=\$EXTRAS_PATH/bin:\$ROCM_PATH/bin:\$PATH
                     export LD_LIBRARY_PATH=\$EXTRAS_PATH/lib:\$ROCM_PATH/lib:\$ROCM_PATH/lib/llvm/lib:\$LD_LIBRARY_PATH
                     EOF

                     source ~/.bashrc

5. Verify your installation.

   .. code-block:: bash

      rvs -g

Tarball uninstalling
====================

1. Remove the installation directory.

   .. important::
      The following command assumes you're working with the
      ``EXTRAS_INSTALL_PATH`` directory set to ``/opt/rocm/extras-7``. If you
      chose a different directory name when installing RVS, adjust the command
      accordingly.

   .. code-block:: bash

      sudo rm -rf /opt/rocm/extras-7

2. Remove RVS environment configuration.

   .. tab-set::

      .. tab-item:: System-wide

         If you opted for a system-wide setup during the installation process,
         remove the RVS environment variables.

         .. code-block:: bash

            sudo rm -f /etc/profile.d/set-rvs-env.sh

      .. tab-item:: User

         If you opted for a user-specific setup during the installation
         process, remove the RVS environment configuration block from your
         shell configuration file (``~/.bashrc`` or ``~/.profile``).