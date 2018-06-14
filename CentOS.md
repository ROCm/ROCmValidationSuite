# CentOS docker image

1. start the docker image with the following command

        sudo docker run --privileged=true -it --network=host --device=/dev/kfd --device=/dev/dri --group-add video -v /home/user1:/home/root 9eb1aca8b124

2. enable devtoolset-7 environment

        scl enable devtoolset-7 bash

3. clone the RVS repository and compile


