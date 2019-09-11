#!/bin/bash

# using the following docker image because it's based off centos5 which is ancient enough
# and can be used to install a newer cmake easily
export DOCKER_IMAGE=${DOCKER_IMAGE:-quay.io/pypa/manylinux1_x86_64}
echo $DOCKER_IMAGE
docker run -v `pwd`:/io $DOCKER_IMAGE /io/tools/scripts/build_all_deps.sh

