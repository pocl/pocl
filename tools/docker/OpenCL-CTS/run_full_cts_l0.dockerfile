# Builds PoCL with Level0 support & OpenCL-CTS in a clean ubuntu environment,
# then runs the Full OpenCL-CTS in both normal and SPIR-V input mode,
# using the CTS's python runner script (run_conformance.py).
#
# NOTE: requires deb packages of intel compute runtime in a "NEO.tar" file
# NOTE: check & override if necessary the ARG variables defined below

# Build docker image:
#     docker build [--build-arg NAME=VALUE ...] --network=host -t pocl_full_cts:24.04 -f run_full_cts.dockerfile .
# Create a container from the image:
#     docker create --name pocl_full_ct --memory=56G --memory-swap=56G --ulimit core=0 pocl_full_cts:24.04
# Run the container:
#     docker start pocl_full_ct
# View stdout of the container:
#     docker logs pocl_full_ct
#
# The full CTS logs () will be put into /srv in the container.
# If you want a copy of the full CTS logs, the simplest way is to bind-mount
# a host machine directory onto /srv in the container at create:
#     docker create ... --mount type=bind,src=/tmp/logs,dst=/srv ...
# alternatively, they can be accessed after run by creating an image of the container:
#     docker commit <ID of pocl_full_ct> -> creates a snapshot image of the container, prints the image ID
#     docker run --network=host --rm -it --entrypoint=/bin/bash --mount type=bind,src=/tmp/logs,dst=/mnt <image ID>
# .. in the running container, copy the logs from /srv to /mnt, then exit.


# note that this won't work on Ubuntu 22.04 without some extra work (ocl-icd is too old)
FROM amd64/ubuntu:24.04

# LLVM version to build PoCL with
ARG LLVM_VERSION=18

RUN apt update
RUN apt upgrade -y
RUN apt install -y python3-dev libpython3-dev build-essential ocl-icd-libopencl1 cmake git \
                   pkg-config libclang-${LLVM_VERSION}-dev clang-${LLVM_VERSION} llvm-${LLVM_VERSION} make \
                   ocl-icd-libopencl1 ocl-icd-dev ocl-icd-opencl-dev libhwloc-dev zlib1g zlib1g-dev clinfo dialog \
                   apt-utils libxml2-dev libclang-cpp${LLVM_VERSION}-dev libclang-cpp${LLVM_VERSION} llvm-${LLVM_VERSION}-dev \
                   libpolly-${LLVM_VERSION}-dev spirv-tools spirv-headers llvm-spirv-${LLVM_VERSION} \
                   libllvmspirvlib-${LLVM_VERSION}-dev libzstd-dev libglew-dev libglut-dev wget

# install Intel GPU drivers
ADD NEO.tar /tmp
RUN cd /tmp/NEO && dpkg -i *.deb

# install Intel LevelZero loader + headers
#COPY level-zero_1.17.45+u24.04_amd64.deb level-zero-devel_1.17.45+u24.04_amd64.deb /tmp/
RUN cd /tmp && wget https://github.com/oneapi-src/level-zero/releases/download/v1.17.45/level-zero-devel_1.17.45+u24.04_amd64.deb
RUN cd /tmp && wget https://github.com/oneapi-src/level-zero/releases/download/v1.17.45/level-zero_1.17.45+u24.04_amd64.deb
RUN cd /tmp && dpkg -i level-zero*

RUN clinfo

# make sure only PoCL is picked up by CTS
RUN rm /etc/OpenCL/vendors/*

# required for run_conformance.py
RUN if [ ! -e /usr/bin/python ]; then ln -s /usr/bin/python3 /usr/bin/python ; fi

# setting to 1 enables Wimpy mode. Will make testing much faster but not valid for submission
ARG USE_WIMPY_MODE=1
# GH username / org name where PoCL is cloned
ARG POCL_REMOTE=pocl
# git branch/commit of PoCL to build
ARG POCL_CHECKOUT=9699c15a26805bdb10823e1f06f9e9c478d35ab8
# C/C++ build flags
ARG POCL_BUILD_FLAGS="-O2 -march=native"

RUN cd /opt && git clone https://github.com/$POCL_REMOTE/pocl.git && cd /opt/pocl && git checkout $POCL_CHECKOUT

RUN mkdir /opt/pocl/build && cd /opt/pocl/build
RUN cd /opt/pocl/build && \
    cmake -DWITH_LLVM_CONFIG=/usr/bin/llvm-config-${LLVM_VERSION} -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DENABLE_LEVEL0=1 -DSTATIC_LLVM=1 -DENABLE_HOST_CPU_DEVICES=0 -DENABLE_ICD=1 -DENABLE_EXAMPLES=0 -DENABLE_TESTS=0 \
    -DCMAKE_C_FLAGS_RELWITHDEBINFO="$POCL_BUILD_FLAGS"  -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="$POCL_BUILD_FLAGS" \
    -DCMAKE_INSTALL_PREFIX=/usr -DENABLE_ICD=1 -DPOCL_ICD_ABSOLUTE_PATH=OFF -DENABLE_POCL_BUILDING=OFF -DENABLE_CONFORMANCE=ON ..
RUN cd /opt/pocl/build && make -j$(nproc) && make install

# required for OpenCL-CTS to find 'cl_offline_compiler' in $PATH
RUN ln -s /opt/pocl/build/cl_offline_compiler.sh /usr/local/bin/cl_offline_compiler

# git branch/commit of OpenCL-CTS to build
#ARG CTS_CHECKOUT=a406b340913f622da089b00f284a597656c10239
# before printf with empty string:
#ARG CTS_CHECKOUT=5b5e43e1fb9af65874915a1a9d92831f5d65b017
# before printf with terminator ("foo\0foo")
ARG CTS_CHECKOUT=d379b58ab6853d7197c26044c95c2fae78445b3b
# header commit; must be usable with the OpenCL-CTS
ARG CL_H_CHECKOUT=1e193332d02e27e15812d24ff2a3a7a908eb92a3
# note that using any flag that enables 256bit or larger AVX registers can cause segfaults.
# this is because in C++11 a std::vector uses plain (unaligned) new/delete,
# these usually align to 16 bytes, which is not enough for 256bit vectors.
# Then std::vector<256bit type> will randomly segfault b/c of misalignment.
ARG CTS_BUILD_FLAGS="-O2 -march=x86-64-v2"

# checkout a specific version of the OpenCL headers; the one on Ubuntu is too old, the one in PoCL is too new
RUN cd /opt && git clone https://github.com/KhronosGroup/OpenCL-Headers && cd /opt/OpenCL-Headers && git checkout $CL_H_CHECKOUT
RUN mv /usr/include/CL/*.hpp /tmp ; rm /usr/include/CL/* ; mv /tmp/*.hpp /usr/include/CL/

# required for git cherry-pick
RUN git config --global user.email "you@gmail.com"
RUN git config --global user.name "Your Name"

# cherry-pick commit 2be73b2be1d87: fix generic_address_space, command_buffer_event_sync, test_compiler and images/test_1D_buffer
RUN cd /opt && git clone https://github.com/KhronosGroup/OpenCL-CTS.git && \
    cd /opt/OpenCL-CTS && git checkout -b level0_cts $CTS_CHECKOUT && \
    git cherry-pick 2be73b2be1d87d5573a32dd1260a1bffd98be211

RUN if [ "x$USE_WIMPY_MODE" = "x1" ]; then cd /opt/OpenCL-CTS && \
    sed -i 's|Select,select/test_select|Select,select/test_select -w|' test_conformance/opencl_conformance_tests_full.csv && \
    sed -i 's|Conversions,conversions/test_conversions|Conversions,conversions/test_conversions -w|' test_conformance/opencl_conformance_tests_full.csv && \
    sed -i 's|Math,math_brute_force/test_bruteforce|Math,math_brute_force/test_bruteforce -w|' test_conformance/opencl_conformance_tests_full.csv && \
    sed -i 's|Select,select/test_select --compilation-mode|Select,select/test_select -w --compilation-mode|' test_conformance/opencl_conformance_tests_full_spirv.csv && \
    sed -i 's|Conversions,conversions/test_conversions --compilation-mode|Conversions,conversions/test_conversions -w --compilation-mode|' test_conformance/opencl_conformance_tests_full_spirv.csv && \
    sed -i 's|Math,math_brute_force/test_bruteforce --compilation-mode|Math,math_brute_force/test_bruteforce -w --compilation-mode|' test_conformance/opencl_conformance_tests_full_spirv.csv && \
    echo "Fixed tests to wimpy mode" ; else echo "Not using wimpy mode" ; fi

RUN mkdir /opt/OpenCL-CTS/build
RUN cd /opt/OpenCL-CTS/build && \
    cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_C_FLAGS_RELWITHDEBINFO="$CTS_BUILD_FLAGS" \
    -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="$CTS_BUILD_FLAGS" -DD3D10_IS_SUPPORTED=0 -DD3D11_IS_SUPPORTED=0 -DGL_IS_SUPPORTED=1 -DGLES_IS_SUPPORTED=0 \
    -DCL_INCLUDE_DIR=/opt/OpenCL-Headers -DCL_LIB_DIR=/usr/lib/x86_64-linux-gnu -DOPENCL_LIBRARIES=OpenCL ..
RUN cd /opt/OpenCL-CTS/build && make -j$(nproc)

RUN /opt/OpenCL-CTS/test_conformance/spirv_new/assemble_spirv.py \
    -s /opt/OpenCL-CTS/test_conformance/spirv_new/spirv_asm \
    -o /opt/OpenCL-CTS/build/test_conformance/spirv_new/spirv_bin \
    -a /usr/bin/spirv-as -l /usr/bin/spirv-val

ENV POCL_CACHE_DIR=/tmp/POCL_CACHE
ENV POCL_DEVICES=level0

# runs both normal mode & spirv mode CTS tests
CMD cd /opt/OpenCL-CTS/build/test_conformance && clinfo && \
    ./run_conformance.py opencl_conformance_tests_full.csv CL_DEVICE_TYPE_GPU log=/srv && \
    ./run_conformance.py opencl_conformance_tests_full_spirv.csv CL_DEVICE_TYPE_GPU log=/srv
