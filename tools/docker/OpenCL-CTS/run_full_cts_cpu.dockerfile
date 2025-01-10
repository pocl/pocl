# Builds PoCL & OpenCL-CTS in a clean ubuntu environment, then
# runs the Full OpenCL-CTS in both normal and SPIR-V input mode,
# using the CTS's python runner script (run_conformance.py).
#
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
                   libllvmspirvlib-${LLVM_VERSION}-dev libzstd-dev libglew-dev libglut-dev
# required for run_conformance.py
RUN if [ ! -e /usr/bin/python ]; then ln -s /usr/bin/python3 /usr/bin/python ; fi

# GH username / org name where PoCL is cloned
ARG POCL_REMOTE=pocl
# git branch/commit of PoCL to build
ARG POCL_CHECKOUT=main
# git branch/commit of OpenCL-CTS to build
ARG CTS_CHECKOUT=899cbf5cd2bfc63c01fab9d71f7a6ec529091845
# C/C++ build flags
ARG POCL_BUILD_FLAGS="-O2 -march=native"
# note that using any flag that enables 256bit or larger AVX registers can cause segfaults.
# this is because in C++11 a std::vector uses plain (unaligned) new/delete,
# these usually align to 16 bytes, which is not enough for 256bit vectors.
# Then std::vector<256bit type> will randomly segfault b/c of misalignment.
ARG CTS_BUILD_FLAGS="-O2 -march=x86-64-v2"

RUN cd /opt && git clone https://github.com/$POCL_REMOTE/pocl.git && cd /opt/pocl && git checkout $POCL_CHECKOUT

RUN mkdir /opt/pocl/build && cd /opt/pocl/build
RUN cd /opt/pocl/build && \
    cmake -DWITH_LLVM_CONFIG=/usr/bin/llvm-config-${LLVM_VERSION} -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_C_FLAGS_RELWITHDEBINFO="$POCL_BUILD_FLAGS"  -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="$POCL_BUILD_FLAGS" \
    -DCMAKE_INSTALL_PREFIX=/usr -DENABLE_ICD=1 -DPOCL_ICD_ABSOLUTE_PATH=OFF -DENABLE_POCL_BUILDING=OFF -DENABLE_CONFORMANCE=ON ..
RUN cd /opt/pocl/build && make -j$(nproc) && make install

# required for OpenCL-CTS to find 'cl_offline_compiler' in $PATH
RUN ln -s /opt/pocl/build/cl_offline_compiler.sh /usr/local/bin/cl_offline_compiler

RUN cd /opt && git clone https://github.com/KhronosGroup/OpenCL-CTS.git && cd /opt/OpenCL-CTS && git checkout $CTS_CHECKOUT
RUN rm -rf /usr/include/CL
RUN mkdir /opt/OpenCL-CTS/build
RUN cd /opt/OpenCL-CTS/build && \
    cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_C_FLAGS_RELWITHDEBINFO="$CTS_BUILD_FLAGS" \
    -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="$CTS_BUILD_FLAGS" -DD3D10_IS_SUPPORTED=0 -DD3D11_IS_SUPPORTED=0 -DGL_IS_SUPPORTED=1 \
    -DGLES_IS_SUPPORTED=0 -DCL_INCLUDE_DIR=/home/sdp/pocl/include -DCL_LIB_DIR=/usr/lib/x86_64-linux-gnu \
    -DOPENCL_LIBRARIES=OpenCL -DCL_INCLUDE_DIR=/opt/pocl/include -DCL_LIB_DIR=/usr/lib/x86_64-linux-gnu ..
RUN cd /opt/OpenCL-CTS/build && make -j$(nproc)

RUN /opt/OpenCL-CTS/test_conformance/spirv_new/assemble_spirv.py \
    -s /opt/OpenCL-CTS/test_conformance/spirv_new/spirv_asm \
    -o /opt/OpenCL-CTS/build/test_conformance/spirv_new/spirv_bin \
    -a /usr/bin/spirv-as -l /usr/bin/spirv-val

ENV POCL_AFFINITY=1
ENV POCL_CACHE_DIR=/tmp/POCL_CACHE

# runs both normal mode & spirv mode CTS tests
CMD cd /opt/OpenCL-CTS/build/test_conformance && \
    ./run_conformance.py opencl_conformance_tests_full.csv CL_DEVICE_TYPE_CPU log=/srv && \
    ./run_conformance.py opencl_conformance_tests_full_spirv.csv CL_DEVICE_TYPE_CPU log=/srv
