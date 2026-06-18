# Docker image with PoCL (CPU driver), SPIRV-Tools, SPIRV-Translator, OpenCL-CTS
# Runs the full OpenCL-CTS in both OpenCL C and SPIR-V input mode,
# using the CTS's python runner script (run_conformance.py).
#
# supports multiple CPU architectures (ARM64, RISCV64, x86-64)
#
# NOTE: check & override if necessary the ARG variables defined below
#
# Build docker image:
#     docker build [--build-arg NAME=VALUE ...] --network=host -t pocl_full_cts -f run_full_cts.dockerfile .
# Create a container from the image:
#     docker create --name pocl_full_ct --memory=56G --memory-swap=56G --ulimit core=0 pocl_full_cts
# Run the container:
#     docker start pocl_full_ct
# View stdout of the container:
#     docker logs pocl_full_ct
#
# The full CTS logs () will be put into /srv in the container.
# If you want a copy of the full CTS logs, the simplest way is to bind-mount
# a host machine directory onto /srv in the container at create:
#     docker create ... --mount type=bind,src=/tmp/logs,dst=/srv ...
#
# alternatively, they can be accessed after run by creating an image of the container:
#     docker commit <ID of pocl_full_ct> -> creates a snapshot image of the container, prints the image ID
#     docker run --network=host --rm -it --entrypoint=/bin/bash --mount type=bind,src=/tmp/logs,dst=/mnt <image ID>
# .. in the running container, copy the logs from /srv to /mnt, then exit.

# CPU to build for. Will setup CPU flags accordingly
# Recognized values: jh7110, spacemit-k1, bcm2712, x86-64
ARG CPU=x86-64
# GCC version to use. Ubuntu 26 & Debian Forky both have GCC 15
ARG GCC=gcc-15
ARG GXX=g++-15

FROM riscv64/debian:forky AS base-jh7110
ENV CPU_FLAGS="-O2 -DNDEBUG -mabi=lp64d -mcpu=sifive-u74"

FROM riscv64/debian:forky AS base-spacemit-k1
ENV CPU_FLAGS="-O2 -DNDEBUG -mabi=lp64d -march=rv64imafdcv_zicbom_zicboz_zicntr_zicond_zicsr_zifencei_zihintpause_zihpm_zfh_zfhmin_zca_zcd_zba_zbb_zbc_zbs_zkt_zve32f_zve32x_zve64d_zve64f_zve64x_zvfh_zvfhmin_zvkt_sscofpmf_sstc_svinval_svnapot_svpbmt"

FROM arm64v8/debian:forky AS base-bcm2712
ENV CPU_FLAGS="-O2 -DNDEBUG -mcpu=cortex-a76"

FROM amd64/ubuntu:26.04 AS base-x86-64
ENV CPU_FLAGS="-O2 -DNDEBUG -march=native"

FROM base-${CPU} AS base
# inherit build args
ARG GCC
ARG GXX

# LLVM version to build PoCL with
ARG LLVM_VERSION=22
ARG LLVM_PREFIX=/usr/lib/llvm-${LLVM_VERSION}

#####################################################################################################################

RUN echo "Using GCC: ${GCC} G++ ${GXX} CPU_FLAGS ${CPU_FLAGS} LLVM_VERSION ${LLVM_VERSION}"

RUN apt update
RUN apt upgrade -y
RUN apt install -y python3-dev libpython3-dev build-essential ocl-icd-libopencl1 cmake git ${GCC} ${GXX} \
                   pkg-config libclang-${LLVM_VERSION}-dev clang-${LLVM_VERSION} llvm-${LLVM_VERSION} make \
                   ocl-icd-libopencl1 ocl-icd-dev ocl-icd-opencl-dev libhwloc-dev zlib1g zlib1g-dev clinfo dialog \
                   apt-utils libxml2-dev libclang-cpp${LLVM_VERSION}-dev libclang-cpp${LLVM_VERSION} \
                   libpolly-${LLVM_VERSION}-dev  llvm-${LLVM_VERSION}-dev \
                   libzstd-dev libglew-dev libglut-dev
# These are NOT available in Debian Forky for LLVM version 22 (only 20 & 21):
# llvm-spirv-${LLVM_VERSION}   libllvmspirvlib-${LLVM_VERSION}-dev

# available but disabled: spirv-tools

# available only in Ubuntu 26.04: libllvmlibc-${LLVM_VERSION}-dev
RUN if [ $(uname -m) == x86_64 ]; then apt install -y libllvmlibc-${LLVM_VERSION}-dev ; fi

# required for run_conformance.py
RUN if [ ! -e /usr/bin/python ]; then ln -s /usr/bin/python3 /usr/bin/python ; fi

#####################################################################################################################

# git branch/commit of OpenCL-Headers to checkout
ARG OCL_HEADERS_CHECKOUT=6fe718c31a45fe251
RUN cd /opt && git clone https://github.com/KhronosGroup/OpenCL-Headers.git
RUN cd /opt/OpenCL-Headers && git checkout $OCL_HEADERS_CHECKOUT

# git branch/commit of SPIRV-Headers to checkout
ARG SPIRV_HDR_CHECKOUT=1e770e7de8373a8dd
RUN cd /opt && git clone https://github.com/KhronosGroup/SPIRV-Headers.git
RUN cd /opt/SPIRV-Headers && git checkout $SPIRV_HDR_CHECKOUT

##### Optional; SPIRV-Tools 2026.1 available in official Forky repositories
# git branch/commit of SPIRV-Tools to checkout
# f700a727ea5df is just before commit 04d0b166dcd62e29509,
# which introduces FPRoundingMode decoration checks
# there is an unsolved bug with those decorations
#
ARG SPIRV_TOOLS_CHECKOUT=f700a727ea5df
RUN cd /opt && git clone https://github.com/KhronosGroup/SPIRV-Tools.git
RUN cd /opt/SPIRV-Tools && git checkout $SPIRV_TOOLS_CHECKOUT && python3 utils/git-sync-deps
RUN mkdir /opt/SPIRV-Tools/build && cd /opt/SPIRV-Tools/build && cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_C_COMPILER=${GCC} -DCMAKE_CXX_COMPILER=${GXX} \
    -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="${CPU_FLAGS}" -DSPIRV_WERROR=0 -DCMAKE_INSTALL_PREFIX=/usr -DSPIRV_SKIP_TESTS=1 ..
RUN cd /opt/SPIRV-Tools/build && make -j$(nproc) install

##### Optional; LLVM-SPIRV 22 translator NOT available in official Forky repositories
# git branch/commit of SPIRV-LLVM-Translator to checkout
ARG SPIRV_TRANS_CHECKOUT=llvm_release_${LLVM_VERSION}0
RUN cd /opt && git clone https://github.com/KhronosGroup/SPIRV-LLVM-Translator.git SPIRV-Trans
RUN cd /opt/SPIRV-Trans && git checkout $SPIRV_TRANS_CHECKOUT
RUN mkdir /opt/SPIRV-Trans/build && cd /opt/SPIRV-Trans/build && cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_C_COMPILER=${GCC} -DCMAKE_CXX_COMPILER=${GXX} \
    -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="${CPU_FLAGS}" -DLLVM_EXTERNAL_SPIRV_HEADERS_SOURCE_DIR=/opt/SPIRV-Headers \
    -DCMAKE_INSTALL_PREFIX=${LLVM_PREFIX} -DLLVM_DIR=${LLVM_PREFIX}/lib/cmake ..
RUN cd /opt/SPIRV-Trans/build && make -j$(nproc) install

#####################################################################################################################

# GH username / org name where PoCL is cloned
ARG POCL_REMOTE=pocl
# git branch/commit of PoCL to build
ARG POCL_CHECKOUT=bd3223040c0c62c461

RUN cd /opt && git clone https://github.com/$POCL_REMOTE/pocl.git && cd /opt/pocl && git checkout $POCL_CHECKOUT

RUN mkdir /opt/pocl/build && cd /opt/pocl/build

RUN cd /opt/pocl/build && \
    cmake -DWITH_LLVM_CONFIG=/usr/bin/llvm-config-${LLVM_VERSION} -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_C_COMPILER=${GCC} -DCMAKE_CXX_COMPILER=${GXX} \
    -DCMAKE_C_FLAGS_RELWITHDEBINFO="${CPU_FLAGS}"  -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="${CPU_FLAGS}" \
    -DCMAKE_INSTALL_PREFIX=/usr -DENABLE_ICD=1 -DPOCL_ICD_ABSOLUTE_PATH=OFF -DENABLE_POCL_BUILDING=OFF -DENABLE_CONFORMANCE=ON \
    -DENABLE_TESTS=0 -DENABLE_EXAMPLES=0 ..
RUN cd /opt/pocl/build && make -j$(nproc) && make install

#####################################################################################################################

# GH username / org name where OpenCL-CTS is cloned
ARG CTS_REMOTE=franz
# git branch/commit of OpenCL-CTS to build
# this tag is v2026-03-25-00 + commit b8eca0fe03eadf6b458
ARG CTS_CHECKOUT=tag_03_25

# required for OpenCL-CTS to find 'cl_offline_compiler' in $PATH
RUN ln -s /opt/pocl/build/cl_offline_compiler.sh /usr/local/bin/cl_offline_compiler

RUN cd /opt && git clone https://github.com/$CTS_REMOTE/OpenCL-CTS.git && cd /opt/OpenCL-CTS && git checkout $CTS_CHECKOUT
# RUN rm -rf /usr/include/CL
RUN mkdir /opt/OpenCL-CTS/build

RUN cd /opt/OpenCL-CTS/build && \
    cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_C_COMPILER=${GCC} -DCMAKE_CXX_COMPILER=${GXX} \
    -DCMAKE_C_FLAGS_RELWITHDEBINFO="${CPU_FLAGS}" \
    -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="${CPU_FLAGS}" \
    -DD3D10_IS_SUPPORTED=0 -DD3D11_IS_SUPPORTED=0 \
    -DGL_IS_SUPPORTED=1 -DGLES_IS_SUPPORTED=0 \
    -DCL_LIB_DIR=/usr/lib/$(gcc -print-multiarch) \
    -DSPIRV_INCLUDE_DIR=/opt/SPIRV-Headers \
    -DCL_INCLUDE_DIR=/opt/OpenCL-Headers \
    -DOPENCL_LIBRARIES=OpenCL \
    ..
RUN cd /opt/OpenCL-CTS/build && make -j$(nproc)

#RUN cd /opt/OpenCL-CTS/build/test_conformance && sed -i 's|cache-path .|cache-path /srv/CTS_SPIRVCACHE|' opencl_conformance_tests_full_spirv.csv

ENV POCL_CACHE_DIR=/tmp/POCL_CACHE
ENV CL_DEVICE_TYPE=cpu

# runs both normal mode & spirv mode CTS tests
CMD cd /opt/OpenCL-CTS/build/test_conformance && \
    ./run_conformance.py opencl_conformance_tests_full.csv CL_DEVICE_TYPE_CPU log=/srv && \
    ./run_conformance.py opencl_conformance_tests_full_spirv.csv CL_DEVICE_TYPE_CPU log=/srv
