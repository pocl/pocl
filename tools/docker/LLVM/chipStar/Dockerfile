# builds chipStar's clone of LLVM (branch "chipStar-llvm-${LLVM_VERSION}") with X86,NVPTX,SPIRV targets

########## UBUNTU stage 2 - install LLVM
FROM amd64/ubuntu:22.04@sha256:3d1556a8a18cf5307b121e0a98e93f1ddf1f3f8e092f1fddfd941254785b95d7

ARG LLVM_VERSION=17
# chipStar branch to use
ARG LLVM_BRANCH="chipStar-llvm-${LLVM_VERSION}"
# SPIRV LLVM translator branch to use
ARG TRANS_BRANCH="chipStar-llvm-${LLVM_VERSION}"

ARG MAX_JOBS=16
ARG PARALLEL_LINK_JOBS=4

ARG CPU_FLAGS="-march=x86-64-v3 -mtune=haswell"
ARG CMAKE_B_FLAGS="-Os -ggdb3 -ggnu-pubnames -gdwarf-4 ${CPU_FLAGS}"

LABEL vendor=pocl distro=Ubuntu version=1.0
ENV DEBIAN_FRONTEND=noninteractive

USER root

RUN echo "Using Branch: ${LLVM_BRANCH} Jobs: ${MAX_JOBS}/${PARALLEL_LINK_JOBS} CMake Flags: ${CMAKE_B_FLAGS}" && sleep 10

ENV TERM dumb
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update
RUN apt upgrade -y
RUN apt install -y build-essential nano cmake git pkg-config make ninja-build zlib1g zlib1g-dev python3-dev libpython3-dev gcc-12 g++-12 libstdc++-12-dev coreutils binutils tar sed bzip2 grep gzip libxml2-dev libzstd-dev patchelf


###########################################################################
############################ build SPIRV_Tools ############################

RUN mkdir -p /home/spirv_tools/build
RUN cd /home/spirv_tools && git clone https://github.com/KhronosGroup/SPIRV-Tools.git source
RUN cd /home/spirv_tools/source && python3 utils/git-sync-deps

RUN cd /home/spirv_tools/build && cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo     \
      -DCMAKE_C_COMPILER=/usr/bin/gcc-12 -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
      -DCMAKE_C_FLAGS_RELWITHDEBINFO="${CMAKE_B_FLAGS}"   \
      -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="${CMAKE_B_FLAGS}" \
      -DSPIRV_SKIP_TESTS=ON -DCMAKE_INSTALL_PREFIX=/usr   \
      -DSKIP_SPIRV_TOOLS_INSTALL=OFF /home/spirv_tools/source

RUN cd /home/spirv_tools/build && make -j${MAX_JOBS} install && make clean

############### build LLVM with LLVM-SPIRV translator, Asserts, RTTI and GDB flags

RUN mkdir -p /home/llvm_build/build ; mkdir -p /home/llvm_build/source

RUN cd /home/llvm_build/source ; git clone --depth=1 --single-branch --branch=${LLVM_BRANCH} https://github.com/CHIP-SPV/llvm-project.git llvm-project

RUN cd /home/llvm_build/source/llvm-project && rm -rf .git/worktrees .github/workflows && git branch

RUN cd /home/llvm_build/source/llvm-project/llvm/tools && \
    git clone --depth=1 --single-branch --branch=${TRANS_BRANCH} https://github.com/CHIP-SPV/SPIRV-LLVM-Translator.git

RUN cd /home/llvm_build/build ; cmake -G Ninja \
     -DCMAKE_C_COMPILER=/usr/bin/gcc-12 -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
     -DCMAKE_INSTALL_PREFIX=/home/LLVM_${LLVM_VERSION} \
     -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
     -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=gold -Wl,--gdb-index" \
     -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=gold -Wl,--gdb-index" \
     -DLLVM_ENABLE_PROJECTS=clang \
     -DLLVM_PARALLEL_LINK_JOBS=${PARALLEL_LINK_JOBS} \
     -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=SPIRV \
     -DBUILD_SHARED_LIBS=ON \
     -DLLVM_ENABLE_RTTI=ON -DLLVM_ENABLE_ASSERTIONS=ON \
     -DLLVM_ENABLE_EH=ON -DLLVM_ENABLE_Z3_SOLVER=OFF \
     -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_C_FLAGS_RELWITHDEBINFO="${CMAKE_B_FLAGS}" -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="${CMAKE_B_FLAGS}" \
     /home/llvm_build/source/llvm-project/llvm

RUN cd /home/llvm_build/build && ninja -j ${MAX_JOBS} && ninja llvm-spirv && ninja install && ninja clean

RUN patchelf --set-rpath '$ORIGIN/../lib' /home/LLVM_${LLVM_VERSION}/bin/llvm-spirv


CMD /bin/bash
