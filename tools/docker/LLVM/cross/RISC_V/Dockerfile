# cross-compiles a LLVM on X86-64 for RISC-V
# How to use:
# 1) change C/CXX flags in cross.cmake to match the target CPU
# 2) create a compressed tar (named SBC_ROOT.tar.xz) with contents of the target
#    board's filesystem with ROOTFS as top dir (ROOTFS/bin, ROOTFS/usr, ... must exist)
# 3) create a compressed tar (llvm-project.tar.xz) with LLVM's git clone, with
#    "llvm-project" as top dir
# 4) (optional) change the FROM directive if needed (and the pkg versions in apt
#    commands), currently uses debian bookworm as host system
# 5) put all files (cross.cmake & tars) into the same directory as this dockerfile,
#    then run docker build

FROM amd64/debian:bookworm@sha256:27586f4609433f2f49a9157405b473c62c3cb28a581c413393975b4e8496d0ab

# number of threads to use for building (make -jX)
ARG JOBS=16
# max number of simultaneous link jobs. These require lots of memory
ARG LINK_JOBS=4
# LLVM version to use
ARG BRANCH=17

LABEL branch=release_${BRANCH} distro=Debian version=1.0
RUN echo "using Branch: release_${BRANCH} Jobs: ${JOBS} Link-jobs: ${LINK_JOBS}" && sleep 10

USER root

ENV TERM dumb
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update
RUN apt upgrade -y
RUN apt install -y build-essential nano cmake git pkg-config make ninja-build zlib1g zlib1g-dev python3-dev libpython3-dev
RUN apt install -y gcc-riscv64-linux-gnu g++-riscv64-linux-gnu cpp-riscv64-linux-gnu binutils-riscv64-linux-gnu

# where the target's root filesystem lives
ENV PATH_TO_SBC_ROOT=/mnt/ROOTFS
# LLVM git repo
ENV LLVM_SOURCE_DIR=/home/llvm/source/llvm-project
ENV LLVM_SOURCE_DIR_PARENT=/home/llvm/source
# cmake build dir
ENV BUILD=/home/llvm/build
# LLVM install dir
ENV INSTALL=/opt/LLVM_${BRANCH}0
# cmake toolchain file
ENV CROSS=/tmp/cross.cmake

### prepare LLVM sources
RUN mkdir -p $LLVM_SOURCE_DIR_PARENT
ADD llvm-project.tar.xz $LLVM_SOURCE_DIR_PARENT

RUN cd $LLVM_SOURCE_DIR && rm -rf .git/worktrees && git reset --hard HEAD && git clean -f -d -x && git fetch
RUN cd $LLVM_SOURCE_DIR && git checkout -b release_${BRANCH} origin/release/${BRANCH}.x && git branch

### unpack SBC root
ADD SBC_ROOT.tar.xz /mnt/

# cross-compile LLVM for target
COPY cross.cmake /tmp/

RUN mkdir -p $BUILD/cross && cd $BUILD/cross

RUN cd $BUILD/cross && cmake -G Ninja -DCMAKE_CROSSCOMPILING=1 -DCMAKE_TOOLCHAIN_FILE=$CROSS -DCMAKE_INSTALL_PREFIX=$INSTALL \
   -DBUILD_SHARED_LIBS=ON -DLLVM_DEFAULT_TARGET_TRIPLE=riscv64-linux-gnu -DLLVM_TARGET_ARCH=riscv64 -DLLVM_TARGETS_TO_BUILD=RISCV \
   -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_ENABLE_THREADS=OFF \
   -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang" \
   -DLLVM_PARALLEL_LINK_JOBS=${LINK_JOBS} \
   $LLVM_SOURCE_DIR/llvm

########## currently build does not finish, requires manually adding -latomic in several places in build.ninja
RUN cd $BUILD/cross && ninja -j ${JOBS} && ninja install && ninja clean

CMD /bin/bash
