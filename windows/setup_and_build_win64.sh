#/bin/bash

# This script requires following software preinstalled:
# * Git + git bash (default installer settings are fine, make sure its in your PATH)
# * Python 2.7
# * Cmake 2.8 or later (make sure it will be added to PATH during installation)
# * Visual Studio Community edition 2013

# no spaces in this path please
export POCLBUILDROOT=/c/pocl-playground

mkdir $POCLBUILDROOT
cd $POCLBUILDROOT

# Get external libs
curl ftp://sourceware.org/pub/pthreads-win32/pthreads-w32-2-9-1-release.zip -O
curl http://www.open-mpi.org/software/hwloc/v1.10/downloads/hwloc-win64-build-1.10.0.zip -O
unzip hwloc-win64-build-1.10.0.zip
unzip pthreads-w32-2-9-1-release.zip -d pthreads-win32-full
cp -r pthreads-win32-full/Pre-built.2 pthreads-win32

# Build llvm
cd $POCLBUILDROOT
git clone --single-branch https://github.com/llvm-mirror/llvm -b release_36
cd llvm/tools
git clone --single-branch https://github.com/llvm-mirror/clang.git -b release_36
mkdir $POCLBUILDROOT/llvm-build
cd $POCLBUILDROOT/llvm-build
cmake -G "Visual Studio 12 Win64" ../llvm
cmake --build . --config MinSizeRel

# Build pocl
cd $POCLBUILDROOT
git clone https://github.com/pocl/pocl.git
mkdir $POCLBUILDROOT/pocl-build
cd $POCLBUILDROOT/pocl-build
export PATH=$PATH:$POCLBUILDROOT/llvm-build/MinSizeRel/bin
Hwloc_ROOT=../hwloc-win64-build-1.10.0/ Pthreads_ROOT=../pthreads-win32/ cmake -DSTATIC_LLVM:BOOL=ON -DDEFAULT_ENABLE_ICD:BOOL=OFF -DCMAKE_INSTALL_PREFIX:PATH=$PWD/../install-pocl -G "Visual Studio 12 Win64" ../pocl/
cmake --build . --config MinSizeRel

## Run test suite
# export PATH=$PATH:$POCLBUILDROOT/pocl-build/lib/CL/MinSizeRel:$POCLBUILDROOT/pocl-build/lib/llvmopencl/MinSizeRel:$POCLBUILDROOT/pocl-build/lib/poclu/MinSizeRel:$POCLBUILDROOT/hwloc-win64-build-1.10.0/bin:$POCLBUILDROOT/pthreads-win32/dll/x64
# cd $POCLBUILDROOT/pocl-build
# ctest -j8
