#!/bin/bash
set -e -x

cd /io
POCL_TAG=`git describe --tags`
export MAKEFLAGS="-j32"

mkdir -p /deps
cd /deps
LICENSE_DIR=/deps/pocl-$POCL_TAG/share/doc/pocl
mkdir -p $LICENSE_DIR

yum install -y git yum libxml2-devel xz

# Need ruby for ocl-icd
curl -L -O http://cache.ruby-lang.org/pub/ruby/2.1/ruby-2.1.2.tar.gz
tar -xf ruby-2.1.2.tar.gz
pushd ruby-2.1.2
./configure
make
make install
popd

# OCL ICD loader
git clone --branch v2.2.12 https://github.com/OCL-dev/ocl-icd
pushd ocl-icd
autoreconf -i
chmod +x configure
./configure --prefix=/usr
make
make install
# this is in pyopencl
cp COPYING $LICENSE_DIR/OCL_ICD.COPYING
popd

# libhwloc for pocl
curl -L -O https://download.open-mpi.org/release/hwloc/v2.0/hwloc-2.0.3.tar.gz
tar -xf hwloc-2.0.3.tar.gz
pushd hwloc-2.0.3
CFLAGS="-fPIC" CXXFLAGS="-fPIC" LDFLAGS="-fPIC" ./configure \
    --disable-cairo \
    --disable-opencl \
    --disable-cuda \
    --disable-nvml \
    --disable-gl \
    --disable-libudev \
    --disable-shared \
    --disable-libxml2
make
make install
cp COPYING $LICENSE_DIR/HWLOC.COPYING
#cp /usr/share/doc/libxml2-devel-*/Copyright $LICENSE_DIR/libxml2.COPYING
popd

# newer cmake for LLVM
/opt/python/cp37-cp37m/bin/pip install cmake
export PATH="/opt/python/cp37-cp37m/lib/python3.7/site-packages/cmake/data/bin/:${PATH}"

# LLVM for pocl
LLVM_VERSION=8.0.1
#curl -L -O http://releases.llvm.org/${LLVM_VERSION}/llvm-${LLVM_VERSION}.src.tar.xz
curl -L -O https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VERSION}/llvm-${LLVM_VERSION}.src.tar.xz
unxz llvm-${LLVM_VERSION}.src.tar.xz
tar -xf llvm-${LLVM_VERSION}.src.tar
pushd llvm-${LLVM_VERSION}.src
mkdir -p build
pushd build
cmake -DPYTHON_EXECUTABLE=/opt/python/cp37-cp37m/bin/python \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DLLVM_TARGETS_TO_BUILD=host \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_RTTI=ON \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DLLVM_INCLUDE_GO_TESTS=OFF \
    -DLLVM_INCLUDE_UTILS=ON \
    -DLLVM_INCLUDE_DOCS=OFF \
    -DLLVM_INCLUDE_EXAMPLES=OFF \
    -DLLVM_ENABLE_TERMINFO=OFF \
    -DLLVM_ENABLE_LIBXML2=OFF \
    -DLLVM_ENABLE_ZLIB=OFF \
    -DLLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN=ON \
    ..

make
make install
popd
cp LICENSE.TXT $LICENSE_DIR/LLVM_LICENSE.txt
popd

# clang for pocl
#curl -L -O http://releases.llvm.org/${LLVM_VERSION}/cfe-${LLVM_VERSION}.src.tar.xz
curl -L -O https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VERSION}/cfe-${LLVM_VERSION}.src.tar.xz
unxz cfe-${LLVM_VERSION}.src.tar.xz
tar -xf cfe-${LLVM_VERSION}.src.tar
pushd cfe-${LLVM_VERSION}.src
mkdir -p build
pushd build
cmake \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DCMAKE_PREFIX_PATH=/usr/local \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_RTTI=ON \
    -DCLANG_INCLUDE_TESTS=OFF \
    -DCLANG_INCLUDE_DOCS=OFF \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DLLVM_INCLUDE_DOCS=OFF \
    -DLLVM_ENABLE_LIBXML2=OFF \
    -DLLVM_ENABLE_ZLIB=OFF \
    -DLLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN=ON \
    ..
make
make install
popd
cp LICENSE.TXT $LICENSE_DIR/clang_LICENSE.txt
popd

# lld for pocl
#curl -L -O http://releases.llvm.org/6.0.1/lld-6.0.1.src.tar.xz
#unxz lld-6.0.1.src.tar.xz
#tar -xf lld-6.0.1.src.tar
#pushd lld-6.0.1.src
#mkdir -p build
#pushd build
#cmake \
#  -DCMAKE_INSTALL_PREFIX=/usr/local \
#  -DCMAKE_PREFIX_PATH=/usr/local \
#  -DCMAKE_BUILD_TYPE=Release \
#..
#make -j16
#make install
#popd
#cp LICENSE.TXT /deps/licenses/pocl/lld_LICENSE.txt
#popd

mkdir -p pocl_build
pushd pocl_build

export LDFLAGS="-Wl,--exclude-libs,ALL"
EXTRA_HOST_LD_FLAGS="$EXTRA_HOST_LD_FLAGS -nodefaultlibs"

cmake -DCMAKE_C_FLAGS="$EXTRA_FLAGS" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="$EXTRA_FLAGS" \
    -DINSTALL_OPENCL_HEADERS="off" \
    -DKERNELLIB_HOST_CPU_VARIANTS=distro \
    -DENABLE_ICD=on \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DCMAKE_INSTALL_PREFIX=/deps/pocl-$POCL_TAG/ \
    -DEXTRA_HOST_LD_FLAGS="${EXTRA_HOST_LD_FLAGS}" \
    /io

make -j16
make install
popd

pushd /io
cp COPYING $LICENSE_DIR/POCL.COPYING
popd

pushd /deps/
echo "libpocl.so" > pocl-$POCL_TAG/etc/OpenCL/vendors/pocl.icd
tar -czvf pocl-$POCL_TAG-x86_64-linux-gnu.tar.gz pocl-$POCL_TAG
mkdir -p /io/release
cp pocl-$POCL_TAG-x86_64-linux-gnu.tar.gz /io/release/

